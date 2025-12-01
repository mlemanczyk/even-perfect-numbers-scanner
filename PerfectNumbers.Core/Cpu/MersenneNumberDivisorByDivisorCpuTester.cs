using System.Runtime.CompilerServices;
using PerfectNumbers.Core.Gpu;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core.Cpu;

public sealed class MersenneNumberDivisorByDivisorCpuTester(ComputationDevice orderDevice = ComputationDevice.Hybrid) : IMersenneNumberDivisorByDivisorTester
{
	private ulong _divisorLimit;
	private int _batchSize = 1_024;
	private ulong _minK = 1UL;

	private readonly TryCycleLengthDelegate _tryCalculateCycleLengthForExponent = orderDevice switch
	{
		ComputationDevice.Gpu => TryCalculateCycleLengthForExponentGpu,
		ComputationDevice.Hybrid => TryCalculateCycleLengthForExponentHybrid,
		_ => TryCalculateCycleLengthForExponentCpu,
	};

	private readonly CalculateCycleLengthDelegate _calculateCycleLength = orderDevice switch
	{
		ComputationDevice.Gpu => CalculateCycleLengthGpu,
		ComputationDevice.Hybrid => CalculateCycleLengthHybrid,
		_ => CalculateCycleLengthCpu,
	};

	private delegate bool TryCycleLengthDelegate(
		ulong divisor,
		PrimeOrderCalculatorAccelerator gpu,
		ulong exponent,
		in MontgomeryDivisorData divisorData,
		out ulong cycleLength,
		out bool primeOrderFailed);

	private delegate ulong CalculateCycleLengthDelegate(
		ulong divisor,
		PrimeOrderCalculatorAccelerator gpu,
		in MontgomeryDivisorData divisorData,
		bool skipPrimeOrderHeuristic);

	[ThreadStatic]
	private static GpuUInt128WorkSet _divisorScanGpuWorkSet;

	private struct GpuUInt128WorkSet
	{
		public GpuUInt128 Step;
		public GpuUInt128 Divisor;
		public GpuUInt128 Limit;
	}

	public int BatchSize
	{
		get => _batchSize;
		set => _batchSize = Math.Max(1, value);
	}

	public ulong MinK
	{
		get => _minK;
		set => _minK = value < 1UL ? 1UL : value;
	}

	public void ConfigureFromMaxPrime(ulong maxPrime)
	{
		_divisorLimit = ComputeDivisorLimitFromMaxPrime(maxPrime);
	}

	public ulong DivisorLimit
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		get
		{
			return _divisorLimit;
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public ulong GetAllowedMaxDivisor(ulong prime) => ComputeAllowedMaxDivisor(prime, _divisorLimit);

	public bool IsPrime(PrimeOrderCalculatorAccelerator gpu, ulong prime, out bool divisorsExhausted, out ulong divisor)
	{
		ulong allowedMax = ComputeAllowedMaxDivisor(prime, _divisorLimit);

		// The CPU by-divisor run always hands us primes with enormous divisor limits, so the fallback below never executes.
		// if (allowedMax < 3UL)
		// {
		//     // EvenPerfectBitScanner routes primes below the small-divisor cutoff to the GPU path, so the CPU path still sees
		//     // trivial candidates during targeted tests. Short-circuit here to keep those runs aligned with the production flow.
		//     divisorsExhausted = true;
		//     return true;
		// }

		bool processedAll;

		bool composite = CheckDivisors(
			gpu,
			prime,
			allowedMax,
			_minK,
			out processedAll,
			out divisor);

		if (composite)
		{
			divisorsExhausted = true;
			return false;
		}

		divisorsExhausted = processedAll || composite;
		divisor = 0UL;
		return true;
	}

	public void PrepareCandidates(in ReadOnlySpan<ulong> primes, Span<ulong> allowedMaxValues)
	{
		ulong divisorLimit = _divisorLimit;
		int length = primes.Length;
		for (int index = 0; index < length; index++)
		{
			allowedMaxValues[index] = ComputeAllowedMaxDivisor(primes[index], divisorLimit);
		}
	}

	public IMersenneNumberDivisorByDivisorTester.IDivisorScanSession CreateDivisorSession(PrimeOrderCalculatorAccelerator gpu)
	{
		MersenneCpuDivisorScanSession? session = ThreadStaticPools.RentMersenneCpuDivisorSession();
		
		if (session is not null)
		{
			session.Configure(gpu, orderDevice);
			return session;
		}

		return new MersenneCpuDivisorScanSession(gpu, orderDevice);
	}

	private bool CheckDivisors(
		PrimeOrderCalculatorAccelerator gpu,
		ulong prime,
		ulong allowedMax,
		ulong minK,
		out bool processedAll,
		out ulong foundDivisor)
	{
		foundDivisor = 0UL;
		processedAll = true;

		// The EvenPerfectBitScanner feeds primes >= 138,000,000 here, so allowedMax >= 3 in production runs.
		// Keeping the guard commented out documents the reasoning for benchmarks and tests.
		// if (allowedMax < 3UL)
		// {
		//     return false;
		// }

		ref GpuUInt128WorkSet workSet = ref _divisorScanGpuWorkSet;

		ref GpuUInt128 step = ref workSet.Step;
		step.High = 0UL;
		step.Low = prime;
		step.ShiftLeft(1);

		ref GpuUInt128 limit = ref workSet.Limit;
		limit.High = 0UL;
		limit.Low = allowedMax;

		ref GpuUInt128 divisor = ref workSet.Divisor;
		divisor.High = step.High;
		divisor.Low = step.Low;
		divisor.Add(1UL);
		if (minK > 1UL)
		{
			GpuUInt128 offset = new(minK - 1UL);
			divisor = step * offset;
			divisor.Add(1UL);
		}

		if (divisor.CompareTo(limit) > 0)
		{
			processedAll = true;
			foundDivisor = 0UL;
			return false;
		}

		// Intentionally recomputes factorizations without a per-thread cache.
		// The previous factor cache recorded virtually no hits and only slowed down the scan.

		ulong stepHigh = step.High;
		ulong stepLow = step.Low;
		ulong limitHigh = limit.High;
		ulong divisorLow = divisor.Low;

		// Keep the divisibility filters aligned with the divisor-cycle generator so the
		// CPU path never requests cycles that were skipped during cache creation.
		LastDigit lastDigit = (prime & 3UL) == 3UL ? LastDigit.Seven : LastDigit.One;
		ushort decimalMask = DivisorGenerator.GetDecimalMask(lastDigit);

		byte step10;
		byte step8;
		byte step3;
		byte step7;
		byte step11;

		byte remainder10;
		byte remainder8;
		byte remainder3;
		byte remainder7;
		byte remainder11;

		if (stepHigh == 0UL && limitHigh == 0UL)
		{
			step10 = (byte)(stepLow % 10UL);
			step8 = (byte)(stepLow & 7UL);
			step3 = (byte)(stepLow % 3UL);
			step7 = (byte)(stepLow % 7UL);
			step11 = (byte)(stepLow % 11UL);

			ulong maxK = allowedMax > 0UL ? (allowedMax - 1UL) / stepLow : 0UL;
			if (maxK == 0UL)
			{
				processedAll = true;
				foundDivisor = 0UL;
				return false;
			}

			ulong startK = minK < 1UL ? 1UL : minK;
			bool processedTop = true;
			bool processedBottom = true;
			bool composite = false;

			if (startK <= maxK)
			{
				composite = CheckDivisors64Range(
					gpu,
					prime,
					stepLow,
					allowedMax,
					startK,
					maxK,
					decimalMask,
					step10,
					step8,
					step3,
					step7,
					step11,
					out processedTop,
					out foundDivisor);
				if (composite)
				{
					processedAll = true;
					return true;
				}
			}

			ulong lowerEnd = startK > 1UL ? startK - 1UL : 0UL;
			if (lowerEnd >= 1UL)
			{
				lowerEnd = Math.Min(lowerEnd, maxK);
				composite = CheckDivisors64Range(
					gpu,
					prime,
					stepLow,
					allowedMax,
					1UL,
					lowerEnd,
					decimalMask,
					step10,
					step8,
					step3,
					step7,
					step11,
					out processedBottom,
					out foundDivisor);
			}

			processedAll = processedTop && processedBottom;
			return composite;
		}

		ulong divisorHigh = divisor.High;
		step10 = (byte)((((stepHigh % 10UL) * 6UL) + (stepLow % 10UL)) % 10UL);
		step8 = (byte)(stepLow % 8UL);
		step3 = (byte)(((stepHigh % 3UL) + (stepLow % 3UL)) % 3UL);
		step7 = (byte)((((stepHigh % 7UL) * 2UL) + (stepLow % 7UL)) % 7UL);
		step11 = (byte)((((stepHigh % 11UL) * 5UL) + (stepLow % 11UL)) % 11UL);

		remainder10 = (byte)((((divisorHigh % 10UL) * 6UL) + (divisorLow % 10UL)) % 10UL);
		remainder8 = (byte)(divisorLow % 8UL);
		remainder3 = (byte)(((divisorHigh % 3UL) + (divisorLow % 3UL)) % 3UL);
		remainder7 = (byte)((((divisorHigh % 7UL) * 2UL) + (divisorLow % 7UL)) % 7UL);
		remainder11 = (byte)((((divisorHigh % 11UL) * 5UL) + (divisorLow % 11UL)) % 11UL);

		var divisorPool = MontgomeryDivisorDataPool.Shared;
		while (divisor.CompareTo(limit) <= 0)
		{
			bool passesSmallModuli = remainder3 != 0 && remainder7 != 0 && remainder11 != 0;
			if (passesSmallModuli && (remainder8 == 1 || remainder8 == 7) && ((decimalMask >> remainder10) & 1) != 0)
			{
				ulong candidate = divisor.Low;
				MontgomeryDivisorData divisorData = divisorPool.FromModulus(candidate);
				ulong divisorCycle;
				// Divisors generated from 2 * k * p + 1 exceed the small-cycle snapshot when p >= 138,000,000, so the short path below never runs.
				if (!_tryCalculateCycleLengthForExponent(
						candidate,
						gpu,
						prime,
						divisorData,
						out ulong computedCycle,
						out bool primeOrderFailed) || computedCycle == 0UL)
				{
					// Divisors produced by 2 * k * p + 1 always exceed PerfectNumberConstants.MaxQForDivisorCycles
					// for the exponents scanned here, so skip the unused cache fallback and compute directly.
					divisorCycle = _calculateCycleLength(
						candidate,
						gpu,
						divisorData,
						skipPrimeOrderHeuristic: primeOrderFailed);
				}
				else
				{
					divisorCycle = computedCycle;
				}

				divisorPool.Return(divisorData);
				if (divisorCycle == prime)
				{
					// A cycle equal to the tested exponent (which is prime in this path) guarantees that the candidate divides
					// the corresponding Mersenne number because the order of 2 modulo the divisor is exactly p.
					foundDivisor = candidate;
					processedAll = true;
					return true;
				}

				if (divisorCycle == 0UL)
				{
					Console.WriteLine($"Divisor cycle was not calculated for {prime}");
				}
			}

			divisor.Add(step);
			remainder10 = AddMod10(remainder10, step10);
			remainder8 = AddMod8(remainder8, step8);
			remainder3 = AddMod3(remainder3, step3);
			remainder7 = AddMod7(remainder7, step7);
			remainder11 = AddMod11(remainder11, step11);
		}

		processedAll = true;
		foundDivisor = 0UL;
		return false;
	}

	private bool CheckDivisors64Range(
		PrimeOrderCalculatorAccelerator gpu,
		ulong prime,
		ulong step,
		ulong allowedMax,
		ulong startK,
		ulong endK,
		ushort decimalMask,
		byte step10,
		byte step8,
		byte step3,
		byte step7,
		byte step11,
		out bool processedAll,
		out ulong foundDivisor)
	{
		if (startK < 1UL || endK < startK)
		{
			processedAll = true;
			foundDivisor = 0UL;
			return false;
		}

		UInt128 step128 = step;
		UInt128 allowedMax128 = allowedMax;
		UInt128 startDivisor128 = (step128 * startK) + UInt128.One;
		if (startDivisor128 > allowedMax128)
		{
			processedAll = true;
			foundDivisor = 0UL;
			return false;
		}

		UInt128 rangeLimit128 = (step128 * endK) + UInt128.One;
		if (rangeLimit128 > allowedMax128)
		{
			rangeLimit128 = allowedMax128;
		}

		ulong startDivisor = (ulong)startDivisor128;
		ulong rangeLimit = (ulong)rangeLimit128;

		byte remainder10 = (byte)(startDivisor % 10UL);
		byte remainder8 = (byte)(startDivisor & 7UL);
		byte remainder3 = (byte)(startDivisor % 3UL);
		byte remainder7 = (byte)(startDivisor % 7UL);
		byte remainder11 = (byte)(startDivisor % 11UL);

		return CheckDivisors64(
			gpu,
			prime,
			step,
			rangeLimit,
			startDivisor,
			decimalMask,
			step10,
			step8,
			step3,
			step7,
			step11,
			remainder10,
			remainder8,
			remainder3,
			remainder7,
			remainder11,
			out processedAll,
			out foundDivisor);
	}

	private bool CheckDivisors64(
		PrimeOrderCalculatorAccelerator gpu,
		ulong prime,
		ulong step,
		ulong limit,
		ulong divisor,
		ushort decimalMask,
		byte step10,
		byte step8,
		byte step3,
		byte step7,
		byte step11,
		byte remainder10,
		byte remainder8,
		byte remainder3,
		byte remainder7,
		byte remainder11,
		out bool processedAll,
		out ulong foundDivisor)
	{
		bool canAdvance = step <= limit;
		Queue<MontgomeryDivisorData> divisorPool = MontgomeryDivisorDataPool.Shared;
		if (!canAdvance)
		{
			if (divisor <= limit)
			{
				if (remainder3 != 0 && remainder7 != 0 && remainder11 != 0 && (remainder8 == 1 || remainder8 == 7) && ((decimalMask >> remainder10) & 1) != 0)
				{
					MontgomeryDivisorData divisorData = divisorPool.FromModulus(divisor);
					ulong divisorCycle;
					if (!_tryCalculateCycleLengthForExponent(
							divisor,
							gpu,
							prime,
							divisorData,
							out ulong computedCycle,
							out bool primeOrderFailed) || computedCycle == 0UL)
					{
						divisorCycle = _calculateCycleLength(
							divisor,
							gpu,
							divisorData,
							skipPrimeOrderHeuristic: primeOrderFailed);
					}
					else
					{
						divisorCycle = computedCycle;
					}

					divisorPool.Return(divisorData);

					if (divisorCycle == prime)
					{
						foundDivisor = divisor;
						processedAll = true;
						return true;
					}

					if (divisorCycle == 0UL)
					{
						Console.WriteLine($"Divisor cycle was not calculated for {prime}");
					}
				}
			}

			processedAll = true;
			foundDivisor = 0UL;
			return false;
		}

		if (divisor > limit)
		{
			processedAll = true;
			foundDivisor = 0UL;
			return false;
		}

		ulong remainingIterations = ((limit - divisor) / step) + 1UL;
		while (true)
		{
			bool passesSmallModuli = remainder3 != 0 && remainder7 != 0 && remainder11 != 0;
			if (passesSmallModuli && (remainder8 == 1 || remainder8 == 7) && ((decimalMask >> remainder10) & 1) != 0)
			{
				MontgomeryDivisorData divisorData = divisorPool.FromModulus(divisor);
				ulong divisorCycle;
				if (!_tryCalculateCycleLengthForExponent(
						divisor,
						gpu,
						prime,
						divisorData,
						out ulong computedCycle,
						out bool primeOrderFailed) || computedCycle == 0UL)
				{
					divisorCycle = _calculateCycleLength(
						divisor,
						gpu,
						divisorData,
						skipPrimeOrderHeuristic: primeOrderFailed);
				}
				else
				{
					divisorCycle = computedCycle;
				}

				divisorPool.Return(divisorData);
				if (divisorCycle == prime)
				{
					foundDivisor = divisor;
					processedAll = true;
					return true;
				}

				if (divisorCycle == 0UL)
				{
					Console.WriteLine($"Divisor cycle was not calculated for {prime}");
				}
			}

			remainingIterations--;
			if (remainingIterations == 0UL)
			{
				processedAll = true;
				foundDivisor = 0UL;
				return false;
			}

			divisor += step;
			remainder10 = AddMod10(remainder10, step10);
			remainder8 = AddMod8(remainder8, step8);
			remainder3 = AddMod3(remainder3, step3);
			remainder7 = AddMod7(remainder7, step7);
			remainder11 = AddMod11(remainder11, step11);
		}
	}

	private static byte CheckDivisor(ulong prime, ulong divisorCycle, in MontgomeryDivisorData divisorData)
	{
		ulong residue = prime.Pow2MontgomeryModWithCycleConvertToStandardCpu(divisorCycle, divisorData);
		return residue == 1UL ? (byte)1 : (byte)0;
	}

	private static bool TryCalculateCycleLengthForExponentCpu(
		ulong divisor,
		PrimeOrderCalculatorAccelerator gpu,
		ulong exponent,
		in MontgomeryDivisorData divisorData,
		out ulong cycleLength,
		out bool primeOrderFailed) => MersenneDivisorCycles.TryCalculateCycleLengthForExponentCpu(
			divisor,
			exponent,
			divisorData,
			out cycleLength,
			out primeOrderFailed);

	private static bool TryCalculateCycleLengthForExponentHybrid(
		ulong divisor,
		PrimeOrderCalculatorAccelerator gpu,
		ulong exponent,
		in MontgomeryDivisorData divisorData,
		out ulong cycleLength,
		out bool primeOrderFailed) => MersenneDivisorCycles.TryCalculateCycleLengthForExponentHybrid(
			gpu,
			divisor,
			exponent,
			divisorData,
			out cycleLength,
			out primeOrderFailed);

	private static bool TryCalculateCycleLengthForExponentGpu(
		ulong divisor,
		PrimeOrderCalculatorAccelerator gpu,
		ulong exponent,
		in MontgomeryDivisorData divisorData,
		out ulong cycleLength,
		out bool primeOrderFailed) => MersenneDivisorCycles.TryCalculateCycleLengthForExponentGpu(
			gpu,
			divisor,
			exponent,
			divisorData,
			out cycleLength,
			out primeOrderFailed);

	private static ulong CalculateCycleLengthCpu(
		ulong divisor,
		PrimeOrderCalculatorAccelerator gpu,
		in MontgomeryDivisorData divisorData,
		bool skipPrimeOrderHeuristic) => MersenneDivisorCycles.CalculateCycleLengthCpu(
			divisor,
			divisorData,
			skipPrimeOrderHeuristic);

	private static ulong CalculateCycleLengthHybrid(
		ulong divisor,
		PrimeOrderCalculatorAccelerator gpu,
		in MontgomeryDivisorData divisorData,
		bool skipPrimeOrderHeuristic) => MersenneDivisorCycles.CalculateCycleLengthHybrid(
			gpu,
			divisor,
			divisorData,
			skipPrimeOrderHeuristic);

	private static ulong CalculateCycleLengthGpu(
		ulong divisor,
		PrimeOrderCalculatorAccelerator gpu,
		in MontgomeryDivisorData divisorData,
		bool skipPrimeOrderHeuristic) => MersenneDivisorCycles.CalculateCycleLengthGpu(
			gpu,
			divisor,
			divisorData,
			skipPrimeOrderHeuristic);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static byte AddMod3(byte value, byte delta)
	{
		const int Modulus = 3;
		int sum = value + delta;

		if (sum >= Modulus)
		{
			sum -= Modulus;
		}

		return (byte)sum;
	}


	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static byte AddMod7(byte value, byte delta)
	{
		const int Modulus = 7;
		int sum = value + delta;

		if (sum >= Modulus)
		{
			sum -= Modulus;
		}

		return (byte)sum;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static byte AddMod10(byte value, byte delta)
	{
		const int Modulus = 10;
		int sum = value + delta;

		if (sum >= Modulus)
		{
			sum -= Modulus;
		}

		return (byte)sum;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static byte AddMod11(byte value, byte delta)
	{
		const int Modulus = 11;
		int sum = value + delta;

		if (sum >= Modulus)
		{
			sum -= Modulus;
		}

		return (byte)sum;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static byte AddMod8(byte value, byte delta)
	{
		return (byte)((value + delta) & 7);
	}

	private static ulong ComputeDivisorLimitFromMaxPrime(ulong maxPrime)
	{
		// The by-divisor CPU configuration only supplies primes greater than 1, so the guard below never trips.
		// if (maxPrime <= 1UL)
		// {
		//     return 0UL;
		// }
		if (maxPrime - 1UL >= 64UL)
		{
			return ulong.MaxValue;
		}

		return (1UL << (int)(maxPrime - 1UL)) - 1UL;
	}

	private static ulong ComputeAllowedMaxDivisor(ulong prime, ulong divisorLimit)
	{
		// Production by-divisor scans only handle primes, so inputs never fall below 2.
		// if (prime <= 1UL)
		// {
		//     return 0UL;
		// }
		if (prime - 1UL >= 64UL)
		{
			return divisorLimit;
		}

		return Math.Min((1UL << (int)(prime - 1UL)) - 1UL, divisorLimit);
	}


}
