using System.Numerics;
using System.Runtime.CompilerServices;
using System.Globalization;
using PerfectNumbers.Core.Gpu;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core.Cpu;

[DeviceDependentTemplate(typeof(ComputationDevice), suffix: "Order")]
public struct MersenneNumberDivisorByDivisorCpuTesterWithTemplate() : IMersenneNumberDivisorByDivisorTester
{
	public BigInteger DivisorLimit;
	public BigInteger MinK = EnvironmentConfiguration.MinK;

#if DEVICE_GPU || DEVICE_HYBRID
	public readonly PrimeOrderCalculatorAccelerator Accelerator = PrimeOrderCalculatorAccelerator.Rent(1);
#endif

#pragma warning disable CS8618 // StateFilePath is always set on EvenPerfectBitScanner execution path when ResumeFromState is called.
	private string StateFilePath;
#pragma warning restore CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider adding the 'required' modifier or declaring as nullable.

	private int _stateCounter;
	private BigInteger _lastSavedK;

	private GpuUInt128WorkSet _divisorScanGpuWorkSet;

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public void ResetStateTracking()
	{
		_stateCounter = 0;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public void ResumeFromState(in string stateFile, in BigInteger lastSavedK, in BigInteger minK)
	{
		StateFilePath = stateFile;
		_lastSavedK = lastSavedK;
		MinK = minK;
		_stateCounter = 0;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	public bool IsPrime(ulong prime, out bool divisorsExhausted, out BigInteger divisor)
	{
		BigInteger allowedMax = MersenneNumberDivisorByDivisorTester.ComputeAllowedMaxDivisorBig(prime, DivisorLimit);

		// The CPU by-divisor run always hands us primes with enormous divisor limits, so the fallback below never executes.
		// if (allowedMax < 3UL)
		// {
		//     // EvenPerfectBitScanner routes primes below the small-divisor cutoff to the GPU path, so the CPU path still sees
		//     // trivial candidates during targeted tests. Short-circuit here to keep those runs aligned with the production flow.
		//     divisorsExhausted = true;
		//     return true;
		// }


		bool composite = CheckDivisors(
			prime,
			allowedMax,
			MinK,
			out bool processedAll,
			out divisor);

		if (composite)
		{
			divisorsExhausted = true;
			return false;
		}

		divisorsExhausted = processedAll || composite;
		divisor = BigInteger.Zero;
		return true;
	}

	public void PrepareCandidates(ulong maxPrime, in ReadOnlySpan<ulong> primes, Span<ulong> allowedMaxValues)
	{
		BigInteger divisorLimitBigInteger = MersenneNumberDivisorByDivisorTester.ComputeDivisorLimitFromMaxPrimeBig(maxPrime);
		DivisorLimit = divisorLimitBigInteger;
		ulong divisorLimit = MersenneNumberDivisorByDivisorTester.GetAllowedDivisorLimitForSpan(divisorLimitBigInteger);
		int length = primes.Length;
		for (int index = 0; index < length; index++)
		{
			allowedMaxValues[index] = MersenneNumberDivisorByDivisorTester.ComputeAllowedMaxDivisor(primes[index], divisorLimit);
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public readonly IMersenneNumberDivisorByDivisorTester.IDivisorScanSession CreateDivisorSession(PrimeOrderCalculatorAccelerator gpu)
	{
#if DEVICE_GPU
		var session = new MersenneCpuDivisorScanSessionWithGpuOrder();
		session.Configure(gpu);
		return session;
#elif DEVICE_HYBRID
		var session = new MersenneCpuDivisorScanSessionWithHybridOrder();
		session.Configure(gpu);
		return session;
#else
		return new MersenneCpuDivisorScanSessionWithCpuOrder();
#endif
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private bool CheckDivisors(
		ulong prime,
		BigInteger allowedMax,
		BigInteger minK,
		out bool processedAll,
		out BigInteger foundDivisor)
	{
		// TODO: Is minK ever < 1? Can this be optimized
		BigInteger normalizedMinK = minK >= BigInteger.One ? minK : BigInteger.One,
				   step = ((BigInteger)prime) << 1,
				   firstDivisor = (step * normalizedMinK) + BigInteger.One;

		bool fits64 = normalizedMinK <= ulong.MaxValue && firstDivisor <= ulong.MaxValue,
			 unlimited = allowedMax.IsZero,
			 // This checks if the limit fits 64-bit
			 composite64 = unlimited || allowedMax <= ulong.MaxValue;
			 
		ulong allowedMax64,
			  foundDivisor64;

		if (fits64 && composite64)
		{
			allowedMax64 = unlimited ? ulong.MaxValue : (ulong)allowedMax;
			composite64 = CheckDivisors64Bit(
				prime,
				allowedMax64,
				(ulong)normalizedMinK,
				out processedAll,
				out foundDivisor64);
			foundDivisor = foundDivisor64;
			return composite64;
		}

		if (fits64 && allowedMax > ulong.MaxValue)
		{
			allowedMax64 = ulong.MaxValue;
			composite64 = CheckDivisors64Bit(
				prime,
				allowedMax64,
				(ulong)normalizedMinK,
				out processedAll,
				out foundDivisor64);
			if (composite64)
			{
				foundDivisor = foundDivisor64;
				return true;
			}

			BigInteger iterations = ((BigInteger)(allowedMax64 - (ulong)firstDivisor) / (ulong)step) + BigInteger.One;
			BigInteger nextK = normalizedMinK + iterations;
			return CheckDivisorsLarge(
				prime,
				allowedMax,
				nextK,
				step,
				out processedAll,
				out foundDivisor);
		}

		return CheckDivisorsLarge(
			prime,
			allowedMax,
			normalizedMinK,
			step,
			out processedAll,
			out foundDivisor);
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private bool CheckDivisors64Bit(
		ulong prime,
		ulong allowedMax,
		ulong minK,
		out bool processedAll,
		out ulong foundDivisor)
	{
		foundDivisor = 0UL;

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

		ulong currentK = minK,
			  // We're reusing candidate variable as stepLow to limit registry pressure
			  candidate = step.Low,
			  // We're reusing computedCycle variable as lowerEnd to limit registry pressure
			  computedCycle,
			  // We're reusing divisorCycle variable as maxK to limit registry pressure
			  divisorCycle;

			 // We're reusing computed variable as processedTop to limit registry pressure
		bool computed,
			 // We're reusing primeOrderFailed variable as processedBottom to limit registry pressure
			 primeOrderFailed,
			 composite;

		if (step.High == 0UL && limit.High == 0UL)
		{
			divisorCycle = allowedMax > 0UL ? (allowedMax - 1UL) / candidate : 0UL;
			if (divisorCycle == 0UL)
			{
				processedAll = true;
				foundDivisor = 0UL;
				return false;
			}

			computed = true;
			primeOrderFailed = true;
			composite = false;

			if (minK <= divisorCycle)
			{
				composite = CheckDivisors64Range(
					prime,
					candidate,
					allowedMax,
					minK,
					divisorCycle,
					ref currentK,
					out computed,
					out foundDivisor);
				if (composite)
				{
					processedAll = true;
					return true;
				}
			}

			computedCycle = minK > 1UL ? minK - 1UL : 0UL;
			if (computedCycle >= 1UL && minK == 1UL)
			{
				computedCycle = Math.Min(computedCycle, divisorCycle);
				currentK = 1UL;
				composite = CheckDivisors64Range(
					prime,
					candidate,
					allowedMax,
					1UL,
					computedCycle,
					ref currentK,
					out primeOrderFailed,
					out foundDivisor);
			}

			processedAll = computed && primeOrderFailed;
			return composite;
		}

#if DEVICE_GPU || DEVICE_HYBRID
		PrimeOrderCalculatorAccelerator gpu = Accelerator;
#endif
		var residueStepper = new MersenneDivisorResidueStepper(prime, step, divisor);

		while (divisor.CompareTo(limit) <= 0)
		{
			if (residueStepper.IsAdmissible())
			{
				candidate = divisor.Low;
				MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(candidate);
				// Divisors generated from 2 * k * p + 1 exceed the small-cycle snapshot when p >= 138,000,000, so the short path below never runs.
#if DEVICE_GPU
				computed = MersenneNumberDivisorByDivisorTester.TryCalculateCycleLengthForExponentGpu(
					candidate,
					gpu,
					prime,
					divisorData,
					out computedCycle,
					out primeOrderFailed);
#elif DEVICE_HYBRID
				computed = MersenneNumberDivisorByDivisorTester.TryCalculateCycleLengthForExponentHybrid(
					candidate,
					gpu,
					prime,
					divisorData,
					out computedCycle,
					out primeOrderFailed);
#else
				computed = MersenneNumberDivisorByDivisorTester.TryCalculateCycleLengthForExponentCpu(
					candidate,
					prime,
					divisorData,
					out computedCycle,
					out primeOrderFailed);
#endif
				if (computed && computedCycle != 0UL)
				{
					divisorCycle = computedCycle;
				}
				else
				{
					// Divisors produced by 2 * k * p + 1 always exceed PerfectNumberConstants.MaxQForDivisorCycles
					// for the exponents scanned here, so skip the unused cache fallback and compute directly.
#if DEVICE_GPU
					divisorCycle = MersenneNumberDivisorByDivisorTester.CalculateCycleLengthGpu(
						candidate,
						gpu,
						divisorData,
						skipPrimeOrderHeuristic: primeOrderFailed);
#elif DEVICE_HYBRID
					divisorCycle = MersenneNumberDivisorByDivisorTester.CalculateCycleLengthHybrid(
						candidate,
						gpu,
						divisorData,
						skipPrimeOrderHeuristic: primeOrderFailed);
#else
					divisorCycle = MersenneNumberDivisorByDivisorTester.CalculateCycleLengthCpu(
						candidate,
						divisorData,
						skipPrimeOrderHeuristic: primeOrderFailed);
#endif
				}

				RecordState(currentK);
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
			currentK++;
			residueStepper.Advance();
		}

		processedAll = true;
		foundDivisor = 0UL;
		return false;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private bool CheckDivisorsLarge(
		ulong prime,
		BigInteger allowedMax,
		BigInteger minK,
		BigInteger step,
		out bool processedAll,
		out BigInteger foundDivisor)
	{
		processedAll = true;
		foundDivisor = BigInteger.Zero;
		BigInteger currentK = minK;
		BigInteger divisor = (step * currentK) + BigInteger.One;
		if (allowedMax.IsZero)
		{
			processedAll = false;
			return false;
		}
		if (divisor > allowedMax)
		{
			return false;
		}

		bool lastIsSeven = (prime & 3UL) == 3UL;
		ushort decimalMask = DivisorGenerator.GetDecimalMask(lastIsSeven);

		var rem10 = new BigIntegerCycleRemainderStepper(10);
		var rem8 = new BigIntegerCycleRemainderStepper(8);
		var rem5 = new BigIntegerCycleRemainderStepper(5);
		var rem3 = new BigIntegerCycleRemainderStepper(3);
		var rem7 = new BigIntegerCycleRemainderStepper(7);
		var rem11 = new BigIntegerCycleRemainderStepper(11);
		var rem13 = new BigIntegerCycleRemainderStepper(13);
		var rem17 = new BigIntegerCycleRemainderStepper(17);
		var rem19 = new BigIntegerCycleRemainderStepper(19);

		byte remainder10 = (byte)rem10.Initialize(divisor);
		byte remainder8 = (byte)rem8.Initialize(divisor);
		byte remainder5 = (byte)rem5.Initialize(divisor);
		byte remainder3 = (byte)rem3.Initialize(divisor);
		byte remainder7 = (byte)rem7.Initialize(divisor);
		byte remainder11 = (byte)rem11.Initialize(divisor);
		byte remainder13 = (byte)rem13.Initialize(divisor);
		byte remainder17 = (byte)rem17.Initialize(divisor);
		byte remainder19 = (byte)rem19.Initialize(divisor);

		while (divisor <= allowedMax)
		{
			bool passesSmallModuli = remainder3 != 0 && remainder5 != 0 && remainder7 != 0 && remainder11 != 0 && remainder13 != 0 && remainder17 != 0 && remainder19 != 0;
			if (passesSmallModuli && (remainder8 == 1 || remainder8 == 7) && ((decimalMask >> remainder10) & 1) != 0)
			{
				if (MersenneNumberDivisorByDivisorTester.IsProbablePrimeBigInteger(divisor))
				{
					BigInteger powResult = BigInteger.ModPow(2, prime, divisor);
					RecordState(currentK);
					if (powResult.IsOne)
					{
						foundDivisor = divisor;
						processedAll = true;
						return true;
					}
				}
				else
				{
					RecordState(currentK);
				}
			}

			currentK++;
			divisor += step;
			remainder10 = (byte)rem10.ComputeNext(divisor);
			remainder8 = (byte)rem8.ComputeNext(divisor);
			remainder5 = (byte)rem5.ComputeNext(divisor);
			remainder3 = (byte)rem3.ComputeNext(divisor);
			remainder7 = (byte)rem7.ComputeNext(divisor);
			remainder11 = (byte)rem11.ComputeNext(divisor);
			remainder13 = (byte)rem13.ComputeNext(divisor);
			remainder17 = (byte)rem17.ComputeNext(divisor);
			remainder19 = (byte)rem19.ComputeNext(divisor);

			if (divisor > allowedMax)
			{
				break;
			}
		}

		processedAll = true;
		foundDivisor = BigInteger.Zero;
		return false;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private bool CheckDivisors64Range(
		ulong prime,
		ulong step,
		ulong allowedMax,
		ulong startK,
		ulong endK,
		ref ulong currentK,
		out bool processedAll,
		out ulong foundDivisor)
	{
		if (startK < 1UL || endK < startK)
		{
			processedAll = true;
			foundDivisor = 0UL;
			return false;
		}
		currentK = startK;

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

		var residueStepper = new MersenneDivisorResidueStepper(prime, (GpuUInt128)step128, (GpuUInt128)startDivisor128);

		return CheckDivisors64(
			prime,
			step,
			rangeLimit,
			startDivisor,
			ref residueStepper,
			ref currentK,
			out processedAll,
			out foundDivisor);
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private bool CheckDivisors64(
			ulong prime,
			ulong step,
			ulong limit,
			ulong divisor,
			ref MersenneDivisorResidueStepper residueStepper,
			ref ulong currentK,
			out bool processedAll,
			out ulong foundDivisor)
	{
#if DEVICE_GPU || DEVICE_HYBRID
		PrimeOrderCalculatorAccelerator gpu = Accelerator;
#endif
		ulong computedCycle;
		bool primeOrderFailed;
		ulong divisorCycle;
		bool computed;
		if (step > limit)
		{
			if (divisor <= limit && residueStepper.IsAdmissible())
			{
				MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(divisor);
#if DEVICE_GPU
				computed = MersenneNumberDivisorByDivisorTester.TryCalculateCycleLengthForExponentGpu(
					divisor,
					gpu,
					prime,
					divisorData,
					out computedCycle,
					out primeOrderFailed);
#elif DEVICE_HYBRID
				computed = MersenneNumberDivisorByDivisorTester.TryCalculateCycleLengthForExponentHybrid(
					divisor,
					gpu,
					prime,
					divisorData,
					out computedCycle,
					out primeOrderFailed);
#else
				computed = MersenneNumberDivisorByDivisorTester.TryCalculateCycleLengthForExponentCpu(
					divisor,
					prime,
					divisorData,
					out computedCycle,
					out primeOrderFailed);
#endif
				if (!computed || computedCycle == 0UL)
				{
#if DEVICE_GPU
					divisorCycle = MersenneNumberDivisorByDivisorTester.CalculateCycleLengthGpu(
						divisor,
						gpu,
						divisorData,
						skipPrimeOrderHeuristic: primeOrderFailed);
#elif DEVICE_HYBRID
					divisorCycle = MersenneNumberDivisorByDivisorTester.CalculateCycleLengthHybrid(
						divisor,
						gpu,
						divisorData,
						skipPrimeOrderHeuristic: primeOrderFailed);
#else
					divisorCycle = MersenneNumberDivisorByDivisorTester.CalculateCycleLengthCpu(
						divisor,
						divisorData,
						skipPrimeOrderHeuristic: primeOrderFailed);
#endif
				}
				else
				{
					divisorCycle = computedCycle;
				}

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
			if (residueStepper.IsAdmissible())
			{
				MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(divisor);
#if DEVICE_GPU
				computed = MersenneNumberDivisorByDivisorTester.TryCalculateCycleLengthForExponentGpu(
					divisor,
					gpu,
					prime,
					divisorData,
					out computedCycle,
					out primeOrderFailed);
#elif DEVICE_HYBRID
				computed = MersenneNumberDivisorByDivisorTester.TryCalculateCycleLengthForExponentHybrid(
					divisor,
					gpu,
					prime,
					divisorData,
					out computedCycle,
					out primeOrderFailed);
#else
				computed = MersenneNumberDivisorByDivisorTester.TryCalculateCycleLengthForExponentCpu(
					divisor,
					prime,
					divisorData,
					out computedCycle,
					out primeOrderFailed);
#endif
				if (!computed || computedCycle == 0UL)
				{
#if DEVICE_GPU
					divisorCycle = MersenneNumberDivisorByDivisorTester.CalculateCycleLengthGpu(
						divisor,
						gpu,
						divisorData,
						skipPrimeOrderHeuristic: primeOrderFailed);
#elif DEVICE_HYBRID
					divisorCycle = MersenneNumberDivisorByDivisorTester.CalculateCycleLengthHybrid(
						divisor,
						gpu,
						divisorData,
						skipPrimeOrderHeuristic: primeOrderFailed);
#else
					divisorCycle = MersenneNumberDivisorByDivisorTester.CalculateCycleLengthCpu(
						divisor,
						divisorData,
						skipPrimeOrderHeuristic: primeOrderFailed);
#endif
				}
				else
				{
					divisorCycle = computedCycle;
				}

				RecordState(currentK);
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
			currentK++;
			residueStepper.Advance();
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private void RecordState(BigInteger k)
	{
		string path = StateFilePath;
		if (k <= _lastSavedK)
		{
			return;
		}

		int next = _stateCounter + 1;
		if (next >= PerfectNumberConstants.ByDivisorStateSaveInterval)
		{
			string? directory = Path.GetDirectoryName(path);
			if (!string.IsNullOrEmpty(directory))
			{
				Directory.CreateDirectory(directory);
			}

			File.AppendAllText(path, k.ToString(CultureInfo.InvariantCulture) + Environment.NewLine);
			_stateCounter = 0;
			_lastSavedK = k;
		}
		else
		{
			_stateCounter = next;
		}
	}
}
