using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Globalization;
using System.Collections.Generic;
using System.Collections.Concurrent;
using PerfectNumbers.Core.Gpu;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core.Cpu;

public enum DivisorSet
{
	OneByOne,
	Pow2Groups,
}

[EnumDependentTemplate(typeof(DivisorSet))]
[DeviceDependentTemplate(typeof(ComputationDevice))]
[NameSuffix("Order")]
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
	private bool _pow2MinusOneChecked;
	private string _pow2MinusOneStateFilePath = string.Empty;

	private GpuUInt128WorkSet _divisorScanGpuWorkSet;
	private static readonly ConcurrentDictionary<ulong, byte> _checkedPow2MinusOne = new();

#if DivisorSet_Pow2Groups
	private const int PercentScale = DivisorByDivisorConfig.PercentScale;
	private static readonly int[] SpecialPercentPromille = DivisorByDivisorConfig.SpecialPercentPromille;
	private static readonly (int Start, int End)[] PercentGroupsPromille = DivisorByDivisorConfig.PercentGroupsPromille;
	private static long _specialHits;
	private static long _groupHits;

	private enum Pow2Phase
	{
		None,
		Special,
		Groups,
	}

	private Pow2Phase _currentPow2Phase;
	private string _specialStateFilePath = string.Empty;
	private string _groupsStateFilePath = string.Empty;
	private int _specialStateCounter;
	private int _groupsStateCounter;
	private BigInteger _specialLastSavedK;
	private BigInteger _groupsLastSavedK;
	private BigInteger _specialResumeK;
	private BigInteger _groupsResumeK;
	private bool _preparedSpecialsInitialized;
	private int _preparedSpecialRangePromille;
	private int[] _preparedSpecialSingles = Array.Empty<int>();
	private (int Start, int End)[] _preparedSpecialRanges = Array.Empty<(int Start, int End)>();
#endif

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool IsMersenneValue(ulong prime, in BigInteger divisor)
	{
		if (divisor <= BigInteger.One || prime == 0UL || prime > int.MaxValue)
		{
			return false;
		}

		int divisorBitLength = GetBitLengthPortable(divisor);
		if (divisorBitLength != (int)prime)
		{
			return false;
		}

		BigInteger plusOne = divisor + BigInteger.One;
		return plusOne > BigInteger.One && IsPowerOfTwoBig(plusOne);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static int GetBitLengthPortable(in BigInteger value)
	{
		if (value.IsZero)
		{
			return 0;
		}

		byte[] bytes = value.ToByteArray(isUnsigned: true, isBigEndian: false);
		byte msb = bytes[^1];
		return ((bytes.Length - 1) * 8) + (BitOperations.Log2(msb) + 1);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool IsPowerOfTwoBig(in BigInteger value)
	{
		return value > BigInteger.One && (value & (value - BigInteger.One)) == BigInteger.Zero;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool TryGetMersenneStopK(ulong prime, in BigInteger step, out BigInteger stopK, out BigInteger tailStartK)
	{
		stopK = BigInteger.Zero;
		tailStartK = BigInteger.Zero;

		if (prime == 0UL || prime > int.MaxValue)
		{
			return false;
		}

		int bits;
		try
		{
			bits = checked((int)prime);
		}
		catch (OverflowException)
		{
			throw;
		}

		BigInteger mersenne;
		try
		{
			mersenne = (BigInteger.One << bits) - BigInteger.One;
		}
		catch (OutOfMemoryException)
		{
			throw;
		}
		BigInteger denominator = step;
		if (denominator <= BigInteger.Zero)
		{
			return false;
		}

		stopK = (mersenne - BigInteger.One) / denominator;
		if (stopK < BigInteger.One)
		{
			return false;
		}

		int tailCount = PerfectNumberConstants.ByDivisorMersenneTailCount;
		if (tailCount < 0)
		{
			tailCount = 0;
		}

		BigInteger tail = stopK - tailCount;
		tailStartK = tail > BigInteger.One ? tail : BigInteger.One;
		return true;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool TryClampMaxKForMersenneTail(ulong prime, in BigInteger step, ref BigInteger maxKAllowed)
	{
		if (maxKAllowed <= BigInteger.Zero)
		{
			return false;
		}

		if (!TryGetMersenneStopK(prime, step, out BigInteger stopK, out BigInteger tailStartK))
		{
			return false;
		}

		if (maxKAllowed < tailStartK)
		{
			return false;
		}

		BigInteger cap = stopK - BigInteger.One;
		if (cap < BigInteger.One)
		{
			return false;
		}

		if (maxKAllowed > cap)
		{
			maxKAllowed = cap;
		}

		return true;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public void ResetStateTracking()
	{
		_stateCounter = 0;
		_pow2MinusOneChecked = false;
#if DivisorSet_Pow2Groups
		_specialStateCounter = 0;
		_groupsStateCounter = 0;
		_currentPow2Phase = Pow2Phase.None;
		_preparedSpecialsInitialized = false;
#endif
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public void ResumeFromState(in string stateFile, in BigInteger lastSavedK, in BigInteger minK)
	{
		StateFilePath = stateFile;
		_lastSavedK = lastSavedK;
		MinK = minK;
		_stateCounter = 0;
		_pow2MinusOneStateFilePath = stateFile + ".pow2minus1";
		_pow2MinusOneChecked = File.Exists(_pow2MinusOneStateFilePath);
#if DivisorSet_Pow2Groups
		_specialStateFilePath = stateFile + ".special";
		_groupsStateFilePath = stateFile + ".groups";
		_specialStateCounter = 0;
		_groupsStateCounter = 0;
		if (!MersenneNumberDivisorByDivisorTester.TryReadLastSavedK(_specialStateFilePath, out _specialLastSavedK))
		{
			_specialLastSavedK = lastSavedK;
		}

		if (!MersenneNumberDivisorByDivisorTester.TryReadLastSavedK(_groupsStateFilePath, out _groupsLastSavedK))
		{
			_groupsLastSavedK = lastSavedK;
		}

		_specialResumeK = _specialLastSavedK + BigInteger.One;
		_groupsResumeK = _groupsLastSavedK + BigInteger.One;
		_currentPow2Phase = Pow2Phase.None;
		_preparedSpecialsInitialized = false;
#endif
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	public bool IsPrime(ulong prime, out bool divisorsExhausted, out BigInteger divisor)
	{
		BigInteger allowedMax = MersenneNumberDivisorByDivisorTester.ComputeAllowedMaxDivisorBig(prime, DivisorLimit);

		if (TryFindPowerOfTwoMinusOneDivisor(prime, allowedMax, out divisor))
		{
			divisorsExhausted = true;
			return false;
		}

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
			out divisorsExhausted,
			out divisor);

		if (composite)
		{
			divisorsExhausted = true;
			return false;
		}

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
		// TODO: Is minK ever < 1? Can this be optimized - normalizedMinK is always >= 1 after this
		BigInteger normalizedMinK = minK >= BigInteger.One ? minK : BigInteger.One,
				   step = ((BigInteger)prime) << 1,
				   firstDivisor = (step * normalizedMinK) + BigInteger.One;

		bool primeLastIsSeven = (prime & 3UL) == 3UL;
		ushort primeDecimalMask = DivisorGenerator.GetDecimalMask(primeLastIsSeven);
		bool fits64 = normalizedMinK <= ulong.MaxValue && firstDivisor <= ulong.MaxValue,
			 unlimited = allowedMax.IsZero,
			 // This checks if the limit fits 64-bit
			 composite64 = unlimited || allowedMax <= ulong.MaxValue;

		ulong allowedMax64,
			  foundDivisor64;

#if DivisorSet_Pow2Groups
			bool compositePow2 = CheckDivisorsPow2Groups(
				prime,
				primeDecimalMask,
				allowedMax,
				normalizedMinK,
				step,
				out processedAll,
				out foundDivisor);
			return compositePow2;
#else
		if (fits64 && composite64)
		{
			allowedMax64 = unlimited ? ulong.MaxValue : (ulong)allowedMax;
			composite64 = CheckDivisors64Bit(
				prime,
				primeDecimalMask,
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
				primeDecimalMask,
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
				primeDecimalMask,
				allowedMax,
				nextK,
				step,
				out processedAll,
				out foundDivisor);
		}
#endif

		return CheckDivisorsLarge(
			prime,
			primeDecimalMask,
			allowedMax,
			normalizedMinK,
			step,
			out processedAll,
			out foundDivisor);
	}

#if DivisorSet_Pow2Groups
	private static readonly (int, int)[] _emptyPow2Groups = Array.Empty<(int, int)>();

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private bool CheckDivisorsPow2Groups(
		ulong prime,
		ushort primeDecimalMask,
		BigInteger allowedMax,
		BigInteger minK,
		BigInteger step,
		out bool processedAll,
		out BigInteger foundDivisor)
	{
		foundDivisor = BigInteger.Zero;
		processedAll = true;

		if (allowedMax.IsZero)
		{
			processedAll = false;
			return false;
		}

		BigInteger normalizedMinK = minK >= BigInteger.One ? minK : BigInteger.One;
		BigInteger maxKAllowed = (allowedMax - BigInteger.One) / step;
		TryClampMaxKForMersenneTail(prime, step, ref maxKAllowed);
		if (maxKAllowed < normalizedMinK)
		{
			return false;
		}

		ReadOnlySpan<int> specialPercents = SpecialPercentPromille;
		ReadOnlySpan<(int Start, int End)> groups = PercentGroupsPromille;

		// Segment 1: ulong range
		if (normalizedMinK <= BigIntegerNumbers.UlongMaxValue)
		{
			ulong startK64 = (ulong)normalizedMinK;
			ulong maxK64 = maxKAllowed > BigIntegerNumbers.UlongMaxValue ? ulong.MaxValue : (ulong)maxKAllowed;
			ulong allowedMax64 = allowedMax > BigIntegerNumbers.UlongMaxValue ? ulong.MaxValue : (ulong)allowedMax;

			bool specialFound = ProcessPow2Phase(
				prime,
				allowedMax64,
				prime << 1,
				_specialResumeK > BigInteger.Zero ? _specialResumeK : startK64,
				startK64,
				Pow2Phase.Special,
				_emptyPow2Groups,
				specialPercents,
				ref processedAll,
				out ulong foundDiv64);
			if (specialFound)
			{
				foundDivisor = foundDiv64;
				return true;
			}

			bool groupFound = ProcessPow2Phase(
				prime,
				allowedMax64,
				prime << 1,
				_groupsResumeK > BigInteger.Zero ? _groupsResumeK : startK64,
				startK64,
				Pow2Phase.Groups,
				groups,
				ReadOnlySpan<int>.Empty,
				ref processedAll,
				out foundDiv64);
			if (groupFound)
			{
				foundDivisor = foundDiv64;
				return true;
			}
		}

		// Segment 2: UInt128 range
		BigInteger start128 = BigInteger.Max(normalizedMinK, BigIntegerNumbers.UlongMaxValuePlusOne);
		if (start128 <= maxKAllowed && start128 <= BigIntegerNumbers.UInt128MaxValue)
		{
			BigInteger end128 = BigInteger.Min(maxKAllowed, BigIntegerNumbers.UInt128MaxValue);
			bool specialFound128 = ProcessPow2PhaseBig(
				prime,
				primeDecimalMask,
				allowedMax,
				step,
				_specialResumeK > start128 ? _specialResumeK : start128,
				normalizedMinK,
				start128,
				end128,
				Pow2Phase.Special,
				_emptyPow2Groups,
				specialPercents,
				ref processedAll,
				out BigInteger foundBig);
			if (specialFound128)
			{
				foundDivisor = foundBig;
				return true;
			}

			bool groupFound128 = ProcessPow2PhaseBig(
				prime,
				primeDecimalMask,
				allowedMax,
				step,
				_groupsResumeK > start128 ? _groupsResumeK : start128,
				normalizedMinK,
				start128,
				end128,
				Pow2Phase.Groups,
				groups,
				ReadOnlySpan<int>.Empty,
				ref processedAll,
				out foundBig);
			if (groupFound128)
			{
				foundDivisor = foundBig;
				return true;
			}
		}

		// Segment 3: BigInteger range
		BigInteger startBig = BigInteger.Max(normalizedMinK, BigIntegerNumbers.UInt128MaxValuePlusOne);
		if (startBig <= maxKAllowed)
		{
			bool specialFoundBig = ProcessPow2PhaseBig(
				prime,
				primeDecimalMask,
				allowedMax,
				step,
				_specialResumeK > startBig ? _specialResumeK : startBig,
				normalizedMinK,
				startBig,
				maxKAllowed,
				Pow2Phase.Special,
				_emptyPow2Groups,
				specialPercents,
				ref processedAll,
				out BigInteger foundBig);
			if (specialFoundBig)
			{
				foundDivisor = foundBig;
				return true;
			}

			bool groupFoundBig = ProcessPow2PhaseBig(
				prime,
				primeDecimalMask,
				allowedMax,
				step,
				_groupsResumeK > startBig ? _groupsResumeK : startBig,
				normalizedMinK,
				startBig,
				maxKAllowed,
				Pow2Phase.Groups,
				groups,
				ReadOnlySpan<int>.Empty,
				ref processedAll,
				out foundBig);
			if (groupFoundBig)
			{
				foundDivisor = foundBig;
				return true;
			}
		}

		return false;
	}
#endif

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private bool CheckDivisors64Bit(
			ulong prime,
			ushort primeDecimalMask,
			ulong allowedMax,
			ulong minK,
			out bool processedAll,
			out ulong foundDivisor)
	{
		foundDivisor = 0UL;

#if DivisorSet_Pow2Groups
			ulong step64 = prime << 1;
			if (step64 == 0UL)
			{
				processedAll = true;
				return false;
			}

			processedAll = true;
			BigInteger stepBig = step64;
			BigInteger maxKAllowed = allowedMax == 0UL ? BigInteger.Zero : (((BigInteger)allowedMax - BigInteger.One) / stepBig);
			TryClampMaxKForMersenneTail(prime, stepBig, ref maxKAllowed);

			ReadOnlySpan<int> specialPercents = SpecialPercentPromille;
			ReadOnlySpan<(int Start, int End)> groups = PercentGroupsPromille;

			// Segment 1: ulong range
			ulong max64K = maxKAllowed <= ulong.MaxValue ? (ulong)maxKAllowed : ulong.MaxValue;
			if (minK <= max64K)
			{
				ulong max64 = allowedMax == 0UL ? ulong.MaxValue : allowedMax;
				bool specialFound = ProcessPow2Phase(
					prime,
					max64,
					step64,
					_specialResumeK > BigInteger.Zero ? _specialResumeK : minK,
					minK,
					Pow2Phase.Special,
					_emptyPow2Groups,
					specialPercents,
					ref processedAll,
					out foundDivisor);
				if (specialFound)
				{
					return true;
				}

				bool groupFound = ProcessPow2Phase(
					prime,
					max64,
					step64,
					_groupsResumeK > BigInteger.Zero ? _groupsResumeK : minK,
					minK,
					Pow2Phase.Groups,
					groups,
					ReadOnlySpan<int>.Empty,
					ref processedAll,
					out foundDivisor);
				if (groupFound)
				{
					return true;
				}
			}

			// Segment 2: UInt128 range
			BigInteger start128 = BigInteger.Max(minK, BigIntegerNumbers.UlongMaxValuePlusOne);
			if (maxKAllowed > BigIntegerNumbers.UlongMaxValue && start128 <= maxKAllowed)
			{
				BigInteger end128 = BigInteger.Min(maxKAllowed, BigIntegerNumbers.UInt128MaxValue);
				bool specialFound128 = ProcessPow2PhaseBig(
					prime,
					primeDecimalMask,
					(BigInteger)allowedMax == BigInteger.Zero ? BigInteger.Zero : (BigInteger)allowedMax,
					stepBig,
					_specialResumeK > start128 ? _specialResumeK : start128,
					minK,
					start128,
					end128,
					Pow2Phase.Special,
					_emptyPow2Groups,
				specialPercents,
				ref processedAll,
				out BigInteger foundBig);
			if (specialFound128)
			{
				foundDivisor = foundBig > ulong.MaxValue ? ulong.MaxValue : (ulong)foundBig;
				return true;
			}

				bool groupFound128 = ProcessPow2PhaseBig(
					prime,
					primeDecimalMask,
					(BigInteger)allowedMax == BigInteger.Zero ? BigInteger.Zero : (BigInteger)allowedMax,
					stepBig,
					_groupsResumeK > start128 ? _groupsResumeK : start128,
					minK,
					start128,
					end128,
					Pow2Phase.Groups,
					groups,
					ReadOnlySpan<int>.Empty,
				ref processedAll,
				out foundBig);
			if (groupFound128)
			{
				foundDivisor = foundBig > ulong.MaxValue ? ulong.MaxValue : (ulong)foundBig;
				return true;
			}
			}

			// Segment 3: BigInteger range beyond UInt128
			BigInteger startBig = BigInteger.Max(minK, BigIntegerNumbers.UInt128MaxValuePlusOne);
			if (maxKAllowed > BigIntegerNumbers.UInt128MaxValue && startBig <= maxKAllowed)
			{
				bool specialFoundBig = ProcessPow2PhaseBig(
					prime,
					primeDecimalMask,
					(BigInteger)allowedMax == BigInteger.Zero ? BigInteger.Zero : (BigInteger)allowedMax,
					stepBig,
					_specialResumeK > startBig ? _specialResumeK : startBig,
					minK,
					startBig,
					maxKAllowed,
					Pow2Phase.Special,
					_emptyPow2Groups,
					specialPercents,
				ref processedAll,
				out BigInteger foundBig);
			if (specialFoundBig)
			{
				foundDivisor = foundBig > ulong.MaxValue ? ulong.MaxValue : (ulong)foundBig;
				return true;
			}

				bool groupFoundBig = ProcessPow2PhaseBig(
					prime,
					primeDecimalMask,
					(BigInteger)allowedMax == BigInteger.Zero ? BigInteger.Zero : (BigInteger)allowedMax,
					stepBig,
					_groupsResumeK > startBig ? _groupsResumeK : startBig,
					minK,
					startBig,
					maxKAllowed,
					Pow2Phase.Groups,
					groups,
					ReadOnlySpan<int>.Empty,
				ref processedAll,
				out foundBig);
			if (groupFoundBig)
			{
				foundDivisor = foundBig > ulong.MaxValue ? ulong.MaxValue : (ulong)foundBig;
				return true;
			}
			}

			return false;
#else
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
				if (divisorCycle == prime && !IsMersenneValue(prime, candidate))
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
#endif
	}

#if DivisorSet_Pow2Groups
	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private bool ProcessPow2Phase(
		ulong prime,
		ulong max64,
		ulong step64,
		BigInteger resumeK,
		ulong minK,
	Pow2Phase phase,
	ReadOnlySpan<(int Start, int End)> ranges,
	ReadOnlySpan<int> singles,
	ref bool processedAll,
	out ulong foundDivisor)
	{
		foundDivisor = 0UL;
		int specialRangePromille = EnvironmentConfiguration.ByDivisorSpecialRange;
		if (specialRangePromille < 0)
		{
			specialRangePromille = 0;
		}
		
		specialRangePromille *= 100; // convert percent to promille-of-percentScale (10_000 = 100%)
		EnsurePreparedSpecials(specialRangePromille);
		if (resumeK > ulong.MaxValue)
		{
			processedAll = false;
			return false;
		}

		BigInteger effectiveResume = resumeK < minK ? minK : resumeK;
		BigInteger firstDivisorBig = (((BigInteger)prime) << 1) * effectiveResume + BigInteger.One;
		if (firstDivisorBig > max64)
		{
			processedAll = false;
			return false;
		}

		ulong firstDivisor = (ulong)firstDivisorBig;
		int startBit = BitOperations.Log2(firstDivisor);
		int maxBit = BitOperations.Log2(max64);
		ulong currentK = (ulong)effectiveResume;
		_currentPow2Phase = phase;
		FlushPow2State(effectiveResume, phase);

		ReadOnlySpan<int> effectiveSingles = singles;
		ReadOnlySpan<(int Start, int End)> effectiveRanges = ranges;
		if (phase == Pow2Phase.Special)
		{
			effectiveSingles = _preparedSpecialSingles;
			effectiveRanges = _preparedSpecialRanges;
		}

		try
		{
			for (int bit = startBit; bit <= maxBit; bit++)
			{
				ulong lower = 1UL << bit;
				ulong upperExclusive = bit == 63 ? ulong.MaxValue : lower << 1;

				if (lower > max64)
				{
					break;
				}

				BigInteger lowerBig = lower;

				for (int i = 0; i < effectiveSingles.Length; i++)
				{
					int startProm = effectiveSingles[i];
					int endProm = effectiveSingles[i];
					if (specialRangePromille > 0)
					{
						startProm = Math.Max(0, effectiveSingles[i] - specialRangePromille);
						endProm = Math.Min(PercentScale, effectiveSingles[i] + specialRangePromille);
					}

					if (specialRangePromille == 0)
					{
						int adjustedEndProm = Math.Min(PercentScale, startProm + 1);
						BigInteger scaledStart = (PercentScale + startProm) * lowerBig;
						BigInteger scaledEnd = (PercentScale + adjustedEndProm) * lowerBig;

						BigInteger qMin = (scaledStart + (PercentScale - 1)) / PercentScale;
						BigInteger qMax = scaledEnd / PercentScale;

						if ((qMin & 1) == 0)
						{
							qMin += 1;
						}

						if ((qMax & 1) == 0)
						{
							qMax -= 1;
						}

						if (qMin < 3 || qMin > qMax)
						{
							continue;
						}

						if (qMin < lowerBig)
						{
							qMin = lowerBig | 1;
						}

						if (qMax >= upperExclusive)
						{
							qMax = upperExclusive - 1;
							if ((qMax & 1) == 0)
							{
								qMax -= 1;
							}
						}

						if (qMin > qMax || qMin > max64)
						{
							continue;
						}

						if (qMax > max64)
						{
							qMax = max64;
							if ((qMax & 1) == 0)
							{
								qMax -= 1;
							}
						}

						BigInteger stepBig = step64;
						BigInteger kMinBig = (qMin - 1 + (stepBig - 1)) / stepBig;
						BigInteger kMaxBig = (qMax - 1) / stepBig;

						if (kMinBig < minK || kMinBig < effectiveResume)
						{
							kMinBig = BigInteger.Max(BigInteger.Max(minK, effectiveResume), kMinBig);
						}

						if (kMinBig > kMaxBig)
						{
							continue;
						}

						if (kMaxBig > ulong.MaxValue)
						{
							kMaxBig = ulong.MaxValue;
						}

						ulong kStart = (ulong)kMinBig;
						ulong kEnd = (ulong)kMaxBig;

						bool composite = CheckDivisors64Range(
							prime,
							step64,
							max64,
							kStart,
							kEnd,
							ref currentK,
							out bool processedRange,
							out foundDivisor);
						if (composite)
						{
							RecordPow2Hit(phase);
							FlushPow2State(currentK, phase);
							_currentPow2Phase = Pow2Phase.None;
							return true;
						}

						if (!processedRange)
						{
							processedAll = false;
						}
					}
					else
					{
						BigInteger scaledStart = (PercentScale + startProm) * lowerBig;
						BigInteger scaledEnd = (PercentScale + endProm) * lowerBig;

						BigInteger qMin = (scaledStart + (PercentScale - 1)) / PercentScale;
						BigInteger qMax = scaledEnd / PercentScale;

						if ((qMin & 1) == 0)
						{
							qMin += 1;
						}

						if ((qMax & 1) == 0)
						{
							qMax -= 1;
						}

						if (qMin < 3 || qMin > qMax)
						{
							continue;
						}

						if (qMin < lowerBig)
						{
							qMin = lowerBig | 1;
						}

						if (qMax >= upperExclusive)
						{
							qMax = upperExclusive - 1;
							if ((qMax & 1) == 0)
							{
								qMax -= 1;
							}
						}

						if (qMin > qMax || qMin > max64)
						{
							continue;
						}

						if (qMax > max64)
						{
							qMax = max64;
							if ((qMax & 1) == 0)
							{
								qMax -= 1;
							}
						}

						BigInteger stepBig = step64;
						BigInteger kMinBig = (qMin - 1 + (stepBig - 1)) / stepBig;
						BigInteger kMaxBig = (qMax - 1) / stepBig;

						if (kMinBig < minK || kMinBig < effectiveResume)
						{
							kMinBig = BigInteger.Max(minK, effectiveResume);
						}

						if (kMinBig > kMaxBig)
						{
							continue;
						}

						if (kMaxBig > ulong.MaxValue)
						{
							kMaxBig = ulong.MaxValue;
						}

						ulong kStart = (ulong)kMinBig;
						ulong kEnd = (ulong)kMaxBig;

				bool composite = CheckDivisors64Range(
					prime,
					step64,
					max64,
					kStart,
					kEnd,
					ref currentK,
					out bool processedRange,
					out foundDivisor);
				if (composite)
				{
					RecordPow2Hit(phase);
					FlushPow2State(currentK, phase);
					_currentPow2Phase = Pow2Phase.None;
					return true;
				}

						if (!processedRange)
						{
							processedAll = false;
						}
					}
				}

				for (int i = 0; i < effectiveRanges.Length; i++)
				{
					int start = effectiveRanges[i].Start;
					int end = effectiveRanges[i].End;

					BigInteger scaledStart = (PercentScale + start) * lowerBig;
					BigInteger scaledEnd = (PercentScale + end) * lowerBig;

					BigInteger qMin = (scaledStart + (PercentScale - 1)) / PercentScale;
					BigInteger qMax = scaledEnd / PercentScale;

					if ((qMin & 1) == 0)
					{
						qMin += 1;
					}

					if ((qMax & 1) == 0)
					{
						qMax -= 1;
					}

					if (qMin < 3 || qMin > qMax)
					{
						continue;
					}

					if (qMin < lowerBig)
					{
						qMin = lowerBig | 1;
					}

					if (qMax >= upperExclusive)
					{
						qMax = upperExclusive - 1;
						if ((qMax & 1) == 0)
						{
							qMax -= 1;
						}
					}

					if (qMin > qMax || qMin > max64)
					{
						continue;
					}

					if (qMax > max64)
					{
						qMax = max64;
						if ((qMax & 1) == 0)
						{
							qMax -= 1;
						}
					}

					BigInteger stepBig = step64;
					BigInteger kMinBig = (qMin - 1 + (stepBig - 1)) / stepBig;
					BigInteger kMaxBig = (qMax - 1) / stepBig;

					BigInteger kStartBig = kMinBig - BigInteger.One;
					if (kStartBig < BigInteger.One)
					{
						kStartBig = kMinBig;
					}

					if (kStartBig < minK || kStartBig < effectiveResume)
					{
						kStartBig = BigInteger.Max(BigInteger.Max(minK, effectiveResume), kMinBig);
					}

					if (kStartBig > kMaxBig)
					{
						continue;
					}

					if (kMaxBig > ulong.MaxValue)
					{
						kMaxBig = ulong.MaxValue;
					}

					ulong kStart = (ulong)kStartBig;
					ulong kEnd = (ulong)kMaxBig;

				bool composite = CheckDivisors64Range(
					prime,
					step64,
					max64,
					kStart,
					kEnd,
					ref currentK,
					out bool processedRange,
					out foundDivisor);
				if (composite)
				{
					RecordPow2Hit(phase);
					FlushPow2State(currentK, phase);
					_currentPow2Phase = Pow2Phase.None;
					return true;
				}

					if (!processedRange)
					{
						processedAll = false;
					}
				}
			}

			if (currentK >= (ulong)effectiveResume)
			{
				FlushPow2State(currentK, phase);
			}

			return false;
		}
		finally
		{
			if (currentK >= (ulong)effectiveResume)
			{
				FlushPow2State(currentK, phase);
			}

			_currentPow2Phase = Pow2Phase.None;
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private void EnsurePreparedSpecials(int specialRangePromille)
	{
		if (_preparedSpecialsInitialized && specialRangePromille == _preparedSpecialRangePromille)
		{
			return;
		}

		_preparedSpecialRangePromille = specialRangePromille;
		if (specialRangePromille == 0)
		{
			_preparedSpecialSingles = SpecialPercentPromille;
			_preparedSpecialRanges = Array.Empty<(int Start, int End)>();
			_preparedSpecialsInitialized = true;
			return;
		}

		int[] specials = SpecialPercentPromille;
		if (specials.Length == 0)
		{
			_preparedSpecialSingles = Array.Empty<int>();
			_preparedSpecialRanges = Array.Empty<(int Start, int End)>();
			_preparedSpecialsInitialized = true;
			return;
		}

		int[] sorted = new int[specials.Length];
		Array.Copy(specials, sorted, specials.Length);
		Array.Sort(sorted);

		int range = specialRangePromille;
		List<(int Start, int End)> merged = new(sorted.Length);
		int currentStart = Math.Max(0, sorted[0] - range);
		int currentEnd = Math.Min(PercentScale, sorted[0] + range);

		for (int i = 1; i < sorted.Length; i++)
		{
			int start = Math.Max(0, sorted[i] - range);
			int end = Math.Min(PercentScale, sorted[i] + range);
			if (start <= currentEnd)
			{
				if (end > currentEnd)
				{
					currentEnd = end;
				}
			}
			else
			{
				merged.Add((currentStart, currentEnd));
				currentStart = start;
				currentEnd = end;
			}
		}

		merged.Add((currentStart, currentEnd));
		_preparedSpecialRanges = merged.ToArray();
		_preparedSpecialSingles = Array.Empty<int>();
		_preparedSpecialsInitialized = true;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private void FlushPow2State(BigInteger k, Pow2Phase phase)
	{
		if (phase == Pow2Phase.None || k < MinK)
		{
			return;
		}

		string targetPath = phase == Pow2Phase.Special ? _specialStateFilePath : _groupsStateFilePath;
		ref int counter = ref (phase == Pow2Phase.Special ? ref _specialStateCounter : ref _groupsStateCounter);
		ref BigInteger last = ref (phase == Pow2Phase.Special ? ref _specialLastSavedK : ref _groupsLastSavedK);

		if (k <= last)
		{
			return;
		}

		string? directory = Path.GetDirectoryName(targetPath);
		if (!string.IsNullOrEmpty(directory))
		{
			Directory.CreateDirectory(directory);
		}

		File.AppendAllText(targetPath, k.ToString(CultureInfo.InvariantCulture) + Environment.NewLine);
		counter = 0;
		last = k;
	}
#endif

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private bool CheckDivisors128Range(
		ulong prime,
		ushort primeDecimalMask,
		UInt128 step,
		BigInteger allowedMax,
		UInt128 startK,
		UInt128 endK,
		out BigInteger foundDivisor,
		out bool processedAll)
	{
		foundDivisor = BigInteger.Zero;
		processedAll = true;

		if (startK > endK)
		{
			return false;
		}

		UInt128 allowedMax128 = allowedMax <= BigIntegerNumbers.UInt128MaxValue ? (UInt128)allowedMax : UInt128.MaxValue;
		UInt128 maxKAllowed = (allowedMax128 - UInt128.One) / step;
		if (endK > maxKAllowed)
		{
			endK = maxKAllowed;
			if (startK > endK)
			{
				return false;
			}
		}

		UInt128 divisor = (step * startK) + UInt128.One;
		processedAll = true;

		for (UInt128 k = startK; k <= endK; k++)
		{
			ushort rem3 = (ushort)(divisor % 3);
			ushort rem5 = (ushort)(divisor % 5);
			ushort rem7 = (ushort)(divisor % 7);
			ushort rem11 = (ushort)(divisor % 11);
			ushort rem13 = (ushort)(divisor % 13);
			ushort rem17 = (ushort)(divisor % 17);
			ushort rem19 = (ushort)(divisor % 19);

			if (rem3 != 0 && rem5 != 0 && rem7 != 0 && rem11 != 0 && rem13 != 0 && rem17 != 0 && rem19 != 0)
			{
				int rem10 = (int)(divisor % 10);
				ushort rem8 = (ushort)(divisor % 8);
				if ((rem8 == 1 || rem8 == 7) && ((primeDecimalMask >> rem10) & 1) != 0)
				{
					BigInteger divisorBig = (BigInteger)divisor;
					RecordState((BigInteger)k);
					BigInteger powResult = BigInteger.ModPow(2, prime, divisorBig);
					if (powResult.IsOne && !IsMersenneValue(prime, divisorBig))
					{
						foundDivisor = divisorBig;
						return true;
					}
				}
			}

			if (k == endK)
			{
				break;
			}

			divisor += step;
		}

		return false;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private bool CheckDivisorsLarge(
		ulong prime,
		ushort primeDecimalMask,
		BigInteger allowedMax,
		BigInteger currentK,
		BigInteger step,
		out bool processedAll,
		out BigInteger foundDivisor)
	{
		foundDivisor = BigInteger.Zero;
		if (allowedMax.IsZero)
		{
			processedAll = false;
			return false;
		}

		processedAll = true;
		BigInteger divisor = (step * currentK) + BigInteger.One;
		if (divisor > allowedMax)
		{
			return false;
		}

		BigInteger maxKAllowed = (allowedMax - BigInteger.One) / step;
		TryClampMaxKForMersenneTail(prime, step, ref maxKAllowed);
		if (currentK > maxKAllowed)
		{
			return false;
		}

		BigInteger remainingIterationsBig = maxKAllowed - currentK + BigInteger.One;

		if (step <= ulong.MaxValue)
		{
			UInt128 step128 = (UInt128)(ulong)step;
			UInt128 maxKByStep = UInt128Numbers.MaxValueMinusOne / step128;
			bool allowedFits128 = allowedMax <= BigIntegerNumbers.UInt128MaxValue;
			if (allowedFits128)
			{
				UInt128 limitK = (((UInt128)allowedMax) - UInt128.One) / step128;
				if (limitK < maxKByStep)
				{
					maxKByStep = limitK;
				}
			}

			BigInteger maxKByStepBig = (BigInteger)maxKByStep;
			if (maxKByStepBig >= currentK)
			{
				if (currentK > BigIntegerNumbers.UInt128MaxValue)
				{
					currentK = maxKByStepBig + BigInteger.One;
				}
				else
				{
					UInt128 startK = (UInt128)currentK;
					bool composite128 = CheckDivisors128Range(
						prime,
						primeDecimalMask,
						step128,
						allowedMax,
						startK,
						maxKByStep,
						out BigInteger found128,
						out bool processed128);
					if (composite128)
					{
						foundDivisor = found128;
						return true;
					}

					if (!processed128)
					{
						processedAll = false;
					}

					currentK = maxKByStepBig + BigInteger.One;
					divisor = (step * currentK) + BigInteger.One;
					if (divisor > allowedMax)
					{
						return false;
					}
				}
			}
		}

		var rem10 = new BigIntegerCycleRemainderStepper(10);
		var rem8 = new BigIntegerCycleRemainderStepper(8);
		var rem3 = new BigIntegerCycleRemainderStepper(3);
		var rem7 = new BigIntegerCycleRemainderStepper(7);
		var rem11 = new BigIntegerCycleRemainderStepper(11);
		var rem13 = new BigIntegerCycleRemainderStepper(13);
		var rem17 = new BigIntegerCycleRemainderStepper(17);
		var rem19 = new BigIntegerCycleRemainderStepper(19);

		byte remainder10 = rem10.Initialize(divisor);
		BigInteger primeBig = (BigInteger)prime;
		while (remainingIterationsBig.Sign > 0)
		{
			bool passesSmallModuli = remainder10 == 1 || remainder10 == 7;
			if (passesSmallModuli && ((primeDecimalMask >> remainder10) & 1) != 0)
			{
				if (rem3.NextIsNotDivisible(divisor) && 
					rem7.NextIsNotDivisible(divisor) &&
					rem11.NextIsNotDivisible(divisor) &&
					rem13.NextIsNotDivisible(divisor) &&
					rem17.NextIsNotDivisible(divisor) &&
					rem19.NextIsNotDivisible(divisor))
				{
					byte remainder8 = rem8.ComputeNext(divisor);
					if (remainder8 == 1 || remainder8 == 7)
					{
						RecordState(currentK);
						BigInteger powResult = BigInteger.ModPow(BigIntegerNumbers.Two, primeBig, divisor);
						if (powResult.IsOne && !IsMersenneValue(prime, divisor))
						{
							foundDivisor = divisor;
							processedAll = true;
							return true;
						}
					}
				}
			}

			currentK++;
			divisor += step;
			remainder10 = rem10.ComputeNext(divisor);

			remainingIterationsBig--;
			if (remainingIterationsBig.IsZero)
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

		ulong maxKAllowed = (allowedMax - 1UL) / step;
		{
			BigInteger maxKBig = maxKAllowed;
			if (TryClampMaxKForMersenneTail(prime, (BigInteger)step, ref maxKBig))
			{
				maxKAllowed = maxKBig > ulong.MaxValue ? ulong.MaxValue : (ulong)maxKBig;
			}
		}
		if (endK > maxKAllowed)
		{
			endK = maxKAllowed;
			if (startK > endK)
			{
				processedAll = true;
				foundDivisor = 0UL;
				return false;
			}
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
		processedAll = true;
		foundDivisor = 0UL;
		ulong computedCycle;
		bool primeOrderFailed;
		ulong divisorCycle;
		bool computed;
		if (step > limit)
		{
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

		ulong maxKAllowed = ((limit - 1UL) / step);
		ulong targetK = ((divisor - 1UL) / step);
		if (targetK > maxKAllowed)
		{
			processedAll = true;
			foundDivisor = 0UL;
			return false;
		}

		ulong remainingIterations = (maxKAllowed - targetK) + 1UL;
		while (remainingIterations > 0UL)
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
				if (divisorCycle == prime && !IsMersenneValue(prime, divisor))
				{
					foundDivisor = divisor;
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
				return false;
			}

			divisor += step;
			currentK++;
			residueStepper.Advance();
		}

		processedAll = true;
		foundDivisor = 0UL;
		return false;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private void RecordState(BigInteger k)
	{
#if DivisorSet_Pow2Groups
		if (_currentPow2Phase == Pow2Phase.Special)
		{
			RecordStateInternal(k, _specialStateFilePath, ref _specialStateCounter, ref _specialLastSavedK, _currentPow2Phase);
			return;
		}

		if (_currentPow2Phase == Pow2Phase.Groups)
		{
			RecordStateInternal(k, _groupsStateFilePath, ref _groupsStateCounter, ref _groupsLastSavedK, _currentPow2Phase);
			return;
		}
#endif
		RecordStateInternal(k, StateFilePath, ref _stateCounter, ref _lastSavedK, default);
	}

#if DivisorSet_Pow2Groups
	private void RecordStateInternal(BigInteger k, string path, ref int counter, ref BigInteger lastSavedK, Pow2Phase phase)
#else
	private void RecordStateInternal(BigInteger k, string path, ref int counter, ref BigInteger lastSavedK, object _)
#endif
	{
		if (k <= lastSavedK)
		{
			return;
		}

		int next = counter + 1;
		if (next >= PerfectNumberConstants.ByDivisorStateSaveInterval)
		{
			string? directory = Path.GetDirectoryName(path);
			if (!string.IsNullOrEmpty(directory))
			{
				Directory.CreateDirectory(directory);
			}

			File.AppendAllText(path, k.ToString(CultureInfo.InvariantCulture) + Environment.NewLine);
			counter = 0;
			lastSavedK = k;
		}
		else
		{
			counter = next;
		}
	}

#if DivisorSet_Pow2Groups
	// Overload used from CheckDivisors64Range through RecordState with current phase routed automatically.
	private void RecordState(BigInteger k, Pow2Phase phase)
	{
		switch (phase)
		{
			case Pow2Phase.Special:
				RecordStateInternal(k, _specialStateFilePath, ref _specialStateCounter, ref _specialLastSavedK, phase);
				break;
			case Pow2Phase.Groups:
				RecordStateInternal(k, _groupsStateFilePath, ref _groupsStateCounter, ref _groupsLastSavedK, phase);
				break;
			default:
				RecordStateInternal(k, StateFilePath, ref _stateCounter, ref _lastSavedK, phase);
				break;
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static void RecordPow2Hit(Pow2Phase phase)
	{
		long special = Interlocked.Read(ref _specialHits);
		long groups = Interlocked.Read(ref _groupHits);
		if (phase == Pow2Phase.Special)
		{
			special = Interlocked.Increment(ref _specialHits);
		}
		else if (phase == Pow2Phase.Groups)
		{
			groups = Interlocked.Increment(ref _groupHits);
		}

		Console.WriteLine($"By-divisor pow2groups hits: specials={special}, groups={groups}");
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static int GetBitLength(BigInteger value)
	{
		if (value.IsZero)
		{
			return 0;
		}

		byte[] bytes = value.ToByteArray();
		int lastIndex = bytes.Length - 1;
		return (lastIndex * 8) + BitOperations.Log2(bytes[lastIndex]) + 1;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private bool ProcessPow2PhaseBig(
		ulong prime,
		ushort primeDecimalMask,
		BigInteger allowedMax,
		BigInteger step,
		BigInteger resumeK,
		BigInteger minK,
		BigInteger segmentStartK,
		BigInteger segmentEndK,
		Pow2Phase phase,
		ReadOnlySpan<(int Start, int End)> ranges,
		ReadOnlySpan<int> singles,
		ref bool processedAll,
		out BigInteger foundDivisor)
	{
		foundDivisor = BigInteger.Zero;
		if (allowedMax.IsZero || segmentStartK > segmentEndK)
		{
			return false;
		}

		int specialRangePromille = EnvironmentConfiguration.ByDivisorSpecialRange;
		if (specialRangePromille < 0)
		{
			specialRangePromille = 0;
		}
		specialRangePromille *= 100; // convert percent to promille-of-percentScale (10_000 = 100%)
		EnsurePreparedSpecials(specialRangePromille);

		BigInteger effectiveStart = segmentStartK;
		if (resumeK > effectiveStart)
		{
			effectiveStart = resumeK;
		}

		if (effectiveStart < minK)
		{
			effectiveStart = minK;
		}

		if (effectiveStart > segmentEndK)
		{
			return false;
		}

		BigInteger firstDivisor = (step * effectiveStart) + BigInteger.One;
		if (firstDivisor > allowedMax)
		{
			processedAll = false;
			return false;
		}

		int startBit = GetBitLength(firstDivisor) - 1;
		int maxBit = GetBitLength(allowedMax) - 1;
		_currentPow2Phase = phase;
		FlushPow2State(effectiveStart, phase);

		ReadOnlySpan<int> effectiveSingles = singles;
		ReadOnlySpan<(int Start, int End)> effectiveRanges = ranges;
		if (phase == Pow2Phase.Special)
		{
			effectiveSingles = _preparedSpecialSingles;
			effectiveRanges = _preparedSpecialRanges;
		}

		try
		{
			for (int bit = startBit; bit <= maxBit; bit++)
			{
				BigInteger lower = BigInteger.One << bit;
				BigInteger upperExclusive = BigInteger.One << (bit + 1);

				if (lower > allowedMax)
				{
					break;
				}

				for (int i = 0; i < effectiveSingles.Length; i++)
				{
					int startProm = effectiveSingles[i];
					int endProm = effectiveSingles[i];
					if (specialRangePromille > 0)
					{
						startProm = Math.Max(0, effectiveSingles[i] - specialRangePromille);
						endProm = Math.Min(PercentScale, effectiveSingles[i] + specialRangePromille);
					}

					BigInteger scaledStart = (PercentScale + startProm) * lower;
					BigInteger scaledEnd = (PercentScale + (specialRangePromille == 0 ? Math.Min(PercentScale, startProm + 1) : endProm)) * lower;

					BigInteger qMin = (scaledStart + (PercentScale - 1)) / PercentScale;
					BigInteger qMax = scaledEnd / PercentScale;

					if (qMin.IsEven)
					{
						qMin += BigInteger.One;
					}

					if (qMax.IsEven)
					{
						qMax -= BigInteger.One;
					}

					if (qMin < 3 || qMin > qMax)
					{
						continue;
					}

					if (qMin < lower)
					{
						qMin = lower | BigInteger.One;
					}

					if (qMax >= upperExclusive)
					{
						qMax = upperExclusive - BigInteger.One;
						if ((qMax & BigInteger.One) == BigInteger.Zero)
						{
							qMax -= BigInteger.One;
						}
					}

					if (qMin > qMax || qMin > allowedMax)
					{
						continue;
					}

					if (qMax > allowedMax)
					{
						qMax = allowedMax;
						if ((qMax & BigInteger.One) == BigInteger.Zero)
						{
							qMax -= BigInteger.One;
						}
					}

					BigInteger kMin = (qMin - BigInteger.One + (step - BigInteger.One)) / step;
					BigInteger kMax = (qMax - BigInteger.One) / step;

					if (kMin < minK || kMin < effectiveStart)
					{
						kMin = BigInteger.Max(BigInteger.Max(minK, effectiveStart), kMin);
					}

					if (kMax > segmentEndK)
					{
						kMax = segmentEndK;
					}

					if (kMin > kMax)
					{
						continue;
					}

					BigInteger currentK = kMin;
					bool composite = CheckDivisorsBigRangePow2(
						prime,
						primeDecimalMask,
						step,
						allowedMax,
						kMin,
						kMax,
						phase,
						ref currentK,
						out bool processedRange,
						out BigInteger foundBig);
					if (composite)
					{
						RecordPow2Hit(phase);
						FlushPow2State(currentK, phase);
						_currentPow2Phase = Pow2Phase.None;
						foundDivisor = foundBig;
						return true;
					}

					if (!processedRange)
					{
						processedAll = false;
					}
				}

				for (int i = 0; i < effectiveRanges.Length; i++)
				{
					int startProm = effectiveRanges[i].Start;
					int endProm = effectiveRanges[i].End;
					if (specialRangePromille > 0 && phase == Pow2Phase.Special)
					{
						startProm = Math.Max(0, startProm - specialRangePromille);
						endProm = Math.Min(PercentScale, endProm + specialRangePromille);
					}

					BigInteger scaledStart = (PercentScale + startProm) * lower;
					BigInteger scaledEnd = (PercentScale + endProm) * lower;

					BigInteger qMin = (scaledStart + (PercentScale - 1)) / PercentScale;
					BigInteger qMax = scaledEnd / PercentScale;

					if ((qMin & BigInteger.One) == BigInteger.Zero)
					{
						qMin += BigInteger.One;
					}

					if ((qMax & BigInteger.One) == BigInteger.Zero)
					{
						qMax -= BigInteger.One;
					}

					if (qMin < 3 || qMin > qMax)
					{
						continue;
					}

					if (qMin < lower)
					{
						qMin = lower | BigInteger.One;
					}

					if (qMax >= upperExclusive)
					{
						qMax = upperExclusive - BigInteger.One;
						if ((qMax & BigInteger.One) == BigInteger.Zero)
						{
							qMax -= BigInteger.One;
						}
					}

					if (qMin > qMax || qMin > allowedMax)
					{
						continue;
					}

					if (qMax > allowedMax)
					{
						qMax = allowedMax;
						if ((qMax & BigInteger.One) == BigInteger.Zero)
						{
							qMax -= BigInteger.One;
						}
					}

					BigInteger kMin = (qMin - BigInteger.One + (step - BigInteger.One)) / step;
					BigInteger kMax = (qMax - BigInteger.One) / step;

					if (kMin < minK || kMin < effectiveStart)
					{
						kMin = BigInteger.Max(BigInteger.Max(minK, effectiveStart), kMin);
					}

					if (kMax > segmentEndK)
					{
						kMax = segmentEndK;
					}

					if (kMin > kMax)
					{
						continue;
					}

					BigInteger currentK = kMin;
					bool composite = CheckDivisorsBigRangePow2(
						prime,
						primeDecimalMask,
						step,
						allowedMax,
						kMin,
						kMax,
						phase,
						ref currentK,
						out bool processedRange,
						out BigInteger foundBig);
					if (composite)
					{
						RecordPow2Hit(phase);
						FlushPow2State(currentK, phase);
						_currentPow2Phase = Pow2Phase.None;
						foundDivisor = foundBig;
						return true;
					}

					if (!processedRange)
					{
						processedAll = false;
					}
				}
			}
		}
		finally
		{
			_currentPow2Phase = Pow2Phase.None;
		}

		return false;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private bool CheckDivisorsBigRangePow2(
		ulong prime,
		ushort primeDecimalMask,
		BigInteger step,
		BigInteger allowedMax,
		BigInteger startK,
		BigInteger endK,
		Pow2Phase phase,
		ref BigInteger currentK,
		out bool processedAll,
		out BigInteger foundDivisor)
	{
		if (startK < BigInteger.One || endK < startK)
		{
			processedAll = true;
			foundDivisor = BigInteger.Zero;
			return false;
		}

		currentK = startK;
		BigInteger divisor = (step * currentK) + BigInteger.One;
		if (divisor > allowedMax && !allowedMax.IsZero)
		{
			processedAll = false;
			foundDivisor = BigInteger.Zero;
			return false;
		}

		BigInteger iterations = (endK - currentK) + BigInteger.One;

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

		while (iterations > BigInteger.Zero)
		{
			bool passesSmallModuli = remainder3 != 0 && remainder5 != 0 && remainder7 != 0 && remainder11 != 0 && remainder13 != 0 && remainder17 != 0 && remainder19 != 0;
			if (passesSmallModuli && (remainder8 == 1 || remainder8 == 7) && ((primeDecimalMask >> remainder10) & 1) != 0)
			{
				RecordState(currentK, phase);
				BigInteger powResult = BigInteger.ModPow(2, prime, divisor);
				if (powResult.IsOne && !IsMersenneValue(prime, divisor))
				{
					foundDivisor = divisor;
					processedAll = true;
					return true;
				}
			}

			currentK += BigInteger.One;
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

			iterations--;
		}

		processedAll = true;
		foundDivisor = BigInteger.Zero;
		return false;
	}
#endif

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private void MarkPow2MinusOneChecked()
	{
		_pow2MinusOneChecked = true;
		if (!string.IsNullOrEmpty(_pow2MinusOneStateFilePath))
		{
			try
			{
				string? directory = Path.GetDirectoryName(_pow2MinusOneStateFilePath);
				if (!string.IsNullOrEmpty(directory))
				{
					Directory.CreateDirectory(directory);
				}

				File.WriteAllText(_pow2MinusOneStateFilePath, "checked");
			}
			catch
			{
				// Best-effort persistence; ignore IO errors.
			}
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private bool TryFindPowerOfTwoMinusOneDivisor(ulong prime, BigInteger allowedMax, out BigInteger divisor)
	{
		divisor = BigInteger.Zero;
		if (_pow2MinusOneChecked)
		{
			return false;
		}

		if (!_checkedPow2MinusOne.TryAdd(prime, 0))
		{
			return false;
		}
		BigInteger limit = allowedMax.IsZero ? new BigInteger(ulong.MaxValue) : allowedMax;

		if (limit <= ulong.MaxValue)
		{
			ulong limit64 = (ulong)limit;
			for (int x = 2; x < 64; x++)
			{
				if ((ulong)x == prime)
				{
					continue;
				}

				ulong d = (1UL << x) - 1UL;
				if (d > limit64)
				{
					break;
				}

				if (BigInteger.ModPow(2, prime, d) == BigInteger.One)
				{
					divisor = d;
					return true;
				}
			}
		}

		BigInteger pow = BigIntegerNumbers.OneShiftedLeft63;
		for (int x = 63; ; x++)
		{
			pow <<= 1;
			BigInteger dBig = pow - BigInteger.One;
			if (dBig > limit)
			{
				break;
			}

			if ((ulong)x == prime)
			{
				continue;
			}

			if (BigInteger.ModPow(2, prime, dBig) == BigInteger.One)
			{
				divisor = dBig;
				_checkedPow2MinusOne[prime] = 1;
				MarkPow2MinusOneChecked();
				return true;
			}
		}

		_checkedPow2MinusOne[prime] = 1;
		MarkPow2MinusOneChecked();
		return false;
	}
}
