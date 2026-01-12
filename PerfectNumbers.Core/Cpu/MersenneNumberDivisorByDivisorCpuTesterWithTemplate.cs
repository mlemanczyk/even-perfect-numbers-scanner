// #define DivisorSet_BitContradiction
// #define DivisorSet_BitTree

using System;
using System.Diagnostics;
using System.Numerics;
using System.IO;
using System.Text;
using System.Runtime.CompilerServices;
using System.Globalization;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Buffers;
using PerfectNumbers.Core.Gpu;
using PerfectNumbers.Core.Gpu.Accelerators;
using PerfectNumbers.Core.ByDivisor;
using PerfectNumbers.Core.Persistence;

namespace PerfectNumbers.Core.Cpu;

public enum DivisorSet
{
	Sequential,
	Pow2Groups,
	Predictive,
	Percentile,
	Additive,
	TopDown,
	BitContradiction,
	BitTree,
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
	private Pow2Minus1StateRepository? _pow2MinusOneRepository;
	private KStateRepository _kRepository = null!;
	private KStateRepository? _cheapRepository;
	private BigInteger _cheapLastSavedK;
	private BigInteger _cheapResumeK;

	private GpuUInt128WorkSet _divisorScanGpuWorkSet;
	private static readonly ConcurrentDictionary<ulong, byte> _checkedPow2MinusOne = new();
	private ulong _currentPrime;
#if DivisorSet_BitContradiction
	private static Persistence.BitContradictionStateRepository? _bitContradictionStateRepository;
#endif
#if DivisorSet_BitTree
	private static Persistence.BitContradictionStateRepository? _bitTreeStateRepository;
#endif
#if DivisorSet_TopDown
	private BigInteger _topDownCursor;
#endif

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
	private KStateRepository? _specialRepository;
	private KStateRepository? _groupsRepository;
#endif

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool IsMersenneValue(ulong prime, in BigInteger divisor)
	{
		if (divisor <= BigInteger.One || prime == 0UL)
		{
			return false;
		}

		BigInteger plusOne = divisor + BigInteger.One;
		if (plusOne <= BigInteger.One || !IsPowerOfTwoBig(plusOne))
		{
			return false;
		}

		int exponentBits = GetBitLengthPortable(plusOne) - 1;
		if (exponentBits <= 0)
		{
			return false;
		}

		return prime == (ulong)exponentBits;
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
	private bool TryBuildCheapRange(in BigInteger minK, in BigInteger chunkSize, in BigInteger maxKAllowed, out BigInteger start, out BigInteger end)
	{
		start = minK > BigInteger.One ? minK : BigInteger.One;
		if (_cheapResumeK > start)
		{
			start = _cheapResumeK;
		}

		if (start > maxKAllowed)
		{
			end = BigInteger.Zero;
			return false;
		}

		BigInteger span = chunkSize > BigInteger.One ? chunkSize - BigInteger.One : BigInteger.Zero;
		end = start + span;
		if (end > maxKAllowed)
		{
			end = maxKAllowed;
		}

		return end >= start;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private void RecordCheapState(in BigInteger k)
	{
		if (_currentPrime == 0UL || _cheapRepository is null)
		{
			return;
		}

		if (k <= _cheapLastSavedK)
		{
			return;
		}

		_cheapRepository.Upsert(_currentPrime, k);
		_cheapLastSavedK = k;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public void ResetStateTracking()
	{
		_stateCounter = 0;
		_pow2MinusOneChecked = false;
		_currentPrime = 0UL;
		_cheapLastSavedK = BigInteger.Zero;
		_cheapResumeK = BigInteger.One;
#if DivisorSet_TopDown
		_topDownCursor = BigInteger.Zero;
#endif
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
		_kRepository = EnvironmentConfiguration.ByDivisorKStateRepository ?? throw new InvalidOperationException("By-divisor k-state repository not initialized.");
		_pow2MinusOneRepository = EnvironmentConfiguration.ByDivisorPow2Minus1Repository;
		_cheapRepository = EnvironmentConfiguration.ByDivisorCheapKStateRepository;
		_cheapLastSavedK = BigInteger.Zero;
		_cheapResumeK = MinK;

		string stateName = Path.GetFileNameWithoutExtension(stateFile);
		if (ulong.TryParse(stateName, NumberStyles.None, CultureInfo.InvariantCulture, out ulong parsedPrime))
		{
			_currentPrime = parsedPrime;
			if (_kRepository.TryGet(parsedPrime, out BigInteger storedK) && storedK > BigInteger.Zero)
			{
				_lastSavedK = storedK;
			}

			_pow2MinusOneChecked = _pow2MinusOneRepository?.IsChecked(parsedPrime) ?? false;
			if (_cheapRepository is not null && _cheapRepository.TryGet(parsedPrime, out BigInteger cheapStored) && cheapStored > BigInteger.Zero)
			{
				_cheapLastSavedK = cheapStored;
				_cheapResumeK = cheapStored + BigInteger.One;
			}
		}
#if DivisorSet_TopDown
		_topDownCursor = _lastSavedK;
#endif
#if DivisorSet_Pow2Groups
		_specialRepository = EnvironmentConfiguration.ByDivisorSpecialStateRepository;
		_groupsRepository = EnvironmentConfiguration.ByDivisorGroupsStateRepository;
		_specialStateCounter = 0;
		_groupsStateCounter = 0;
		_specialLastSavedK = lastSavedK;
		_groupsLastSavedK = lastSavedK;
		if (_currentPrime != 0UL)
		{
			if (_specialRepository is not null && _specialRepository.TryGet(_currentPrime, out BigInteger storedSpecial) && storedSpecial > BigInteger.Zero)
			{
				_specialLastSavedK = storedSpecial;
			}

			if (_groupsRepository is not null && _groupsRepository.TryGet(_currentPrime, out BigInteger storedGroups) && storedGroups > BigInteger.Zero)
			{
				_groupsLastSavedK = storedGroups;
			}
		}

		_specialResumeK = _specialLastSavedK + BigInteger.One;
		_groupsResumeK = _groupsLastSavedK + BigInteger.One;
		_currentPow2Phase = Pow2Phase.None;
		_preparedSpecialsInitialized = false;
#endif
#if DivisorSet_BitContradiction
		_bitContradictionStateRepository = EnvironmentConfiguration.BitContradictionStateRepository;
#endif
#if DivisorSet_BitTree
		_bitTreeStateRepository = EnvironmentConfiguration.BitContradictionStateRepository;
#endif
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	public bool IsPrime(ulong prime, out bool divisorsExhausted, out BigInteger divisor)
	{
		_currentPrime = prime;
		BigInteger allowedMax = MersenneNumberDivisorByDivisorTester.ComputeAllowedMaxDivisorBig(prime, DivisorLimit);

		// if (TryFindPowerOfTwoMinusOneDivisor(prime, allowedMax, out divisor))
		// {
		// 	divisorsExhausted = true;
		// 	return false;
		// }

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
		BigInteger normalizedMinK = minK >= BigInteger.One ? minK : BigInteger.One;
		BigInteger step = ((BigInteger)prime) << 1;
		BigInteger firstDivisor = (step * normalizedMinK) + BigInteger.One;

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
			// Console.WriteLine($"[by-divisor-plan] p={prime} suffix={(prime & 1023UL)} using pow2-groups plan; model skipped.");
			return compositePow2;
#else
#if DivisorSet_BitContradiction
		bool compositeBitContradiction = CheckDivisorsBitContradiction(
			prime,
			primeDecimalMask,
			allowedMax,
			normalizedMinK,
			step,
			out processedAll,
			out foundDivisor);
		return compositeBitContradiction;
#endif
#if DivisorSet_BitTree
		bool compositeBitTree = CheckDivisorsBitTree(
			prime,
			primeDecimalMask,
			allowedMax,
			normalizedMinK,
			step,
			out processedAll,
			out foundDivisor);
		return compositeBitTree;
#endif
#if DivisorSet_TopDown
		bool compositeTopDown = CheckDivisorsTopDown(
			prime,
			allowedMax,
			normalizedMinK,
			out processedAll,
			out foundDivisor);
		return compositeTopDown;
#endif
#if DivisorSet_Predictive || DivisorSet_Percentile || DivisorSet_Additive
		ByDivisorClassModel? classifier = EnvironmentConfiguration.ByDivisorClassModel;
		if (classifier is not null)
		{
			bool compositePlanned = TryExecutePlannedScan(
				classifier,
				prime,
				allowedMax,
				normalizedMinK,
				out processedAll,
				out foundDivisor);
			if (compositePlanned)
			{
				return true;
			}

			if (processedAll)
			{
				return false;
			}
		}
		else
		{
			Console.WriteLine($"[by-divisor-plan] p={prime} suffix={(prime & 1023UL)} model unavailable, falling back to sequential scan.");
		}
#endif

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
#if DivisorSet_BitTree
	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private bool CheckDivisorsBitTree(
		ulong prime,
		ushort primeDecimalMask,
		BigInteger allowedMax,
		BigInteger minK,
		BigInteger step,
		out bool processedAll,
		out BigInteger foundDivisor)
	{
		processedAll = true;
		foundDivisor = BigInteger.Zero;

		// Default: frontier/windowed scan over (q,a) space.
		MersenneCombinedDivisorScannerFullStreamExtended.QNibbleGenerator16.SelfTest(138000001);
		MersenneCombinedDivisorScannerFullStreamExtended.ScanResult scanResult = MersenneCombinedDivisorScannerFullStreamExtended.TryFindDivisorByStreamingScan(prime, 128, false, out BigInteger branchDivisor);
		// if (TryExactBitTreeCheck(prime, allowedMax, minK, step, out BigInteger branchDivisor))
		if (scanResult == MersenneCombinedDivisorScannerFullStreamExtended.ScanResult.FoundDivisor)
		{
			foundDivisor = branchDivisor;
			return true;
		}
		else if (scanResult == MersenneCombinedDivisorScannerFullStreamExtended.ScanResult.Candidate)
		{
			processedAll = false;
		}

		return false;
	}

	private bool TryBitTreeBranchScan(ulong prime, BigInteger allowedMax, out BigInteger divisor)
	{
		divisor = BigInteger.Zero;

		if (prime == 0UL)
		{
			return false;
		}

		int pBits = checked((int)prime);
		var stack = new Stack<(int Index, int Carry, BigInteger Q, BigInteger A)>();
		// q0 = 1, a0 = 1 (to make r0 = 1)
		stack.Push((Index: 1, Carry: 0, Q: BigInteger.One, A: BigInteger.One));

		while (stack.Count > 0)
		{
			var state = stack.Pop();
			int j = state.Index;
			int carry = state.Carry;
			BigInteger qVal = state.Q;
			BigInteger aVal = state.A;

			if (j >= pBits)
			{
				if (carry == 0)
				{
					if (allowedMax.IsZero || qVal <= allowedMax)
					{
						divisor = qVal;
						return true;
					}
				}
				continue;
			}

			for (int qBit = 0; qBit <= 1; qBit++)
			{
				BigInteger qCandidate = qVal;
				if (qBit == 1)
				{
					qCandidate |= (BigInteger.One << j);
				}

				// Sum over assigned bits up to column j (a_i and q_{j-i} known up to j).
				int sum = carry;
				for (int i = 0; i <= j; i++)
				{
					bool aBit = ((aVal >> i) & BigInteger.One) == BigInteger.One;
					bool qBitVal = ((qCandidate >> (j - i)) & BigInteger.One) == BigInteger.One;
					if (aBit && qBitVal)
					{
						sum += 1;
					}
				}

				// Required result bit = 1 for columns < p
				if ((sum & 1) != 1)
				{
					continue;
				}

				// Set a_j to satisfy parity (already implied by sum&1==1)
				BigInteger aNext = aVal;
				if (((aVal >> j) & BigInteger.One) == BigInteger.Zero)
				{
					// keep as 0; sum already odd
				}

				int carryNext = (sum - 1) >> 1;

				if (!allowedMax.IsZero && qCandidate > allowedMax)
				{
					continue;
				}

				stack.Push((j + 1, carryNext, qCandidate, aNext));
			}
		}

		return false;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static bool TryBitTreeFrontierScan(ulong prime, in BigInteger allowedMax, out BigInteger divisor)
	{
		divisor = BigInteger.Zero;

		const int WindowSize = 200_000_000;
		const int MemoryFrontierLimit = 1_000_000;
		const int DiskBatchMax = 100_000;

		int pBits;
		try
		{
			pBits = checked((int)prime);
		}
		catch (OverflowException)
		{
			Console.WriteLine($"[bittree-frontier] exponent too large to check p={prime}");
			return false;
		}

		int maxQBits = allowedMax.IsZero ? pBits : GetBitLengthPortable(allowedMax);
		if (maxQBits <= 0)
		{
			return false;
		}

		int hardLimit = pBits + maxQBits + 2;
		Span<int> initialOnes = stackalloc int[1];
		initialOnes[0] = 0;

		var frontier = new Queue<BitTreeFrontierState>(capacity: MemoryFrontierLimit);
		BigInteger initialQ = BigInteger.One;
		BigInteger initialA = BigInteger.One;
		frontier.Enqueue(new BitTreeFrontierState(
			column: 1,
			carry: 0,
			baseIndex: 0,
			qSuffix: initialQ,
			aSuffix: initialA,
			maxQIndex: 0,
			lastNonZeroA: 0,
			qOnes: initialOnes.ToArray()));
		using BitTreeFrontierRepository frontierRepo = BitTreeFrontierRepository.Open(Path.Combine("Checks"), "bittree.frontier.faster");

		Console.WriteLine($"[bittree-frontier] p={prime} active_paths=1 column=1");

		long prunedPaths = 0;
		long pruneLogThreshold = 1_000_000;
		const int ColumnLogStep = 1_000_000;
		int lastLoggedColumn = 1;
		void TrackPrune(int column)
		{
			prunedPaths++;
			if (prunedPaths >= pruneLogThreshold)
			{
				Console.WriteLine($"[bittree-frontier] p={prime} pruned_paths={prunedPaths} column={column}");
				pruneLogThreshold <<= 1;
			}
			if (column - lastLoggedColumn >= ColumnLogStep)
			{
				lastLoggedColumn = column;
				Console.WriteLine($"[bittree-frontier] p={prime} exploring_column={column}");
			}
		}

		int maxFrontier = 1;
		int printCount = 0;

		static byte[] SerializeState(in BitTreeFrontierState state)
		{
			using var ms = new MemoryStream();
			using var bw = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);
			bw.Write(state.Column);
			bw.Write(state.Carry);
			bw.Write(state.BaseIndex);
			bw.Write(state.MaxQIndex);
			bw.Write(state.LastNonZeroA);
			bw.Write(state.QOnes.Length);
			for (int i = 0; i < state.QOnes.Length; i++)
			{
				bw.Write(state.QOnes[i]);
			}
			byte[] qBytes = state.QSuffix.ToByteArray(isUnsigned: true, isBigEndian: false);
			bw.Write(qBytes.Length);
			bw.Write(qBytes);
			byte[] aBytes = state.ASuffix.ToByteArray(isUnsigned: true, isBigEndian: false);
			bw.Write(aBytes.Length);
			bw.Write(aBytes);
			bw.Flush();
			return ms.ToArray();
		}

		static BitTreeFrontierState DeserializeState(BinaryReader br)
		{
			int column = br.ReadInt32();
			int carry = br.ReadInt32();
			int baseIndex = br.ReadInt32();
			int maxQIndex = br.ReadInt32();
			int lastNonZeroA = br.ReadInt32();
			int onesLength = br.ReadInt32();
			int[] qOnes = new int[onesLength];
			for (int i = 0; i < onesLength; i++)
			{
				qOnes[i] = br.ReadInt32();
			}
			int qLen = br.ReadInt32();
			byte[] qBytes = br.ReadBytes(qLen);
			int aLen = br.ReadInt32();
			byte[] aBytes = br.ReadBytes(aLen);
			BigInteger qSuffix = new BigInteger(qBytes, isUnsigned: true, isBigEndian: false);
			BigInteger aSuffix = new BigInteger(aBytes, isUnsigned: true, isBigEndian: false);
			return new BitTreeFrontierState(column, carry, baseIndex, qSuffix, aSuffix, maxQIndex, lastNonZeroA, qOnes);
		}

		static byte[] SerializeBatch(IEnumerable<BitTreeFrontierState> states)
		{
			using var ms = new MemoryStream();
			using var bw = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);
			foreach (var s in states)
			{
				byte[] payload = SerializeState(in s);
				bw.Write(payload.Length);
				bw.Write(payload);
			}
			bw.Flush();
			return ms.ToArray();
		}

		static List<BitTreeFrontierState> DeserializeBatch(byte[] batch)
		{
			var list = new List<BitTreeFrontierState>();
			using var ms = new MemoryStream(batch);
			using var br = new BinaryReader(ms, Encoding.UTF8, leaveOpen: true);
			while (ms.Position < ms.Length)
			{
				int len = br.ReadInt32();
				byte[] payload = br.ReadBytes(len);
				using var itemStream = new MemoryStream(payload);
				using var itemReader = new BinaryReader(itemStream, Encoding.UTF8, leaveOpen: true);
				list.Add(DeserializeState(itemReader));
			}
			return list;
		}

		void SpillFrontierIfNeeded()
		{
			if (frontier.Count <= MemoryFrontierLimit)
			{
				return;
			}

			int toSpill = frontier.Count - MemoryFrontierLimit;
			if (toSpill <= 0 || toSpill < DiskBatchMax)
			{
				return;
			}

			var batchStates = new List<BitTreeFrontierState>(toSpill);
			for (int i = 0; i < toSpill && frontier.Count > 0; i++)
			{
				batchStates.Add(frontier.Dequeue());
			}

			byte[] batchBytes = SerializeBatch(batchStates);
			frontierRepo.AppendBatch(batchBytes);
			// Console.WriteLine($"[bittree-frontier] spilled {batchStates.Count} states to disk; in-memory={frontier.Count}");
		}

		bool ReloadFrontierFromRepo()
		{
			if (frontier.Count > 0)
			{
				return true;
			}

			if (frontierRepo.TryDequeueBatch(1, out List<byte[]> batches) && batches.Count > 0)
			{
				foreach (byte[] b in batches)
				{
					foreach (var s in DeserializeBatch(b))
					{
						frontier.Enqueue(s);
					}
				}
				Console.WriteLine($"[bittree-frontier] reloaded {frontier.Count} states from disk.");
				return frontier.Count > 0;
			}

			return false;
		}


		while (frontier.Count > 0)
		{
			BitTreeFrontierState state = frontier.Dequeue();
			if (state.Column > hardLimit)
			{
				TrackPrune(state.Column);
				continue;
			}

			state = SlideWindow(state, WindowSize);

			for (int qBitChoice = 0; qBitChoice <= 1; qBitChoice++)
			{
				BigInteger qCandidate = state.QSuffix;
				int maxQIndex = state.MaxQIndex;
				int[] qOnes = state.QOnes;

				if (qBitChoice == 1)
				{
					int bitIndex = state.Column - state.BaseIndex;
					qCandidate |= BigInteger.One << bitIndex;
					maxQIndex = Math.Max(maxQIndex, state.Column);

					int[] extended = new int[qOnes.Length + 1];
					Array.Copy(qOnes, extended, qOnes.Length);
					extended[qOnes.Length] = state.Column;
					qOnes = extended;
				}

				BigInteger absoluteQ = qCandidate << state.BaseIndex;
				if (!allowedMax.IsZero && absoluteQ > allowedMax)
				{
					TrackPrune(state.Column);
					continue;
				}

				int qBitLength = maxQIndex + 1;
				if (qBitLength > maxQBits)
				{
					TrackPrune(state.Column);
					continue;
				}

				int maxAIndexAllowed = pBits - qBitLength;
				if (maxAIndexAllowed < 0)
				{
					TrackPrune(state.Column);
					continue;
				}

				int sum = state.Carry;
				for (int i = 0; i < qOnes.Length; i++)
				{
					int offset = qOnes[i];
					if (offset == 0 || offset > state.Column)
					{
						continue;
					}

					int aIndex = state.Column - offset;
					int aBitIndex = aIndex - state.BaseIndex;
					if (aBitIndex >= 0 && aBitIndex < WindowSize && ((state.ASuffix >> aBitIndex) & BigInteger.One) == BigInteger.One)
					{
						sum += 1;
					}
				}

				int requiredParity = state.Column < pBits ? 1 : 0;
				int parity = sum & 1;
				int aBit = parity == requiredParity ? 0 : 1;

				BigInteger aCandidate = state.ASuffix;
				int aBitIndexCurrent = state.Column - state.BaseIndex;
				if (aBit == 1)
				{
					if (aBitIndexCurrent >= WindowSize)
					{
						TrackPrune(state.Column);
						continue;
					}
					aCandidate |= BigInteger.One << aBitIndexCurrent;
				}

				int lastNonZeroA = aBit == 1 ? state.Column : state.LastNonZeroA;
				if (lastNonZeroA > maxAIndexAllowed)
				{
					TrackPrune(state.Column);
					continue;
				}

				int columnTotal = sum + aBit;
				int carry = columnTotal >> 1;

				bool doneWithOnes = lastNonZeroA < 0 || (state.Column + 1 - lastNonZeroA) > maxQIndex;
				if (state.Column + 1 >= pBits && carry == 0 && doneWithOnes)
				{
					divisor = absoluteQ;
					return true;
				}

				if (state.Column + 1 > hardLimit)
				{
					TrackPrune(state.Column);
					continue;
				}

				int[] nextOnes = qOnes;
				if (qBitChoice == 1)
				{
					nextOnes = new int[qOnes.Length + 1];
					Array.Copy(qOnes, nextOnes, qOnes.Length);
					nextOnes[^1] = state.Column;
				}

				frontier.Enqueue(new BitTreeFrontierState(
					column: state.Column + 1,
					carry: carry,
					baseIndex: state.BaseIndex,
					qSuffix: qCandidate,
					aSuffix: aCandidate,
					maxQIndex: maxQIndex,
					lastNonZeroA: lastNonZeroA,
					qOnes: nextOnes));

				SpillFrontierIfNeeded();

				if (frontier.Count > maxFrontier)
				{
					maxFrontier = frontier.Count;
					printCount++;
					if (printCount == 10_000_000)
					{
						printCount = 0;
						Console.WriteLine($"[bittree-frontier] p={prime} active_paths={maxFrontier} column={state.Column + 1}");
					}
				}
			}

			if (frontier.Count == 0 && !ReloadFrontierFromRepo())
			{
				break;
			}
		}

		return false;
	}

	private readonly struct BitTreeFrontierState
	{
		public BitTreeFrontierState(
			int column,
			int carry,
			int baseIndex,
			BigInteger qSuffix,
			BigInteger aSuffix,
			int maxQIndex,
			int lastNonZeroA,
			int[] qOnes)
		{
			Column = column;
			Carry = carry;
			BaseIndex = baseIndex;
			QSuffix = qSuffix;
			ASuffix = aSuffix;
			MaxQIndex = maxQIndex;
			LastNonZeroA = lastNonZeroA;
			QOnes = qOnes;
		}

		public readonly int Column;
		public readonly int Carry;
		public readonly int BaseIndex;
		public readonly BigInteger QSuffix;
		public readonly BigInteger ASuffix;
		public readonly int MaxQIndex;
		public readonly int LastNonZeroA;
		public readonly int[] QOnes;
	}

	private readonly struct BitTreeBacktrackFrame
	{
		public BitTreeBacktrackFrame(BitTreeFrontierState state, int nextChoice)
		{
			State = state;
			NextChoice = nextChoice;
		}

	public BitTreeFrontierState State { get; }
	public int NextChoice { get; }

	public BitTreeBacktrackFrame WithNextChoice(int nextChoice)
	{
		return new BitTreeBacktrackFrame(State, nextChoice);
	}
}


	private static BitTreeFrontierState SlideWindow(in BitTreeFrontierState state, int windowSize)
	{
		int span = state.Column - state.BaseIndex + 1;
		if (span < windowSize)
		{
			return state;
		}

		int shift = span - (windowSize - 1);
		int newBase = state.BaseIndex + shift;

		BigInteger qSuffix = state.QSuffix >> shift;
		BigInteger aSuffix = state.ASuffix >> shift;

		List<int> filtered = null;
		for (int i = 0; i < state.QOnes.Length; i++)
		{
			int pos = state.QOnes[i];
			if (pos >= newBase)
			{
				filtered ??= new List<int>(state.QOnes.Length - i);
				filtered.Add(pos);
			}
		}

		int[] qOnes = filtered == null ? Array.Empty<int>() : filtered.ToArray();

		int lastNonZeroA = state.LastNonZeroA;
		if (lastNonZeroA < newBase)
		{
			lastNonZeroA = -1;
		}

		return new BitTreeFrontierState(
			column: state.Column,
			carry: state.Carry,
			baseIndex: newBase,
			qSuffix: qSuffix,
			aSuffix: aSuffix,
			maxQIndex: state.MaxQIndex,
			lastNonZeroA: lastNonZeroA,
			qOnes: qOnes);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private bool TryExactBitTreeCheck(
		ulong prime,
		BigInteger allowedMax,
		BigInteger currentK,
		BigInteger step,
		out BigInteger foundDivisor)
	{
		foundDivisor = BigInteger.Zero;

		bool hasLimit = !allowedMax.IsZero;
		BigInteger maxK = hasLimit ? (allowedMax - BigInteger.One) / step : BigInteger.Zero;
		ulong computedCycle;

		bool decided;
		bool divides;
		BigInteger divisor = (step * currentK) + BigInteger.One;
		while (true)
		{
// 			if (divisor <= ulong.MaxValue)
// 			{
// 				ulong divisor64 = (ulong)divisor;
// 				MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(divisor64);
// 				// We're reusing divides variable to limit registry pressure as primeOrderFailed flag.
// #if DEVICE_GPU
// 				decided = MersenneNumberDivisorByDivisorTester.TryCalculateCycleLengthForExponentGpu(
// 					divisor64,
// 					Accelerator,
// 					prime,
// 					divisorData,
// 					out computedCycle,
// 					out divides);
// #elif DEVICE_HYBRID
// 				decided = MersenneNumberDivisorByDivisorTester.TryCalculateCycleLengthForExponentHybrid(
// 					divisor64,
// 					Accelerator,
// 					prime,
// 					divisorData,
// 					out computedCycle,
// 					out divides);
// #else
// 				decided = MersenneNumberDivisorByDivisorTester.TryCalculateCycleLengthForExponentCpu(
// 					divisor64,
// 					prime,
// 					divisorData,
// 					out computedCycle,
// 					out divides);
// #endif

// 				if (!decided)
// 				{
// #if DEVICE_GPU
// 					computedCycle = MersenneNumberDivisorByDivisorTester.CalculateCycleLengthGpu(
// 						divisor64,
// 						Accelerator,
// 						divisorData,
// 						skipPrimeOrderHeuristic: divides);
// #elif DEVICE_HYBRID
// 					computedCycle = MersenneNumberDivisorByDivisorTester.CalculateCycleLengthHybrid(
// 						divisor64,
// 						Accelerator,
// 						divisorData,
// 						skipPrimeOrderHeuristic: divides);
// #else
// 					computedCycle = MersenneNumberDivisorByDivisorTester.CalculateCycleLengthCpu(
// 						divisor64,
// 						divisorData,
// 						skipPrimeOrderHeuristic: divides);
// #endif
// 				}

// 				if (computedCycle > 0UL)
// 				{
// 					if ((prime % computedCycle) != 0UL)
// 					{
// 						goto MOVE_NEXT;
// 					}
// 				}
// 				else
// 				{
// 					computedCycle = prime;
// 				}
// 			}
// 			else
// 			{
				computedCycle = prime;
			// }

			decided = TryExactBitTreeCheck(computedCycle, divisor, out divides);
			if (!decided)
			{
				throw new InvalidOperationException($"BitTree check did not decide for p={prime}, divisor={divisor} (k={currentK}).");
			}

			if (divides)
			{
				foundDivisor = divisor;
				return true;
			}

			RecordState(currentK);

		MOVE_NEXT:
			if (hasLimit && currentK >= maxK)
			{
				break;
			}

			currentK += BigInteger.One;
			divisor += step;
		}

		return false;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private bool TryExactBitTreeCheck(ulong exponent, BigInteger divisor, out bool divides)
	{
		Stopwatch sw = Stopwatch.StartNew();
		int qBitLength = GetBitLengthPortable(divisor);
		int aBitLength = GetBitLengthPortable(exponent);

		ArrayPool<int> intPool = ThreadStaticPools.IntPool;
		int pBits = checked((int)exponent);

		int[] oneOffsetsArr = intPool.Rent(qBitLength);
		int oneCount = 0;
		int i;
		for (i = 0; i < qBitLength; i++)
		{
			if (((divisor >> i) & BigInteger.One) == BigInteger.One)
			{
				oneOffsetsArr[oneCount++] = i;
			}
		}

		// This condition will never trigger on EvenPerfectBitScanner execution paths. We always provide valid q != 0.
		// if (oneCount == 0)
		// {
		// 	divides = false;
		// 	intPool.Return(oneOffsetsArr);
		// 	return true;
		// }

		// i's in oneOffsetsArr are built in ascending order by i. We don't need to sort it.
		// Array.Sort(oneOffsetsArr, 0, oneCount);
		ReadOnlySpan<int> oneOffsets = oneOffsetsArr.AsSpan(0, oneCount);
		int maxOffset = oneOffsets[^1];

		int windowSize = qBitLength;
		byte[] aWindowArray = ArrayPool<byte>.Shared.Rent(windowSize);
		Span<byte> aWindow = aWindowArray.AsSpan(0, windowSize);
		aWindow.Clear();

		long lastNonZeroA = -1;
		int carry = 0;
		int column = 0;

		while (true)
		{
			bool targetOne = column < pBits;
			int sum = carry;

			for (i = 0; i < oneCount; i++)
			{
				int offset = oneOffsets[i];
				if (offset == 0)
				{
					continue; // a_bit for current column, handled below
				}

				if (offset > column)
				{
					break;
				}

				int aIndex = column - offset;
				byte aBit = aWindow[aIndex % windowSize];
				sum += aBit;
			}

			int requiredParity = targetOne ? 1 : 0;
			int parity = sum & 1;
			byte aCurrent = (byte)(parity == requiredParity ? 0 : 1);
			int columnTotal = sum + aCurrent;

			if ((columnTotal & 1) != requiredParity)
			{
				ArrayPool<byte>.Shared.Return(aWindowArray);

				divides = false;
				intPool.Return(oneOffsetsArr);
				return true;
			}

			aWindow[column % windowSize] = aCurrent;
			if (aCurrent == 1)
			{
				lastNonZeroA = column;
			}

			carry = columnTotal >> 1;
			column++;

			bool doneWithOnes = lastNonZeroA < 0 || column - lastNonZeroA > maxOffset;
			if (column >= pBits && carry == 0 && doneWithOnes)
			{
				break;
			}

			if (column > pBits + maxOffset + 2 && carry == 0 && doneWithOnes)
			{
				break;
			}

			if (column > pBits + maxOffset + aBitLength + 2)
			{
				ArrayPool<byte>.Shared.Return(aWindowArray);

				divides = false;
				intPool.Return(oneOffsetsArr);
				return true;
			}
		}
		sw.Stop();

		ArrayPool<byte>.Shared.Return(aWindowArray);

		divides = carry == 0;
		var elapsed = sw.Elapsed;
		Console.WriteLine($"[bittree ({elapsed})] Prime={exponent}, Divisor={divisor} Decided=True Divides={divides}");
		if (!divides)
		{
			string message = $"[bittree] p={exponent} ruled out divisor={divisor} (parity/carry exhaustion)";
			_bitTreeStateRepository?.Upsert(exponent, message);
		}

		intPool.Return(oneOffsetsArr);
		return true;
	}

	// [MethodImpl(MethodImplOptions.AggressiveInlining)]
	// private bool TryExactBitTreeCheck(ulong prime, BigInteger divisor, ulong effectiveExponent, out bool divides)
	// {
	// 	divides = false;

	// 	// These conditions below will never be met on EvenPerfectBitScanner execution paths.
	// 	// if (prime == 0 || divisor <= BigInteger.One)
	// 	// {
	// 	// 	Console.WriteLine($"[bittree] invalid inputs {prime} / {divisor}");
	// 	// 	return false;
	// 	// }

	// 	// if ((divisor & BigInteger.One) == BigInteger.Zero)
	// 	// {
	// 	// 	divides = false;
	// 	// 	return true;
	// 	// }

	// 	int qBitLength = GetBitLengthPortable(divisor);
	// 	if (prime <= (ulong)qBitLength)
	// 	{
	// 		divides = false;
	// 		return true;
	// 	}

	// 	ulong exponent = effectiveExponent > 0UL ? effectiveExponent : prime;
	// 	int pBits;
	// 	try
	// 	{
	// 		pBits = checked((int)exponent);
	// 	}
	// 	catch (OverflowException)
	// 	{
	// 		Console.WriteLine($"[bittree] exponent too large to check p={exponent}");
	// 		return false;
	// 	}

	// 	int[] oneOffsetsArr = new int[qBitLength];
	// 	int oneCount = 0;
	// 	for (int i = 0; i < qBitLength; i++)
	// 	{
	// 		if (((divisor >> i) & BigInteger.One) == BigInteger.One)
	// 		{
	// 			oneOffsetsArr[oneCount++] = i;
	// 		}
	// 	}

	// 	if (oneCount == 0)
	// 	{
	// 		divides = false;
	// 		return true;
	// 	}

	// 	Array.Sort(oneOffsetsArr, 0, oneCount);
	// 	ReadOnlySpan<int> oneOffsets = oneOffsetsArr.AsSpan(0, oneCount);
	// 	int maxOffset = oneOffsets[^1];

	// 	int windowSize = qBitLength;
	// 	bool rented = windowSize <= 1024;
	// 	byte[] aWindowArray = rented ? ArrayPool<byte>.Shared.Rent(1024) : new byte[windowSize];
	// 	Span<byte> aWindow = aWindowArray.AsSpan(0, windowSize);
	// 	aWindow.Clear();

	// 	long lastNonZeroA = -1;
	// 	int carry = 0;
	// 	int column = 0;

	// 	while (true)
	// 	{
	// 		bool targetOne = column < pBits;
	// 		int sum = carry;

	// 		for (int idx = 0; idx < oneCount; idx++)
	// 		{
	// 			int offset = oneOffsets[idx];
	// 			if (offset == 0)
	// 			{
	// 				continue; // a_bit for current column, handled below
	// 			}

	// 			if (offset > column)
	// 			{
	// 				break;
	// 			}

	// 			int aIndex = column - offset;
	// 			byte aBit = aWindow[aIndex % windowSize];
	// 			sum += aBit;
	// 		}

	// 		int requiredParity = targetOne ? 1 : 0;
	// 		int parity = sum & 1;
	// 		byte aCurrent = (byte)(parity == requiredParity ? 0 : 1);
	// 		int columnTotal = sum + aCurrent;

	// 		if ((columnTotal & 1) != requiredParity)
	// 		{
	// 			if (rented)
	// 			{
	// 				ArrayPool<byte>.Shared.Return(aWindowArray);
	// 			}
	// 			divides = false;
	// 			return true;
	// 		}

	// 		aWindow[column % windowSize] = aCurrent;
	// 		if (aCurrent == 1)
	// 		{
	// 			lastNonZeroA = column;
	// 		}

	// 		carry = columnTotal >> 1;
	// 		column++;

	// 		bool doneWithOnes = lastNonZeroA < 0 || column - lastNonZeroA > maxOffset;
	// 		if (column >= pBits && carry == 0 && doneWithOnes)
	// 		{
	// 			break;
	// 		}

	// 		if (column > pBits + maxOffset + 2 && carry == 0 && doneWithOnes)
	// 		{
	// 			break;
	// 		}

	// 		if (column > pBits + maxOffset + windowSize + 2)
	// 		{
	// 			if (rented)
	// 			{
	// 				ArrayPool<byte>.Shared.Return(aWindowArray);
	// 			}
	// 			divides = false;
	// 			return true;
	// 		}
	// 	}

	// 	if (rented)
	// 	{
	// 		ArrayPool<byte>.Shared.Return(aWindowArray);
	// 	}
	// 	divides = carry == 0;
	// 	Console.WriteLine($"[bittree] Prime={prime}, Divisor={divisor} Decided=True Divides={divides}");
	// 	if (!divides)
	// 	{
	// 		string message = $"[bittree] p={prime} ruled out divisor={divisor} (parity/carry exhaustion)";
	// 		_bitTreeStateRepository?.Upsert(prime, message);
	// 	}

	// 	return true;
	// }
#endif

#if !DivisorSet_Pow2Groups && (DivisorSet_Predictive || DivisorSet_Percentile || DivisorSet_Additive)
	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private bool TryExecutePlannedScan(
		ByDivisorClassModel model,
		ulong prime,
		BigInteger allowedMax,
		BigInteger minK,
		out bool processedAll,
		out BigInteger foundDivisor)
	{
		processedAll = false;
		foundDivisor = BigInteger.Zero;

		if (minK > ulong.MaxValue)
		{
			// Console.WriteLine($"[by-divisor-plan] p={prime} using BigInteger plan (minK={minK}).");
			return TryExecutePlannedScanBig(model, prime, allowedMax, minK, out processedAll, out foundDivisor);
		}

		ulong allowedMax64;
		if (allowedMax.IsZero)
		{
			allowedMax64 = ulong.MaxValue;
		}
		else if (allowedMax > ulong.MaxValue)
		{
			// Console.WriteLine($"[by-divisor-plan] p={prime} using BigInteger plan (allowedMax={allowedMax}).");
			return TryExecutePlannedScanBig(model, prime, allowedMax, minK, out processedAll, out foundDivisor);
		}
		else
		{
			allowedMax64 = (ulong)allowedMax;
		}
		bool composite = TryExecutePlannedScan64(
			model,
			prime,
			allowedMax64,
			(ulong)minK,
			out processedAll,
			out ulong foundDivisor64);
		if (composite)
		{
			foundDivisor = foundDivisor64;
			return true;
		}

		if (processedAll)
		{
			foundDivisor = BigInteger.Zero;
		}

		return false;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private bool TryExecutePlannedScanBig(
		ByDivisorClassModel model,
		ulong prime,
		in BigInteger allowedMax,
		in BigInteger minK,
		out bool processedAll,
		out BigInteger foundDivisor)
	{
		processedAll = true;
		foundDivisor = BigInteger.Zero;

		if (allowedMax.IsZero)
		{
			processedAll = false;
			return false;
		}

		BigInteger step = ((BigInteger)prime) << 1;
		BigInteger maxKAllowed = (allowedMax - BigInteger.One) / step;
		TryClampMaxKForMersenneTail(prime, step, ref maxKAllowed);
		if (maxKAllowed <= BigInteger.Zero || minK > maxKAllowed)
		{
			processedAll = false;
			return false;
		}

		ByDivisorScanTuning tuning = EnvironmentConfiguration.ByDivisorScanTuning;
		ByDivisorScanPlan plan = model.BuildPlan(prime, allowedMax, tuning);
		int suffix = (int)(prime & 1023UL);
		_ = suffix;
		// ByDivisorClassEntry entry = model.Entries[suffix];
		// Console.WriteLine($"[by-divisor-plan] p={prime} suffix={suffix} k50={entry.K50:F2} k75={entry.K75:F2} gap={entry.GapProbability:F3} cheap<= {plan.CheapLimit:F0} start={plan.Start:F0} target={plan.Target:F0} rho1={plan.Rho1:F2} rho2={plan.Rho2:F2} maxK={maxKAllowed}");

		BigInteger cheapChunk = BigInteger.Min((BigInteger)plan.CheapLimit, maxKAllowed);
		BigInteger cheapStart = BigInteger.Zero;
		BigInteger cheapEnd = BigInteger.Zero;
		if (TryBuildCheapRange(minK, cheapChunk, maxKAllowed, out cheapStart, out cheapEnd))
		{
			BigInteger currentK = cheapStart;
			bool compositeCheap = CheckDivisorsBigPlanRange(
				prime,
				step,
				allowedMax,
				cheapStart,
				cheapEnd,
				out bool processedRange,
				out BigInteger foundRange,
				ref currentK);
			if (compositeCheap)
			{
				RecordCheapState(currentK);
				foundDivisor = foundRange;
				return true;
			}

			if (!processedRange)
			{
				processedAll = false;
				return false;
			}

			RecordCheapState(cheapEnd);
		}

		BigInteger startPlan = plan.Start <= 0d ? minK : new BigInteger(Math.Ceiling(plan.Start));
		if (startPlan < minK)
		{
			startPlan = minK;
		}

		BigInteger target = plan.Target <= 0d ? startPlan + BigInteger.One : new BigInteger(Math.Ceiling(plan.Target));
		if (target < startPlan)
		{
			target = startPlan + BigInteger.One;
		}

		BigInteger current = (cheapEnd > BigInteger.Zero ? cheapEnd + BigInteger.One : minK);
		if (current < startPlan)
		{
			current = startPlan;
		}

#if DivisorSet_Predictive
		bool loggedProgressive = false;
		while (current <= maxKAllowed)
		{
			if (!loggedProgressive)
			{
				// Console.WriteLine($"[by-divisor-progressive] p={prime} startK={current} target={target} rho={(current < target ? plan.Rho1 : plan.Rho2):F2} maxK={maxKAllowed}");
				loggedProgressive = true;
			}

			BigInteger startRange = current;
			BigInteger endRange = current;
			bool composite = CheckDivisorsBigPlanRange(
				prime,
				step,
				allowedMax,
				startRange,
				endRange,
				out bool processedRange,
				out BigInteger foundRange,
				ref startRange);
			if (composite)
			{
				foundDivisor = foundRange;
				return true;
			}

			if (!processedRange)
			{
				processedAll = false;
				return false;
			}

			double multiplier = current < target ? plan.Rho1 : plan.Rho2;
			BigInteger advanced = new BigInteger(Math.Ceiling((double)current * multiplier));
			if (advanced <= current)
			{
				advanced = current + BigInteger.One;
			}

			if (advanced > maxKAllowed)
			{
				break;
			}

			current = advanced;
		}
#elif DivisorSet_Percentile
		BigInteger k10 = plan.K10 > 0d ? new BigInteger(Math.Ceiling(plan.K10)) : current;
		BigInteger k25 = plan.K25 > 0d ? new BigInteger(Math.Ceiling(plan.K25)) : k10 + BigInteger.One;
		BigInteger k50 = plan.K50 > 0d ? new BigInteger(Math.Ceiling(plan.K50)) : k25 + BigInteger.One;
		BigInteger k75 = plan.K75 > 0d ? new BigInteger(Math.Ceiling(plan.K75)) : k50 + BigInteger.One;

		BigInteger[] stops = new BigInteger[]
		{
			k25 - BigInteger.One,
			k50 - BigInteger.One,
			k75 - BigInteger.One,
		};

		BigInteger startRangeK = current > k10 ? current : k10;
		for (int i = 0; i < stops.Length; i++)
		{
			BigInteger endRange = stops[i];
			if (endRange < startRangeK)
			{
				continue;
			}

			bool composite = CheckDivisorsBigPlanRange(
				prime,
				step,
				allowedMax,
				startRangeK,
				endRange,
				out bool processedRange,
				out BigInteger foundRange,
				ref startRangeK);
			if (composite)
			{
				foundDivisor = foundRange;
				return true;
			}

			if (!processedRange)
			{
				processedAll = false;
				return false;
			}

			startRangeK = endRange + BigInteger.One;
			if (startRangeK > maxKAllowed)
			{
				return false;
			}
		}

		// After percentile sweeps, escalate geometrically.
		BigInteger startGeo = startRangeK;
		if (startGeo < k75)
		{
			startGeo = k75;
		}

		bool loggedPercentile = false;
		while (startGeo <= maxKAllowed)
		{
			if (!loggedPercentile)
			{
				loggedPercentile = true;
			}

			BigInteger scanStart = startGeo;
			BigInteger scanEnd = scanStart;
			bool composite = CheckDivisorsBigPlanRange(
				prime,
				step,
				allowedMax,
				scanStart,
				scanEnd,
				out bool processedRange,
				out BigInteger foundRange,
				ref scanStart);
			if (composite)
			{
				foundDivisor = foundRange;
				return true;
			}

			if (!processedRange)
			{
				processedAll = false;
				return false;
			}

			BigInteger advanced = new BigInteger(Math.Ceiling((double)startGeo * plan.Rho2));
			if (advanced <= startGeo)
			{
				advanced = startGeo + BigInteger.One;
			}

			if (advanced > maxKAllowed)
			{
				break;
			}

			startGeo = advanced;
		}
#else
		BigInteger delta = plan.DeltaK > 0d ? new BigInteger(Math.Ceiling(plan.DeltaK)) : BigInteger.One;
		if (delta < BigInteger.One)
		{
			delta = BigInteger.One;
		}

		if (delta > maxKAllowed)
		{
			delta = maxKAllowed;
		}

		bool loggedAdditive = false;
		while (current <= maxKAllowed)
		{
			if (!loggedAdditive)
			{
				loggedAdditive = true;
			}

			BigInteger startRange = current;
			bool composite = CheckDivisorsBigPlanRange(
				prime,
				step,
				allowedMax,
				startRange,
				startRange,
				out bool processedRange,
				out BigInteger foundRange,
				ref startRange);
			if (composite)
			{
				foundDivisor = foundRange;
				return true;
			}

			if (!processedRange)
			{
				processedAll = false;
				return false;
			}

			BigInteger advanced = current + delta;
			if (advanced <= current)
			{
				advanced = current + BigInteger.One;
			}

			if (advanced > maxKAllowed)
			{
				break;
			}

			current = advanced;
		}
#endif

		return false;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private bool CheckDivisorsBigPlanRange(
		ulong prime,
		in BigInteger step,
		in BigInteger allowedMax,
		in BigInteger startK,
		in BigInteger endK,
		out bool processedAll,
		out BigInteger foundDivisor,
		ref BigInteger currentK)
	{
		processedAll = true;
		foundDivisor = BigInteger.Zero;

		if (startK < BigInteger.One || endK < startK)
		{
			return false;
		}

		BigInteger maxKAllowed = (allowedMax - BigInteger.One) / step;
		TryClampMaxKForMersenneTail(prime, step, ref maxKAllowed);
		if (startK > maxKAllowed)
		{
			return false;
		}

		BigInteger effectiveEnd = endK > maxKAllowed ? maxKAllowed : endK;
		ushort primeDecimalMask = DivisorGenerator.GetDecimalMask((prime & 3UL) == 3UL);
		BigInteger divisor = (step * startK) + BigInteger.One;
		if (divisor > allowedMax)
		{
			return false;
		}

		BigInteger iterations = (effectiveEnd - startK) + BigInteger.One;
		currentK = startK;

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

		while (iterations.Sign > 0)
		{
			if (remainder10 == 1 || remainder10 == 7)
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
						if (((primeDecimalMask >> remainder10) & 1) != 0)
						{
							RecordState(currentK);
							BigInteger powResult = BigInteger.ModPow(BigIntegerNumbers.Two, primeBig, divisor);
							// if (powResult.IsOne && !IsMersenneValue(prime, divisor))
							if (powResult.IsOne)
							{
								foundDivisor = divisor;
								return true;
							}
						}
					}
				}
			}

			iterations--;
			if (iterations.IsZero)
			{
				break;
			}

			currentK++;
			divisor += step;
			remainder10 = rem10.ComputeNext(divisor);
		}

		return false;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private bool TryExecutePlannedScan64(
		ByDivisorClassModel model,
		ulong prime,
		ulong allowedMax,
		ulong minK,
		out bool processedAll,
		out ulong foundDivisor)
	{
		processedAll = true;
		foundDivisor = 0UL;

		ulong step64 = prime << 1;
		if (step64 == 0UL)
		{
			processedAll = false;
			return false;
		}

		ByDivisorScanTuning tuning = EnvironmentConfiguration.ByDivisorScanTuning;
		ByDivisorScanPlan plan = model.BuildPlan(prime, allowedMax == ulong.MaxValue ? BigInteger.Zero : new BigInteger(allowedMax), tuning);
		int suffix = (int)(prime & 1023UL);
		_ = suffix;
		// ByDivisorClassEntry entry = model.Entries[suffix];
		// Console.WriteLine($"[by-divisor-plan] p={prime} suffix={suffix} k50={entry.K50:F2} k75={entry.K75:F2} gap={entry.GapProbability:F3} cheap<= {plan.CheapLimit:F0} start={plan.Start:F0} target={plan.Target:F0} rho1={plan.Rho1:F2} rho2={plan.Rho2:F2}");

		ulong effectiveAllowed = allowedMax == 0UL ? ulong.MaxValue : allowedMax;
		ulong maxKAllowed = effectiveAllowed == ulong.MaxValue ? ulong.MaxValue : ((effectiveAllowed - 1UL) / step64);
		if (maxKAllowed == 0UL || maxKAllowed < minK)
		{
			processedAll = false;
			return false;
		}

		BigInteger cheapChunk = new BigInteger(Math.Min(plan.CheapLimit, (double)maxKAllowed));
		BigInteger cheapStart = BigInteger.Zero;
		BigInteger cheapEnd = BigInteger.Zero;
		if (TryBuildCheapRange(new BigInteger(minK), cheapChunk, new BigInteger(maxKAllowed), out cheapStart, out cheapEnd))
		{
			if (cheapStart <= ulong.MaxValue && cheapEnd <= ulong.MaxValue)
			{
				ulong cheapStart64 = (ulong)cheapStart;
				ulong cheapEnd64 = (ulong)cheapEnd;
				ulong currentK = cheapStart64;
				bool composite = CheckDivisors64Range(
					prime,
					step64,
					effectiveAllowed,
					cheapStart64,
					cheapEnd64,
					ref currentK,
					out bool processedRange,
					out ulong foundRange);
				if (composite)
				{
					RecordCheapState(currentK);
					foundDivisor = foundRange;
					return true;
				}

				if (!processedRange)
				{
					processedAll = false;
					return false;
				}

				RecordCheapState(cheapEnd);
			}
		}

		double nextK = Math.Max(plan.Start, (cheapEnd > BigInteger.Zero ? (double)(cheapEnd + BigInteger.One) : minK));
		if (nextK < minK)
		{
			nextK = minK;
		}

		double target = plan.Target <= 0d ? nextK : plan.Target;
		if (target < nextK)
		{
			target = nextK + 1d;
		}

		ulong current = (ulong)Math.Ceiling(nextK);
#if DivisorSet_Predictive
		bool loggedProgressive = false;
		while (current <= maxKAllowed)
		{
			if (!loggedProgressive)
			{
				Console.WriteLine($"[by-divisor-progressive] p={prime} startK={current} target={target:F0} rho={(current < target ? plan.Rho1 : plan.Rho2):F2} maxK={maxKAllowed}");
				loggedProgressive = true;
			}

			ulong currentK = current;
			bool compositeSingle = CheckDivisors64Range(
				prime,
				step64,
				effectiveAllowed,
				currentK,
				currentK,
				ref currentK,
				out bool processedSingle,
				out ulong foundSingle);
			if (compositeSingle)
			{
				foundDivisor = foundSingle;
				return true;
			}

			if (!processedSingle)
			{
				processedAll = false;
				return false;
			}

			double multiplier = current < target ? plan.Rho1 : plan.Rho2;
			double advanced = Math.Ceiling(current * multiplier);
			if (advanced <= current)
			{
				advanced = current + 1d;
			}

			if (advanced > maxKAllowed)
			{
				break;
			}

			current = (ulong)advanced;
		}
#elif DivisorSet_Percentile
		ulong k10 = plan.K10 > 0d ? (ulong)Math.Ceiling(plan.K10) : current;
		ulong k25 = plan.K25 > 0d ? (ulong)Math.Ceiling(plan.K25) : k10 + 1UL;
		ulong k50 = plan.K50 > 0d ? (ulong)Math.Ceiling(plan.K50) : k25 + 1UL;
		ulong k75 = plan.K75 > 0d ? (ulong)Math.Ceiling(plan.K75) : k50 + 1UL;

		ReadOnlySpan<ulong> stops = stackalloc ulong[3]
		{
			k25 > 0UL ? k25 - 1UL : 0UL,
			k50 > 0UL ? k50 - 1UL : 0UL,
			k75 > 0UL ? k75 - 1UL : 0UL,
		};

		ulong rangeStart = current > k10 ? current : k10;
		for (int i = 0; i < stops.Length; i++)
		{
			ulong stop = stops[i];
			if (stop < rangeStart)
			{
				continue;
			}

			ulong currentK = rangeStart;
			bool composite = CheckDivisors64Range(
				prime,
				step64,
				effectiveAllowed,
				rangeStart,
				stop,
				ref currentK,
				out bool processedRange,
				out ulong foundRange);
			if (composite)
			{
				foundDivisor = foundRange;
				return true;
			}

			if (!processedRange)
			{
				processedAll = false;
				return false;
			}

			rangeStart = stop + 1UL;
			if (rangeStart > maxKAllowed)
			{
				return false;
			}
		}

		ulong geoStart = rangeStart < k75 ? k75 : rangeStart;
		while (geoStart <= maxKAllowed)
		{
			ulong currentK = geoStart;
			bool composite = CheckDivisors64Range(
				prime,
				step64,
				effectiveAllowed,
				currentK,
				currentK,
				ref currentK,
				out bool processedSingle,
				out ulong foundSingle);
			if (composite)
			{
				foundDivisor = foundSingle;
				return true;
			}

			if (!processedSingle)
			{
				processedAll = false;
				return false;
			}

			double advanced = Math.Ceiling(geoStart * plan.Rho2);
			if (advanced <= geoStart)
			{
				advanced = geoStart + 1d;
			}

			if (advanced > maxKAllowed)
			{
				break;
			}

			geoStart = (ulong)advanced;
		}
#else
		ulong delta = plan.DeltaK > 0d ? (ulong)Math.Ceiling(plan.DeltaK) : 1UL;
		if (delta == 0UL)
		{
			delta = 1UL;
		}

		if (delta > maxKAllowed)
		{
			delta = maxKAllowed;
		}

		while (current <= maxKAllowed)
		{
			ulong currentK = current;
			bool composite = CheckDivisors64Range(
				prime,
				step64,
				effectiveAllowed,
				currentK,
				currentK,
				ref currentK,
				out bool processedSingle,
				out ulong foundSingle);
			if (composite)
			{
				foundDivisor = foundSingle;
				return true;
			}

			if (!processedSingle)
			{
				processedAll = false;
				return false;
			}

			ulong advanced = current + delta;
			if (advanced <= current || advanced == 0UL)
			{
				advanced = current + 1UL;
			}

			if (advanced > maxKAllowed)
			{
				break;
			}

			current = advanced;
		}
#endif

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
				// if (divisorCycle == prime && !IsMersenneValue(prime, candidate))
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

		if (!string.IsNullOrEmpty(targetPath))
		{
			string? directory = Path.GetDirectoryName(targetPath);
			if (!string.IsNullOrEmpty(directory))
			{
				Directory.CreateDirectory(directory);
			}

			File.AppendAllText(targetPath, k.ToString(CultureInfo.InvariantCulture) + Environment.NewLine);
		}

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
					// if (powResult.IsOne && !IsMersenneValue(prime, divisorBig))
					if (powResult.IsOne)
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
						// if (powResult.IsOne && !IsMersenneValue(prime, divisor))
						if (powResult.IsOne)
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

#if DivisorSet_TopDown
	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private bool CheckDivisorsTopDown(
		ulong prime,
		BigInteger allowedMax,
		BigInteger minK,
		out bool processedAll,
		out BigInteger foundDivisor)
	{
		foundDivisor = BigInteger.Zero;
		processedAll = false;

		BigInteger normalizedMinK = minK >= BigInteger.One ? minK : BigInteger.One;
		ulong step = prime << 1;
		ulong allowedMax64 = allowedMax.IsZero ? ulong.MaxValue : (allowedMax > ulong.MaxValue ? ulong.MaxValue : (ulong)allowedMax);
		if (allowedMax64 <= 1UL)
		{
			processedAll = true;
			return false;
		}

		ulong maxKAllowed = (allowedMax64 - 1UL) / step;
		{
			BigInteger maxKBig = maxKAllowed;
			if (TryClampMaxKForMersenneTail(prime, (BigInteger)step, ref maxKBig))
			{
				maxKAllowed = maxKBig > ulong.MaxValue ? ulong.MaxValue : (ulong)maxKBig;
			}
		}

		if (maxKAllowed < 1UL)
		{
			processedAll = true;
			return false;
		}

		if (normalizedMinK > maxKAllowed)
		{
			processedAll = true;
			return false;
		}

		ulong minK64 = normalizedMinK > ulong.MaxValue ? ulong.MaxValue : (ulong)normalizedMinK;
		if (minK64 < 1UL)
		{
			minK64 = 1UL;
		}

		ulong cursor = _topDownCursor > ulong.MaxValue ? ulong.MaxValue : (ulong)_topDownCursor;
		ulong currentK = maxKAllowed;
		if (cursor > 0 && cursor <= maxKAllowed)
		{
			currentK = maxKAllowed - cursor;
		}

		if (currentK < minK64)
		{
			processedAll = true;
			return false;
		}

		UInt128 step128 = step;
		UInt128 divisor = (step128 * currentK) + UInt128.One;
		var residueStepper = new MersenneDivisorResidueStepperDescending(prime, new GpuUInt128(step), new GpuUInt128(divisor));
		while (currentK >= minK64)
		{
			if (divisor > allowedMax64)
			{
				processedAll = true;
				return false;
			}

			if (residueStepper.IsAdmissible())
			{
				ulong divisor64 = (ulong)divisor;
				MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(divisor64);
#if DEVICE_GPU
				bool computed = MersenneNumberDivisorByDivisorTester.TryCalculateCycleLengthForExponentGpu(
					divisor64,
					Accelerator,
					prime,
					divisorData,
					out ulong computedCycle,
					out bool primeOrderFailed);
#elif DEVICE_HYBRID
				bool computed = MersenneNumberDivisorByDivisorTester.TryCalculateCycleLengthForExponentHybrid(
					divisor64,
					Accelerator,
					prime,
					divisorData,
					out ulong computedCycle,
					out bool primeOrderFailed);
#else
				bool computed = MersenneNumberDivisorByDivisorTester.TryCalculateCycleLengthForExponentCpu(
					divisor64,
					prime,
					divisorData,
					out ulong computedCycle,
					out bool primeOrderFailed);
#endif
				ulong divisorCycle;
				if (!computed || computedCycle == 0UL)
				{
#if DEVICE_GPU
					divisorCycle = MersenneNumberDivisorByDivisorTester.CalculateCycleLengthGpu(
						divisor64,
						Accelerator,
						divisorData,
						skipPrimeOrderHeuristic: primeOrderFailed);
#elif DEVICE_HYBRID
					divisorCycle = MersenneNumberDivisorByDivisorTester.CalculateCycleLengthHybrid(
						divisor64,
						Accelerator,
						divisorData,
						skipPrimeOrderHeuristic: primeOrderFailed);
#else
					divisorCycle = MersenneNumberDivisorByDivisorTester.CalculateCycleLengthCpu(
						divisor64,
						divisorData,
						skipPrimeOrderHeuristic: primeOrderFailed);
#endif
				}
				else
				{
					divisorCycle = computedCycle;
				}

				RecordState(cursor);
				if (divisorCycle == prime && !IsMersenneValue(prime, divisor64))
				{
					foundDivisor = divisor64;
					return true;
				}
			}

			cursor++;
			if (cursor > maxKAllowed)
			{
				break;
			}

			if (currentK == minK64)
			{
				break;
			}

			currentK--;
			divisor -= step128;
			residueStepper.Retreat();
		}

		processedAll = true;
		return false;
	}
#endif

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
#if DivisorSet_TopDown
		if (_currentPrime != 0UL)
		{
			if (k > _lastSavedK)
			{
				int next = _stateCounter + 1;
				if (next >= PerfectNumberConstants.ByDivisorStateSaveInterval)
				{
					_kRepository.Upsert(_currentPrime, k);
					_stateCounter = 0;
					_lastSavedK = k;
					_topDownCursor = k;
				}
				else
				{
					_stateCounter = next;
				}
				_topDownCursor = k;
			}
		}
		return;
#endif
#if DivisorSet_Pow2Groups
		if (_currentPow2Phase == Pow2Phase.Special)
		{
			RecordStatePow2(k, _specialRepository, ref _specialStateCounter, ref _specialLastSavedK);
			return;
		}

		if (_currentPow2Phase == Pow2Phase.Groups)
		{
			RecordStatePow2(k, _groupsRepository, ref _groupsStateCounter, ref _groupsLastSavedK);
			return;
		}
#endif
		if (_currentPrime != 0UL)
		{
			if (k > _lastSavedK)
			{
				int next = _stateCounter + 1;
				if (next >= PerfectNumberConstants.ByDivisorStateSaveInterval)
				{
					_kRepository.Upsert(_currentPrime, k);
					_lastSavedK = k;
					_stateCounter = 0;
				}
				else
				{
					_stateCounter = next;
				}
			}
		}
	}

#if DivisorSet_Pow2Groups
	private void RecordState(BigInteger k, Pow2Phase phase)
	{
		switch (phase)
		{
			case Pow2Phase.Special:
				RecordStatePow2(k, _specialRepository, ref _specialStateCounter, ref _specialLastSavedK);
				break;
			case Pow2Phase.Groups:
				RecordStatePow2(k, _groupsRepository, ref _groupsStateCounter, ref _groupsLastSavedK);
				break;
			default:
				RecordState(k);
				break;
		}
	}

	private void RecordStatePow2(BigInteger k, KStateRepository? repo, ref int counter, ref BigInteger lastSavedK)
	{
		if (_currentPrime == 0UL || repo is null)
		{
			return;
		}

		if (k <= lastSavedK)
		{
			return;
		}

		int next = counter + 1;
		if (next >= PerfectNumberConstants.ByDivisorStateSaveInterval)
		{
			repo.Upsert(_currentPrime, k);
			counter = 0;
			lastSavedK = k;
		}
		else
		{
			counter = next;
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

#if DivisorSet_BitContradiction
	private static readonly BigInteger Five = (BigInteger)5;

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private bool CheckDivisorsBitContradiction(
		ulong prime,
		ushort primeDecimalMask,
		BigInteger allowedMax,
		BigInteger minK,
		BigInteger step,
		out bool processedAll,
		out BigInteger foundDivisor)
	{
		processedAll = true;
		foundDivisor = BigInteger.Zero;

		BigInteger currentK = minK;
		bool hasLimit = !allowedMax.IsZero;
		BigInteger maxK = hasLimit ? (allowedMax - BigInteger.One) / step : BigInteger.Zero;
		int stepMod8 = (int)(step % 8);
		int kRem8 = (int)(currentK % 8);

		BigInteger divisor = (step * currentK) + BigInteger.One;
		while (true)
		{
			int remainder8 = (stepMod8 * kRem8 + 1) & 7;
			if (remainder8 != 1 && remainder8 != 7)
			{
				goto MOVE_NEXT;
			}

			if (
					divisor <= ulong.MaxValue &&
					!PrimeTesterByLastDigit.IsPrimeCpu((ulong)divisor)
				)
			{
				goto MOVE_NEXT;
			}

			ulong effectiveExponent = 0UL;
// 			if (divisor <= ulong.MaxValue)
// 			{
// 				ulong divisor64 = (ulong)divisor;
// 				MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(divisor64);
// #if DEVICE_GPU
// 				bool computed = MersenneNumberDivisorByDivisorTester.TryCalculateCycleLengthForExponentGpu(
// 					divisor64,
// 					Accelerator,
// 					prime,
// 					divisorData,
// 					out ulong computedCycle,
// 					out bool primeOrderFailed);
// #elif DEVICE_HYBRID
// 				bool computed = MersenneNumberDivisorByDivisorTester.TryCalculateCycleLengthForExponentHybrid(
// 					divisor64,
// 					Accelerator,
// 					prime,
// 					divisorData,
// 					out ulong computedCycle,
// 					out bool primeOrderFailed);
// #else
// 				bool computed = MersenneNumberDivisorByDivisorTester.TryCalculateCycleLengthForExponentCpu(
// 					divisor64,
// 					prime,
// 					divisorData,
// 					out ulong computedCycle,
// 					out bool primeOrderFailed);
// #endif
// 				ulong divisorCycle;
// 				if (!computed || computedCycle == 0UL)
// 				{
// #if DEVICE_GPU
// 					divisorCycle = MersenneNumberDivisorByDivisorTester.CalculateCycleLengthGpu(
// 						divisor64,
// 						Accelerator,
// 						divisorData,
// 						skipPrimeOrderHeuristic: primeOrderFailed);
// #elif DEVICE_HYBRID
// 					divisorCycle = MersenneNumberDivisorByDivisorTester.CalculateCycleLengthHybrid(
// 						divisor64,
// 						Accelerator,
// 						divisorData,
// 						skipPrimeOrderHeuristic: primeOrderFailed);
// #else
// 					divisorCycle = MersenneNumberDivisorByDivisorTester.CalculateCycleLengthCpu(
// 						divisor64,
// 						divisorData,
// 						skipPrimeOrderHeuristic: primeOrderFailed);
// #endif
// 				}
// 				else
// 				{
// 					divisorCycle = computedCycle;
// 				}

// 				if (divisorCycle > 0UL && (prime % divisorCycle) != 0UL)
// 				{
// 					if (hasLimit && currentK >= maxK)
// 					{
// 						break;
// 					}

// 					currentK += BigInteger.One;
// 					continue;
// 				}

// 				if (divisorCycle > 0UL)
// 				{
// 					effectiveExponent = divisorCycle;
// 				}
// 			}

			bool decided = TryExactBitContradictionCheck(prime, divisor, effectiveExponent, out bool divides);
			if (!decided)
			{
				throw new InvalidOperationException($"BitContradiction check did not decide for p={prime}, divisor={divisor} (k={currentK}).");
			}

			if (divides)
			{
				foundDivisor = divisor;
				return true;
			}

			RecordState(currentK);
		MOVE_NEXT:
			if (hasLimit && currentK >= maxK)
			{
				break;
			}

			currentK += BigInteger.One;
			divisor += step;
			kRem8 = (kRem8 + 1) & 7;
		}

		return false;

		bool fits64 = minK <= ulong.MaxValue;
		bool limitFits64 = allowedMax.IsZero || allowedMax <= ulong.MaxValue;

		if (fits64 && limitFits64)
		{
			ulong allowedMax64 = allowedMax.IsZero ? ulong.MaxValue : (ulong)allowedMax;
			return CheckDivisors64Bit(
				prime,
				primeDecimalMask,
				allowedMax64,
				(ulong)minK,
				out processedAll,
				out ulong found64);
		}

		if (fits64 && allowedMax > ulong.MaxValue)
		{
			ulong allowedMax64 = ulong.MaxValue;
			bool composite64 = CheckDivisors64Bit(
				prime,
				primeDecimalMask,
				allowedMax64,
				(ulong)minK,
				out processedAll,
				out ulong found64);
			if (composite64)
			{
				foundDivisor = found64;
				return true;
			}

			BigInteger iterations = ((BigInteger)(allowedMax64 - (((ulong)prime << 1) * (ulong)minK) - 1UL) / (ulong)step) + BigInteger.One;
			BigInteger nextK = minK + iterations;
			return CheckDivisorsLarge(
				prime,
				primeDecimalMask,
				allowedMax,
				nextK,
				step,
				out processedAll,
				out foundDivisor);
		}

		return CheckDivisorsLarge(
			prime,
			primeDecimalMask,
			allowedMax,
			minK,
			step,
			out processedAll,
			out foundDivisor);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool TryExactBitContradictionCheck(ulong prime, BigInteger divisor, ulong effectiveExponent, out bool divides)
	{
		Stopwatch sw = Stopwatch.StartNew();
		divides = false;
		const ulong DebugTopDownDivisor = 1209708008767UL;

		if (prime == 0 || divisor <= BigInteger.One)
		{
			Console.WriteLine($"[bitcontradiction] invalid inputs {prime} / {divisor}");
			return false;
		}

		int qBitLength = GetBitLengthPortable(divisor);
		if (qBitLength > (int)prime)
		{
			divides = false;
			return true;
		}

		// Fast structural contradictions only: q must be odd and the implied maximum length of a must be positive.
		if ((divisor & BigInteger.One) == BigInteger.Zero)
		{
			divides = false;
			return true;
		}

		if (prime <= (ulong)qBitLength)
		{
			divides = false;
			return true;
		}

		// Iterative column-wise propagation without storing all columns; we only keep assigned bits of 'a'.
		Span<int> oneOffsets = stackalloc int[qBitLength];
		int oneCount = 0;
		for (int i = 0; i < qBitLength; i++)
		{
			if (((divisor >> i) & BigInteger.One) == BigInteger.One)
			{
				oneOffsets[oneCount++] = i;
			}
		}

		if (oneCount == 0)
		{
			divides = false;
			return true;
		}

		ReadOnlySpan<int> oneOffsetsSlice = oneOffsets[..oneCount];
		ulong exponent = effectiveExponent > 0UL ? effectiveExponent : prime;
		bool decided = BitContradictionSolver.TryCheckDivisibilityFromOneOffsets(oneOffsetsSlice, exponent, out divides, out var reason);
		if (decided && divides)
		{
			// Verify to avoid false positives on structurally admissible but non-dividing q.
			bool structuralOk = ((divisor - BigInteger.One) % (((BigInteger)prime) << 1)) == BigInteger.Zero;
			bool dividesExact = structuralOk && BigInteger.ModPow(2, prime, divisor) == BigInteger.One;
			if (!dividesExact)
			{
				divides = false;
				reason = BitContradictionSolver.ContradictionReason.ParityUnreachable;
			}
		}

		sw.Stop();
		if (decided && !divides)
		{
			if (reason == BitContradictionSolver.ContradictionReason.ParityUnreachable &&
				divisor <= ulong.MaxValue &&
				(ulong)divisor == DebugTopDownDivisor)
			{
				var failure = BitContradictionSolver.LastTopDownFailure;
				if (failure.HasValue)
				{
					var info = failure.Value;
					Console.WriteLine($"[bit-contradiction] top-down prune failed at column={info.Column} carry=[{info.CarryMin},{info.CarryMax}] unknown={info.Unknown}");
				}
			}

			string message = $"[bit-contradiction] p={prime} ruled out divisor={divisor} ({reason})";
			// Console.WriteLine(message);
			_bitContradictionStateRepository?.Upsert(prime, message);
		}

		Console.WriteLine($"[bitcontradiction ({sw.Elapsed})] Prime={prime}, Divisor={divisor} Decided={decided} Divides={divides} Reason={reason}");
		return decided;
	}
#endif

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private void MarkPow2MinusOneChecked()
	{
		_pow2MinusOneChecked = true;
		if (_currentPrime != 0UL)
		{
			_pow2MinusOneRepository?.MarkChecked(_currentPrime);
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
		_currentPrime = prime;
		MarkPow2MinusOneChecked();
		return false;
	}
}
