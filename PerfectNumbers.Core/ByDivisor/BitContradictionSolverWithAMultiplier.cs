using System.Runtime.CompilerServices;
using System.Numerics;

namespace PerfectNumbers.Core.ByDivisor;

internal sealed class PrivateWorkSet
{
	// Per-thread scratch buffers to avoid per-run allocations in the hot path.
	// Buffers grow on demand and are never shrunk.
	public ulong[] KnownOne0 = [];
	public ulong[] KnownOne1 = [];
	public ulong[] KnownZero0 = [];
	public ulong[] KnownZero1 = [];
	public ulong[] QMaskWords = [];
	public ulong[] AOneWin = [];
	public ulong[] AZeroWin = [];
	public int[] PrevOffsets = [];
	public int PrevOffsetsLength;
	public bool HasPrevOffsets;
	public BigInteger[] InvByKMod = [];
	public bool[] HasInvByKMod = [];
	public bool HasLastInvMod2k;
	public BigInteger LastInvMod2k = BigInteger.Zero;
	public BigInteger LastPow2K = BigInteger.Zero;
	public BigInteger LastPow2QMod2k = BigInteger.Zero;
	public bool HasLastPow2QMod2k;
	public byte[] QMod2kBytes = [];

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void EnsureCapacity(int wordCount, int qWordCount)
	{
		if (KnownOne0.Length < wordCount)
		{
			EnsureUlong(ref KnownOne0, wordCount);
			EnsureUlong(ref KnownOne1, wordCount);
			EnsureUlong(ref KnownZero0, wordCount);
			EnsureUlong(ref KnownZero1, wordCount);
		}

		if (QMaskWords.Length < qWordCount)
		{
			EnsureUlong(ref QMaskWords, qWordCount);
			EnsureUlong(ref AOneWin, qWordCount);
			EnsureUlong(ref AZeroWin, qWordCount);
		}

		if (InvByKMod.Length != 1024)
		{
			InvByKMod = new BigInteger[1024];
			HasInvByKMod = new bool[1024];
		}
		if (QMod2kBytes.Length != 128)
		{
			QMod2kBytes = new byte[128];
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void EnsureUlong(ref ulong[] arr, int needed) => Array.Resize(ref arr, needed);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void EnsureInt(ref int[] arr, int needed) => Array.Resize(ref arr, needed);
}


/// <summary>
/// Helper utilities for the BitContradiction divisor search. These methods implement the
/// interval-based carry propagation described in the BitContradictions plan document.
/// The solver itself will build on top of these primitives.
/// </summary>
internal static class BitContradictionSolverWithAMultiplier
{
	private const int PingPongBlockColumns = 4096;
	private const int TailCarryBatchColumns = 16384;
	private const int TailCarryBatchRepeats = 1;
	private const int TopDownBorrowColumns = 4096;
	private const int TopDownBorrowMinUnknown = 12;
	private const int PrefixRunCacheOffsets = 256;
	private const int PrefixRunCacheCapacity = 256;
	private const int ModLocalPrefilterMaxABits = 20;
	private const int TopDownCacheCapacity = 16;
	private const int HighBitCarryPrefilterColumns = 16;
	private const int MiniPingPongPrefilterColumns = 60;

	private static readonly int[] ModLocalPrefilterModuli = { 65537, 65521 };
	private static readonly ulong[] BitMask64 = CreateBitMask64();

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ulong[] CreateBitMask64()
	{
		ulong[] masks = new ulong[64];
		ulong value = 1UL;
		for (int i = 0; i < 64; i++)
		{
			masks[i] = value;
			value <<= 1;
		}
		return masks;
	}


#if DETAILED_LOG
	[ThreadStatic] private static long _statStableAdvanceCalls;
	[ThreadStatic] private static long _statStableAdvanceCols;
	[ThreadStatic] private static long _statStableAdvanceColsBatched;
	[ThreadStatic] private static long _statScalarCols;

	[ThreadStatic] private static long _statDeltaRequests;
	[ThreadStatic] private static long _statDeltaHits;
	[ThreadStatic] private static long _statDeltaBuilds;

	[ThreadStatic] private static long _statStableAdvanceBigShift; // how many times columnCount>=64 in TryAdvanceStableUnknown
	[ThreadStatic] private static long _statBatchEntry;
	[ThreadStatic] private static long _statBatchCols;
	[ThreadStatic] private static long _statCleanChecks;
	[ThreadStatic] private static long _statCleanFalse;
	[ThreadStatic] private static int _statFirstDirtyColumn;
	[ThreadStatic] private static int _statFirstDirtyWord;
	[ThreadStatic] private static ulong _statFirstDirtyQ;
	[ThreadStatic] private static ulong _statFirstDirtyKnown;

	public readonly struct TopDownPruneFailure(long column, long carryMin, long carryMax, long unknown)
	{
		public readonly long Column = column;
		public readonly long CarryMin = carryMin;
		public readonly long CarryMax = carryMax;
		public readonly long Unknown = unknown;
	}

	private readonly struct TopDownCacheValue(bool success, TopDownPruneFailure? failure)
	{
		public readonly bool Success = success;
		public readonly TopDownPruneFailure? Failure = failure;
	}

	[ThreadStatic]
	private static Dictionary<TopDownCacheKey, TopDownCacheValue>? _topDownCache;

	[ThreadStatic]
	private static TopDownPruneFailure? _lastTopDownFailure;
	internal static TopDownPruneFailure? LastTopDownFailure => _lastTopDownFailure;


	private static void PrintStats()
	{
		Console.WriteLine(
			$"[STATS] stableCalls={_statStableAdvanceCalls} stableCols={_statStableAdvanceColsBatched} " +
			$"scalarCols={_statScalarCols} bigShift={_statStableAdvanceBigShift} " +
			$"deltaReq={_statDeltaRequests} deltaHits={_statDeltaHits} deltaBuilds={_statDeltaBuilds}");
		Console.WriteLine(
			$"[CLEAN] checks={_statCleanChecks} false={_statCleanFalse} " +
			$"firstDirtyCol={_statFirstDirtyColumn} word={_statFirstDirtyWord} " +
			$"qWord=0x{_statFirstDirtyQ:X16} knownWord=0x{_statFirstDirtyKnown:X16}");
	}
#else
	[ThreadStatic]
	private static Dictionary<TopDownCacheKey, bool>? _topDownCache;
#endif
	[ThreadStatic]
	private static long _dynamicModChecks;
	[ThreadStatic]
	private static long _dynamicModFailures;
	[ThreadStatic]
	private static Dictionary<QModCacheKey, int>? _qModCache;
	[ThreadStatic]
	private static List<TopDownCacheKey>? _topDownCacheOrder;
	[ThreadStatic]
	private static Dictionary<long, PrefixRunCacheEntry>? _prefixRunCache;
	[ThreadStatic]
	private static List<long>? _prefixRunCacheOrder;
	[ThreadStatic]
	private static Dictionary<int, int[]>? _pow2DiffModCache;

	[ThreadStatic]
	private static PrivateWorkSet _privateWorkSet;

	private readonly struct QModCacheKey(long hash, int mod)
	{
		public readonly long Hash = hash;
		public readonly int Mod = mod;

		public bool Equals(QModCacheKey other) => Hash == other.Hash && Mod == other.Mod;
		public override bool Equals(object? obj) => obj is QModCacheKey other && Equals(other);
		public override int GetHashCode() => HashCode.Combine(Hash, Mod);
	}

	private readonly struct TopDownCacheKey(long hash, int length, long maxAllowedA, long pLong)
	{
		public readonly long Hash = hash;
		public readonly int Length = length;
		public readonly long MaxAllowedA = maxAllowedA;
		public readonly long PLong = pLong;

		public bool Equals(TopDownCacheKey other) => Hash == other.Hash && Length == other.Length && MaxAllowedA == other.MaxAllowedA && PLong == other.PLong;
		public override bool Equals(object? obj) => obj is TopDownCacheKey other && Equals(other);
		public override int GetHashCode() => HashCode.Combine(Hash, Length, MaxAllowedA, PLong);
	}

	private readonly struct PrefixRunCacheEntry(
		int prefixCount,
		int runCount,
		int lastOffset,
		int lastRunStart,
		int lastRunLength,
		int[] runStarts,
		int[] runLengths)
	{
		public readonly int PrefixCount = prefixCount;
		public readonly int RunCount = runCount;
		public readonly int LastOffset = lastOffset;
		public readonly int LastRunStart = lastRunStart;
		public readonly int LastRunLength = lastRunLength;
		public readonly int[] RunStarts = runStarts;
		public readonly int[] RunLengths = runLengths;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void ComputeForcedRemainingAndClean(
		in ulong[] qMask,
		in ulong[] oneWin,
		in ulong[] zeroWin,
		int wordCount,
		out int forced,
		out int remaining,
		out bool clean)
	{
		forced = 0;
		remaining = 0;
		clean = true;

		for (int i = 0; i < wordCount; i++)
		{
			ulong q = qMask[i];
			ulong one = oneWin[i];
			ulong known = one | zeroWin[i];

			one &= q;
			ulong unknownBits = q & ~known;

			forced += BitOperations.PopCount(one);
			remaining += BitOperations.PopCount(unknownBits);

			if ((q & known) != 0)
				clean = false;
		}
	}

	private static long ComputeQOffsetsHash(in ReadOnlySpan<int> qOneOffsets)
	{
		const long fnvOffset = unchecked(1469598103934665603);
		const long fnvPrime = 1099511628211;
		long hash = fnvOffset;
		int qOneOffsetsLength = qOneOffsets.Length;
		for (int i = 0; i < qOneOffsetsLength; i++)
		{
			hash ^= qOneOffsets[i];
			hash *= fnvPrime;
		}

		return hash;
	}

	private static int GetQModCached(ReadOnlySpan<int> qOneOffsets, long qHash, int mod)
	{
		Dictionary<QModCacheKey, int>? cache = _qModCache;
		if (cache is null)
		{
			cache = [];
			_qModCache = cache;

		}
		var key = new QModCacheKey(qHash, mod);
		if (cache.TryGetValue(key, out int cached))
		{
			return cached;
		}

		int value = ComputeQMod(qOneOffsets, mod);
		cache[key] = value;
		return value;
	}

	private static long ComputeQOffsetsPrefixHash(in ReadOnlySpan<int> qOneOffsets, int count)
	{
		const long fnvOffset = unchecked(1469598103934665603);
		const long fnvPrime = 1099511628211;
		long hash = fnvOffset;
		for (int i = 0; i < count; i++)
		{
			hash ^= qOneOffsets[i];
			hash *= fnvPrime;
		}

		hash ^= count;
		hash *= fnvPrime;
		return hash;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool IsCleanInQ(in ulong[] qMask, in ulong[] oneWin, in ulong[] zeroWin, int wordCount)
	{
		for (int i = 0; i < wordCount; i++)
		{
			if ((qMask[i] & (oneWin[i] | zeroWin[i])) != 0)
				return false;
		}

		return true;
	}

	private static bool TryGetPrefixRunCache(long key, out PrefixRunCacheEntry entry)
	{
		var cache = _prefixRunCache;
		if (cache is null)
		{
			cache = [];
			_prefixRunCache = cache;
		}

		return cache.TryGetValue(key, out entry);
	}

	private static void StorePrefixRunCache(long key, in PrefixRunCacheEntry entry)
	{
		_prefixRunCacheOrder ??= new List<long>(PrefixRunCacheCapacity + 1);

		if (_prefixRunCache!.ContainsKey(key))
		{
			_prefixRunCache[key] = entry;
			return;
		}

		if (_prefixRunCacheOrder.Count >= PrefixRunCacheCapacity)
		{
			long evict = _prefixRunCacheOrder[0];
			_prefixRunCacheOrder.RemoveAt(0);
			_prefixRunCache.Remove(evict);
		}

		_prefixRunCache[key] = entry;
		_prefixRunCacheOrder.Add(key);
	}

	private static PrefixRunCacheEntry BuildPrefixRunCache(in ReadOnlySpan<int> qOneOffsets, int prefixCount)
	{
		// TODO: Remove per-call allocations
		int[] starts = new int[prefixCount];
		int[] lengths = new int[prefixCount];
		int runCount = 0;
		int prev = qOneOffsets[0];
		int currentStart = prev;
		int currentLength = 1;

		for (int i = 1; i < prefixCount; i++)
		{
			int offset = qOneOffsets[i];
			if (offset == prev + 1)
			{
				currentLength++;
			}
			else
			{
				starts[runCount] = currentStart;
				lengths[runCount] = currentLength;
				runCount++;
				currentStart = offset;
				currentLength = 1;
			}

			prev = offset;
		}

		starts[runCount] = currentStart;
		lengths[runCount] = currentLength;
		runCount++;

		int[] cachedStarts = new int[runCount];
		int[] cachedLengths = new int[runCount];
		Array.Copy(starts, cachedStarts, runCount);
		Array.Copy(lengths, cachedLengths, runCount);

		return new PrefixRunCacheEntry(
			prefixCount,
			runCount,
			prev,
			currentStart,
			currentLength,
			cachedStarts,
			cachedLengths);
	}

#if DETAILED_LOG
	private static bool TryGetTopDownCache(in TopDownCacheKey key, out TopDownCacheValue value)
#else
	private static bool TryGetTopDownCache(in TopDownCacheKey key, out bool value)
#endif
	{
		// TODO: Remove this check after Initialize
		if (_topDownCache == null)
		{
			value = default;
			return false;
		}

		return _topDownCache.TryGetValue(key, out value);
	}

#if DETAILED_LOG
	private static void StoreTopDownCache(in TopDownCacheKey key, in TopDownCacheValue value)
#else
	private static void StoreTopDownCache(in TopDownCacheKey key, in bool value)
#endif
	{
		// TODO: Remove this check after Initialize
		_topDownCache ??= [];
		_topDownCacheOrder ??= new List<TopDownCacheKey>(TopDownCacheCapacity + 1);

		if (_topDownCache.ContainsKey(key))
		{
			_topDownCache[key] = value;
			return;
		}

		if (_topDownCacheOrder.Count >= TopDownCacheCapacity)
		{
			var evict = _topDownCacheOrder[0];
			_topDownCacheOrder.RemoveAt(0);
			_topDownCache.Remove(evict);
		}

		_topDownCache[key] = value;
		_topDownCacheOrder.Add(key);
	}

	internal static ColumnBounds ComputeColumnBounds(
		ReadOnlySpan<bool> multiplicand,
		ReadOnlySpan<bool> multiplier,
		int columnIndex)
	{
		int forced = 0;
		int possible = 0;

		int multiplicandLength = multiplicand.Length;
		int multiplierLength = multiplier.Length;
		for (int i = 0; i <= columnIndex; i++)
		{
			bool a = i < multiplicandLength && multiplicand[i];
			bool q = columnIndex - i < multiplierLength && multiplier[columnIndex - i];
			if (!a && !q)
			{
				continue;
			}

			if (a && q)
			{
				forced++;
			}

			possible++;
		}

		return new ColumnBounds(forced, possible);
	}

	internal static bool TryPropagateCarry(
		ref CarryRange currentCarry,
		int forcedOnes,
		int possibleOnes,
		int requiredResultBit)
	{
		long minSum = currentCarry.Min + forcedOnes;
		long maxSum = currentCarry.Max + possibleOnes;
		int parity = requiredResultBit & 1;

		minSum = AlignUpToParity(minSum, parity);
		maxSum = AlignDownToParity(maxSum, parity);

		if (minSum > maxSum)
		{
			return false;
		}

		currentCarry = new CarryRange((minSum - requiredResultBit) >> 1, (maxSum - requiredResultBit) >> 1);
		return true;
	}

	/// <summary>
	/// Mini ping-pong (Option B): compute a conservative allowed carry range after walking
	/// top-down for a given number of columns starting at <paramref name="startColumn"/>.
	///
	/// Semantics:
	/// - We treat "carryOut at column = startColumn+1" as the starting set (seed).
	/// - Each step processes one column (carryOut -> carryIn) moving downward.
	/// - After <paramref name="steps"/> columns, the resulting set corresponds to carry-in
	///   at column (startColumn - steps + 1).
	/// </summary>
	private static bool TryHighBitCarryRangeToColumnJ(
		ReadOnlySpan<int> qOneOffsets,
		int qOneOffsetsLength,
		int pLong,
		int maxAllowedA,
		int startColumn,          // e.g. p-1, then J1-1, ...
		int steps,                // number of columns to process downward
		long startCarryOutMin,    // seed carryOut range
		long startCarryOutMax,
		out long allowedMin,
		out long allowedMax)
	{
		allowedMin = 0;
		allowedMax = long.MaxValue;

		if (steps <= 0)
		{
			// No processing: carry-in == seed (conservative)
			allowedMin = startCarryOutMin < 0 ? 0 : startCarryOutMin;
			allowedMax = startCarryOutMax < 0 ? 0 : startCarryOutMax;
			if (allowedMax < allowedMin) (allowedMin, allowedMax) = (0, long.MaxValue);
			return true;
		}

		// Clamp startColumn to legal high column range. If outside, be conservative.
		int maxHigh = pLong - 1;
		if (startColumn > maxHigh || startColumn < 0)
		{
			allowedMin = 0;
			allowedMax = long.MaxValue;
			return true;
		}

		// Determine last processed column.
		int endColumn = startColumn - (steps - 1);
		if (endColumn < 0)
		{
			// Processing would go past column 0. We can clamp steps to reach column 0 exactly.
			steps = startColumn + 1;
			endColumn = 0;
			if (steps <= 0)
			{
				allowedMin = 0;
				allowedMax = 0;
				return true;
			}
		}

		// Two fixed interval buffers on stack: each interval is [lo, hi], disjoint and sorted.
		Span<long> lo0 = stackalloc long[64];
		Span<long> hi0 = stackalloc long[64];
		Span<long> lo1 = stackalloc long[64];
		Span<long> hi1 = stackalloc long[64];

		Span<long> outLo = lo0, outHi = hi0;
		Span<long> inLo = lo1, inHi = hi1;

		int outCount = 1;
		outLo[0] = startCarryOutMin;
		outHi[0] = startCarryOutMax;

		int inCount = 0;
		bool flip = false;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static bool AddInterval(ref int count, Span<long> loArr, Span<long> hiArr, long lo, long hi)
		{
			if (hi < lo) return true;
			if (lo < 0) lo = 0;
			if (hi < 0) return true;

			if (count == 0)
			{
				loArr[0] = lo;
				hiArr[0] = hi;
				count = 1;
				return true;
			}

			long lastHi = hiArr[count - 1];
			if (lo <= lastHi + 1)
			{
				if (hi > lastHi) hiArr[count - 1] = hi;
				return true;
			}

			if ((uint)count >= (uint)loArr.Length)
			{
				// Too many intervals: stop filtering conservatively.
				return false;
			}

			loArr[count] = lo;
			hiArr[count] = hi;
			count++;
			return true;
		}

		for (int step = 0; step < steps; step++)
		{
			int column = startColumn - step;

			// n = count of q-offsets t such that 0 <= column - t <= maxAllowedA
			// <=> column - maxAllowedA <= t <= column
			int lowerT;
			long lowerTLong = (long)column - (long)maxAllowedA;
			if (lowerTLong <= 0) lowerT = 0;
			else if (lowerTLong >= int.MaxValue) lowerT = int.MaxValue;
			else lowerT = (int)lowerTLong;

			int upperT = column <= 0 ? 0 : column;

			FindBounds(qOneOffsets, lowerT, upperT, out int left, out int right);
			int n = right - left;
			if (n < 0) return false;

			inCount = 0;

			for (int i = 0; i < outCount; i++)
			{
				long a = outLo[i];
				long b = outHi[i];

				if (b < 0) continue;
				if (a < 0) a = 0;
				if (b < a) continue;

				const long TwicePlusOneOverflowLimit = (long.MaxValue - 1) >> 1;

				// If carryOut is too large for safe 2*x+1 in long, we must remain conservative.
				if (a > TwicePlusOneOverflowLimit || b > TwicePlusOneOverflowLimit)
				{
					allowedMin = 0;
					allowedMax = long.MaxValue;
					return true;
				}

				long rMin = (a << 1) + 1;
				long rMax = (b << 1) + 1;

				// carryIn ∈ [ max(0, r-n), r ]
				if (n <= 0)
				{
					if (!AddInterval(ref inCount, inLo, inHi, rMin, rMax))
					{
						allowedMin = 0;
						allowedMax = long.MaxValue;
						return true;
					}
					continue;
				}

				long lo = rMin - n;
				if (lo < 0) lo = 0;

				if (!AddInterval(ref inCount, inLo, inHi, lo, rMax))
				{
					allowedMin = 0;
					allowedMax = long.MaxValue;
					return true;
				}
			}

			if (inCount <= 0)
			{
				return false;
			}

			// Swap buffers for next step.
			outCount = inCount;
			flip = !flip;
			if (!flip)
			{
				outLo = lo0; outHi = hi0;
				inLo = lo1; inHi = hi1;
			}
			else
			{
				outLo = lo1; outHi = hi1;
				inLo = lo0; inHi = hi0;
			}
		}

		// After steps, 'out' corresponds to carry-in at the last processed column (endColumn).
		allowedMin = outLo[0];
		allowedMax = outHi[outCount - 1];
		return true;
	}

	private static bool TryModularPrefilter(
		ReadOnlySpan<int> qOneOffsets,
		long qHash,
		ulong p,
		long maxAllowedA)
	{
		if (maxAllowedA < 0 || maxAllowedA > ModLocalPrefilterMaxABits)
		{
			return true;
		}

		int bitCount = (int)(maxAllowedA + 1);
		if (bitCount <= 0 || bitCount >= 62)
		{
			return true;
		}

		long maxAValue = (1L << bitCount) - 1L;
		int modLocalPrefilterModuliLength = ModLocalPrefilterModuli.Length;
		for (int i = 0; i < modLocalPrefilterModuliLength; i++)
		{
			int mod = ModLocalPrefilterModuli[i];
			int qMod = GetQModCached(qOneOffsets, qHash, mod);
			int target = ModPow2Minus1(p, mod);
			if (qMod == 0)
			{
				if (target != 0)
				{
					return false;
				}

				continue;
			}

			int inverse = ModPow(qMod, mod - 2, mod);
			long x = (long)target * inverse % mod;
			if (x > maxAValue)
			{
				return false;
			}
		}

		return true;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void FindBounds(ReadOnlySpan<int> values, int lowerTarget, int upperTarget, out int lower, out int upper)
	{
		// Find lower bound for lowerTarget
		int lo = 0;
		int valuesLength = values.Length;
		int hi = valuesLength;
		int mid;
		while (lo < hi)
		{
			mid = lo + ((hi - lo) >> 1);
			if (values[mid] < lowerTarget)
			{
				lo = mid + 1;
			}
			else
			{
				hi = mid;
			}
		}

		lower = lo;

		// Find upper bound for upperTarget, starting from the lower bound's result.
		// Since upperTarget >= lowerTarget is assumed, we can start the search for the upper bound
		// from the position of the lower bound, effectively reducing the search space.
		hi = valuesLength;
		while (lo < hi)
		{
			mid = lo + ((hi - lo) >> 1);
			if (values[mid] <= upperTarget)
			{
				lo = mid + 1;
			}
			else
			{
				hi = mid;
			}
		}

		upper = lo;
	}

	/// <summary>
	/// Extremely small, allocation-free prefilter over the last few columns (near p-1).
	///
	/// It uses only the fact that the product must have no carry beyond column p-1 (i.e. carry-out from
	/// column p-1 is zero) and that bits 0..p-1 of M_p are all ones.
	///
	/// This is intentionally conservative (never rejects a valid divisor). In practice it can still
	/// remove some q with "awkward" high-bit geometry where the required carry values near the top
	/// become impossible given the availability of contributions implied by qOneOffsets and maxAllowedA.
	/// </summary>
	private static bool TryHighBitTopCarryPrefilter(
		ReadOnlySpan<int> qOneOffsets,
		int qOneOffsetsLength,
		long pLong,
		long maxAllowedA)
	{
		// We process last K columns: (p-1), (p-2), ...
		int columns = HighBitCarryPrefilterColumns;
		long maxPossibleColumns = pLong - 1;
		if (columns > maxPossibleColumns)
			columns = (int)maxPossibleColumns;

		// Two fixed interval buffers on stack.
		// Each interval is [lo, hi], disjoint and sorted.
		// 32 is intentionally small: in practice this stays tiny (often 1-2 intervals).
		Span<long> lo0 = stackalloc long[64];
		Span<long> hi0 = stackalloc long[64];
		Span<long> lo1 = stackalloc long[64];
		Span<long> hi1 = stackalloc long[64];

		// "out" = carryOut set for current column (start at top with {0})
		Span<long> outLo = lo0, outHi = hi0;
		Span<long> inLo = lo1, inHi = hi1;

		int outCount = 1;
		outLo[0] = 0;
		outHi[0] = 0;

		int inCount = 0;
		bool flip = false;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static void AddInterval(ref int count, Span<long> loArr, Span<long> hiArr, long lo, long hi)
		{
			if (hi < lo) return;

			if (lo < 0) lo = 0;
			if (hi < 0) return;

			if (count == 0)
			{
				loArr[0] = lo;
				hiArr[0] = hi;
				count = 1;
				return;
			}

			// Merge into last interval if overlapping/adjacent.
			long lastHi = hiArr[count - 1];
			if (lo <= lastHi + 1)
			{
				if (hi > lastHi)
					hiArr[count - 1] = hi;
				return;
			}

			// Append
			if ((uint)count >= (uint)loArr.Length)
			{
				// If this ever happens, we conservatively stop filtering (do not reject).
				// Returning "true" keeps correctness (prefilter is optional).
				count = 0;
				return;
			}

			loArr[count] = lo;
			hiArr[count] = hi;
			count++;
		}

		for (int step = 0; step < columns; step++)
		{
			long column = maxPossibleColumns - step;
			if (column < 0)
				break;

			// n = count of q-offsets t such that 0 <= column - t <= maxAllowedA
			// <=> column - maxAllowedA <= t <= column
			long lowerTLong = column - maxAllowedA;

			int lowerT =
				lowerTLong <= 0 ? 0 :
				(lowerTLong >= int.MaxValue ? int.MaxValue : (int)lowerTLong);

			int upperT =
				column <= 0 ? 0 :
				(column >= int.MaxValue ? int.MaxValue : (int)column);

			FindBounds(qOneOffsets, lowerT, upperT, out int left, out int right);
			int n = right - left;

			if (n < 0)
			{
				// Console.WriteLine("n < 0 - excluding in pre-filter");
				return false;
			}

			// Build carryIn set into inLo/inHi
			inCount = 0;

			// For M_p, bit in columns [0..p-1] is 1:
			// carryIn + ones = 1 + 2*carryOut
			// carryIn = (2*carryOut + 1) - ones, with ones in [0..min(n, 2*carryOut+1)].
			//
			// That yields carryIn interval:
			//   r = 2*carryOut + 1
			//   carryIn ∈ [ r - min(n, r), r ] = [ max(0, r-n), r ].
			//
			// Union over carryOut intervals is merged cheaply.

			for (int i = 0; i < outCount; i++)
			{
				long a = outLo[i];
				long b = outHi[i];

				if (b < 0) continue;
				if (a < 0) a = 0;
				if (b < a) continue;

				const long TwicePlusOneOverflowLimit = (long.MaxValue - 1) >> 1;
				// When carryOut is already too big, top-down becomes useless in long.
				// We want to be conservative to avoid rejecting correct divisors => stop filtering with this filter.
				if (a > TwicePlusOneOverflowLimit || b > TwicePlusOneOverflowLimit)
				{
					return true;
				}

				// r ranges from rMin..rMax over carryOut in [a..b]
				long rMin = (a << 1) + 1;
				long rMax = (b << 1) + 1;

				if (n <= 0)
				{
					// ones=0 only -> carryIn=r (odd sequence). Conservatively interval [rMin..rMax].
					AddInterval(ref inCount, inLo, inHi, rMin, rMax);
					if (inCount == 0) return true; // overflow of intervals -> stop filtering
					continue;
				}

				if (n == 1)
				{
					// ones in {0,1} -> carryIn in {r, r-1}. Conservatively [rMin-1 .. rMax].
					AddInterval(ref inCount, inLo, inHi, rMin - 1, rMax);
					if (inCount == 0) return true;
					continue;
				}

				// n >= 2 : union becomes dense enough that interval bounds are safe conservative approximations.
				if (rMax <= n)
				{
					// For all r<=n, lower bound is 0.
					AddInterval(ref inCount, inLo, inHi, 0, rMax);
					if (inCount == 0) return true;
				}
				else if (rMin > n)
				{
					// Always r>n => [r-n .. r]
					AddInterval(ref inCount, inLo, inHi, rMin - n, rMax);
					if (inCount == 0) return true;
				}
				else
				{
					// Crossing: part uses [0..r], later [r-n..r]. For n>=2 union is [0..rMax].
					AddInterval(ref inCount, inLo, inHi, 0, rMax);
					if (inCount == 0) return true;
				}
			}

			if (inCount <= 0)
			{
				// Console.WriteLine("inCount <= 0 - excluding in pre-filter");
				return false;
			}

			// Prepare for next column: carryOut := carryIn (shift down one column)
			outCount = inCount;

			// Toggle which buffers are used for out vs in.
			// Avoid tuple swap / temp Span to satisfy Span lifetime rules.
			flip = !flip;
			if (!flip)
			{
				outLo = lo0; outHi = hi0;
				inLo = lo1; inHi = hi1;
			}
			else
			{
				outLo = lo1; outHi = hi1;
				inLo = lo0; inHi = hi0;
			}
		}

		return true;
	}


	private static int[] GetPow2DiffLut(int mod)
	{
		_pow2DiffModCache ??= new Dictionary<int, int[]>();
		if (_pow2DiffModCache.TryGetValue(mod, out int[]? lut))
		{
			return lut;
		}

		int[] values = new int[256];
		values[0] = mod > 1 ? 1 : 0;
		for (int i = 1; i < values.Length; i++)
		{
			values[i] = (values[i - 1] << 1) % mod;
		}

		_pow2DiffModCache[mod] = values;
		return values;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void ComputeForcedRemainingAndClean_Prefix(
		in ulong[] qMask, in ulong[] oneWin, in ulong[] zeroWin,
		int words, ulong lastMask,
		out int forced, out int remaining)
	{
		forced = 0;
		remaining = 0;

		int wordsMinusOne = words - 1;
		for (int i = 0; i < words; i++)
		{
			ulong unknownBits = qMask[i];
			if (i == wordsMinusOne) unknownBits &= lastMask;

			ulong one = oneWin[i];
			ulong known = one | zeroWin[i];

			// one is forcedBits here. We're reusing variable to limit registry pressure.
			one &= unknownBits;
			unknownBits &= ~known;

			forced += BitOperations.PopCount(one);
			remaining += BitOperations.PopCount(unknownBits);
		}
	}

	private static int ComputeQMod(ReadOnlySpan<int> qOneOffsets, int mod)
	{
		int count = qOneOffsets.Length;
		if (count == 0)
		{
			return 0;
		}

		int[] diffLut = GetPow2DiffLut(mod);
		int diffLutLength = diffLut.Length;
		long sum = 0;
		int prevOffset = qOneOffsets[0];
		long pow = PowMod2((ulong)prevOffset, mod);
		sum += pow;

		for (int i = 1; i < count; i++)
		{
			int offset = qOneOffsets[i];
			int diff = offset - prevOffset;
			if (diff <= 0)
			{
				prevOffset = offset;
				continue;
			}

			if (diff < diffLutLength)
			{
				pow = pow * diffLut[diff] % mod;
			}
			else
			{
				pow = pow * PowMod2((ulong)diff, mod) % mod;
			}

			sum += pow;
			if (sum >= mod)
			{
				sum %= mod;
			}

			prevOffset = offset;
		}

		return (int)(sum % mod);
	}

	private static bool ExistsCongruentInRange(BigInteger minA, BigInteger maxA, int residue, int mod)
	{
		if (minA > maxA)
		{
			return false;
		}

		int minMod = (int)((minA % mod + mod) % mod);
		int delta = residue - minMod;
		if (delta < 0)
		{
			delta += mod;
		}

		BigInteger candidate = minA + delta;
		return candidate <= maxA;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void GetQPrefixMask(int startColumn, int qBitLen, int qWordCount, out int words, out ulong lastMask)
	{
		// We want only offsets t <= startColumn.
		// start column is inclusive for maxT
		if (startColumn >= qBitLen - 1)
		{
			words = qWordCount;
			lastMask = ulong.MaxValue; // caller can still & lastWordMask separately
			return;
		}

		if (startColumn < 0)
		{
			words = 0;
			lastMask = 0;
			return;
		}

		qBitLen = startColumn >> 6;
		words = qBitLen + 1;

		qWordCount = startColumn & 63;
		lastMask = qWordCount == 63 ? ulong.MaxValue : ((1UL << (qWordCount + 1)) - 1UL);
	}

	private static void ComputeARangeFromTopConstraints(
		long maxAllowedA,
		bool top0KnownOne,
		bool top0KnownZero,
		bool top1KnownOne,
		long zeroTailStart,
		out BigInteger minA,
		out BigInteger maxA)
	{
		_dynamicModChecks++;
		if (zeroTailStart <= maxAllowedA + 1)
		{
			zeroTailStart--;
		}

		// zeroTailStart is effectiveTop from here. We're reusing variable to limit registry pressure.
		if (zeroTailStart < 0)
		{
			minA = BigInteger.Zero;
			maxA = BigInteger.Zero;
			return;
		}

		int shift = zeroTailStart + 1L > int.MaxValue ? int.MaxValue : (int)(zeroTailStart + 1L);
		maxA = Pow2Provider.BigIntegerMinusOne(shift);
		minA = BigInteger.Zero;

		if (zeroTailStart == maxAllowedA)
		{
			if (top0KnownOne)
			{
				shift = maxAllowedA > int.MaxValue ? int.MaxValue : (int)maxAllowedA;
			}
			else if (top0KnownZero && top1KnownOne && maxAllowedA - 1L >= 0)
			{
				// zeroTailStart is index from here. We're reusing the variable to limit registry pressure.
				zeroTailStart = maxAllowedA - 1L;
				shift = zeroTailStart > int.MaxValue ? int.MaxValue : (int)zeroTailStart;
			}

			minA = Pow2Provider.BigInteger(shift);
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool IsCleanInQ_Prefix(ulong[] qMask, ulong[] oneWin, ulong[] zeroWin, int words, ulong lastMask)
	{
		for (int i = 0; i < words; i++)
		{
			ulong q = qMask[i];
			if (i == words - 1) q &= lastMask;
			if ((q & (oneWin[i] | zeroWin[i])) != 0) return false;
		}

		return true;
	}

	private static bool TryDynamicModularRangePrune(
		ReadOnlySpan<int> qOneOffsets,
		long qHash,
		ulong p,
		long maxAllowedA,
		bool top0KnownOne,
		bool top0KnownZero,
		bool top1KnownOne,
		long zeroTailStart)
	{
		_dynamicModChecks++;
		long effectiveTop = maxAllowedA;
		if (zeroTailStart <= effectiveTop + 1)
		{
			effectiveTop = zeroTailStart - 1;
		}

		if (effectiveTop > 4096)
		{
			return true;
		}

		ComputeARangeFromTopConstraints(
			maxAllowedA,
			top0KnownOne,
			top0KnownZero,
			top1KnownOne,
			zeroTailStart,
			out BigInteger minA,
			out BigInteger maxA);

		int modLocalPrefilterModuliLength = ModLocalPrefilterModuli.Length;
		for (int i = 0; i < modLocalPrefilterModuliLength; i++)
		{
			int mod = ModLocalPrefilterModuli[i];
			int qMod = GetQModCached(qOneOffsets, qHash, mod);
			int target = ModPow2Minus1(p, mod);
			if (qMod == 0)
			{
				if (target != 0)
				{
					return false;
				}
				continue;
			}

			int inverse = ModPow(qMod, mod - 2, mod);
			int aResidue = (int)((long)target * inverse % mod);
			if (!ExistsCongruentInRange(minA, maxA, aResidue, mod))
			{
				_dynamicModFailures++;
				return false;
			}
		}

		return true;
	}

	private static int ModPow2Minus1(ulong p, int mod)
	{
		int pow = PowMod2(p, mod);
		int result = pow - 1;
		return result < 0 ? result + mod : result;
	}

	private static int PowMod2(ulong exp, int mod)
	{
		long result = 1;
		long baseVal = 2;
		ulong value = exp;
		while (value > 0)
		{
			if ((value & 1) != 0)
			{
				result = (result * baseVal) % mod;
			}

			baseVal = (baseVal * baseVal) % mod;
			value >>= 1;
		}

		return (int)result;
	}

	private static int ModPow(int value, int exp, int mod)
	{
		long result = 1 % mod;
		long baseVal = value % mod;
		int e = exp;
		while (e > 0)
		{
			if ((e & 1) != 0)
			{
				result = (result * baseVal) % mod;
			}

			baseVal = (baseVal * baseVal) % mod;
			e >>= 1;
		}

		return (int)result;
	}

	// internal static bool TryCheckDivisibilityFromDivisor(ulong q, ulong p, out bool divides, out ContradictionReason reason)
	// {
	// 	divides = false;
	// 	if (q == 0UL)
	// 	{
	// 		reason = ContradictionReason.ParityUnreachable;
	// 		return true;
	// 	}

	// 	Span<int> oneOffsets = stackalloc int[64];
	// 	int count = 0;
	// 	ulong remaining = q;
	// 	int baseIndex = 0;
	// 	while (remaining != 0UL)
	// 	{
	// 		int tz = BitOperations.TrailingZeroCount(remaining);
	// 		baseIndex += tz;
	// 		oneOffsets[count++] = baseIndex;
	// 		remaining >>= tz + 1;
	// 		baseIndex++;
	// 	}

	// 	return TryCheckDivisibilityFromOneOffsets(oneOffsets.Slice(0, count), p, out divides, out reason);
	// }

	[ThreadStatic]
	private static bool _initialized;

	public static void Initialize()
	{
		if (_initialized)
		{
			return;
		}

		// _privateWorkSet = new();
		// _stableUnknownDelta8 = new();
		_initialized = true;
		// Start with a small size and grow on demand.
		_stableUnknownDelta8ByUnknown = new int[64][];
		_privateWorkSet = new PrivateWorkSet();
	}

	/// <summary>
	/// Full column-wise propagation for a given set of q one-bit offsets (LSB=0) and exponent p.
	/// Returns true if the solver could decide locally (divides or contradiction).
	/// </summary>
#if DETAILED_LOG
	internal static bool TryCheckDivisibilityFromOneOffsets(
		ReadOnlySpan<int> qOneOffsets,
		ulong p,
		BigInteger k,
		bool isPowerOfTwo,
		out bool divides,
		out ContradictionReason reason)
#else
	internal static bool TryCheckDivisibilityFromOneOffsets(
		ReadOnlySpan<int> qOneOffsets,
		ulong p,
		BigInteger k,
		bool isPowerOfTwo,
		out bool divides)
#endif
	{
		PrivateWorkSet workSet = _privateWorkSet;

		divides = false;

#if DETAILED_LOG
		_lastTopDownFailure = null;
		_statStableAdvanceCalls = 0;
		_statStableAdvanceCols = 0;
		_statStableAdvanceColsBatched = 0;
		_statScalarCols = 0;
		_statDeltaRequests = 0;
		_statDeltaHits = 0;
		_statDeltaBuilds = 0;
		_statStableAdvanceBigShift = 0;
		_statCleanChecks = 0;
		_statCleanFalse = 0;
		_statFirstDirtyColumn = -1;
#endif
		_dynamicModChecks = 0;
		_dynamicModFailures = 0;

		// These conditions will never be true on EvenPerfectBitScanner execution path.
		// if (p == 0 || qOneOffsets.Length == 0)
		// {
		//     reason = ContradictionReason.ParityUnreachable;
		//     return true;
		// }

		// While the scanner supports p >= 31, it's a tiny subset of all checked exponents. No need to support it.
		// Exact small-case evaluation to short-circuit both success and failure deterministically.
		// int highestBit = qOneOffsets[^1];
		// if (p <= 64 && highestBit < 63)
		// {
		//     ulong q = 0;
		//     foreach (int bit in qOneOffsets)
		//     {
		//         q |= 1UL << bit;
		//     }

		//     ulong mersenne = (1UL << (int)p) - 1UL;
		//     divides = q != 0 && (mersenne % q == 0);
		//     reason = divides ? ContradictionReason.None : ContradictionReason.ParityUnreachable;
		//     return true;
		// }

		int qOneOffsetsLength = qOneOffsets.Length;
		int maxOffset = qOneOffsets[qOneOffsetsLength - 1];
		int qBitLen = maxOffset + 1;
		if ((ulong)qBitLen >= p)
		{
#if DETAILED_LOG
			reason = ContradictionReason.TruncatedLength;
#endif
			return true;
		}

		// q from EvenPerfectBitScanner will always have the correct 2kp+1 form.
		// if (qOneOffsets[0] != 0)
		// {
		//     reason = ContradictionReason.ParityUnreachable;
		//     return true;
		// }

		const int ForcedALowBits = 1024;
		int windowSize = qBitLen < ForcedALowBits ? ForcedALowBits : qBitLen;
		windowSize = (windowSize + 63) & ~63;
		int aWordCount = (windowSize + 63) >> 6;
		int qWordCount = (qBitLen + 63) >> 6;
		workSet.EnsureCapacity(aWordCount, qWordCount);
		ulong[] knownOne0 = workSet.KnownOne0;
		ulong[] knownOne1 = workSet.KnownOne1;
		ulong[] knownZero0 = workSet.KnownZero0;
		ulong[] knownZero1 = workSet.KnownZero1;
		ulong[] qMaskWords = workSet.QMaskWords;

		Array.Clear(knownOne0, 0, aWordCount);
		Array.Clear(knownOne1, 0, aWordCount);
		Array.Clear(knownZero0, 0, aWordCount);
		Array.Clear(knownZero1, 0, aWordCount);
		Array.Clear(qMaskWords, 0, qWordCount);

		int segmentBase = 0;
		int pLong = (int)p;
		int maxAllowedA = pLong - qBitLen;
		if (maxAllowedA < 0)
		{
#if DETAILED_LOG
			reason = ContradictionReason.TruncatedLength;
#endif
			return true;
		}

		// Force low bits of 'a' from a ≡ -q^{-1} (mod 2^1024).
		// This is a necessary condition for q | (2^p - 1) when p >= 1024 (true here).
		const int ForcedBits = ForcedALowBits;
		const int ForcedBitsMinusOne = ForcedALowBits - 1;
		BigInteger mask = Pow2Provider.BigIntegerMinusOne(ForcedBits);
		BigInteger qMod2k;
		bool usedPow2Transform = false;
		if (isPowerOfTwo && workSet.HasLastPow2QMod2k && workSet.LastPow2K != BigInteger.Zero && k == (workSet.LastPow2K << 1))
		{
			qMod2k = ((workSet.LastPow2QMod2k << 1) - BigInteger.One) & mask;
			usedPow2Transform = true;
		}
		else
		{
			qMod2k = BigInteger.Zero;
		}

		byte[] qMod2kBytes = workSet.QMod2kBytes;
		int i, t;
		ulong[] bitMask64 = BitMask64;
		if (usedPow2Transform)
		{
			for (i = 0; i < qOneOffsetsLength; i++)
			{
				t = qOneOffsets[i];
				qMaskWords[t >> 6] |= bitMask64[t & 63];
			}
		}
		else
		{
			Array.Clear(qMod2kBytes, 0, qMod2kBytes.Length);
			int currentByte = -1;
			byte currentValue = 0;
			for (i = 0; i < qOneOffsetsLength; i++)
			{
				t = qOneOffsets[i];
				if (t < ForcedBits)
				{
					int byteIndex = t >> 3;
					if (byteIndex != currentByte)
					{
						if (currentByte >= 0)
						{
							qMod2kBytes[currentByte] = currentValue;
						}
						currentByte = byteIndex;
						currentValue = 0;
					}
					currentValue |= (byte)(1 << (t & 7));
				}

				qMaskWords[t >> 6] |= bitMask64[t & 63];
			}
			if (currentByte >= 0)
			{
				qMod2kBytes[currentByte] = currentValue;
			}

			qMod2k = new BigInteger(qMod2kBytes, isUnsigned: true, isBigEndian: false);
		}

		// Compute inv = q^{-1} mod 2^1024 using Hensel/Newton iteration in Z/2^kZ.
		BigInteger[] invByKMod = workSet.InvByKMod;
		bool[] hasInvByKMod = workSet.HasInvByKMod;
		int kMod = (int)(k & 1023);
		BigInteger inv;
		if (hasInvByKMod[kMod])
		{
			inv = invByKMod[kMod];
		}
		else if (workSet.HasLastInvMod2k)
		{
			inv = workSet.LastInvMod2k;
		}
		else
		{
			inv = qMod2k;
		}

		for (int iter = 0; iter < 2; iter++)
		{
			inv = (inv * (BigIntegerNumbers.Two - (qMod2k * inv))) & mask;
		}

		if (((qMod2k * inv) & mask) != BigInteger.One)
		{
			inv = qMod2k;
			for (int iter = 0; iter < 10; iter++)
			{
				inv = (inv * (BigIntegerNumbers.Two - (qMod2k * inv))) & mask;
			}
		}

		BigInteger aLow = (-inv) & mask;

		if (isPowerOfTwo)
		{
			workSet.LastPow2QMod2k = qMod2k;
			workSet.LastPow2K = k;
			workSet.HasLastPow2QMod2k = true;
		}

		workSet.LastInvMod2k = inv;
		workSet.HasLastInvMod2k = true;
		invByKMod[kMod] = inv;
		hasInvByKMod[kMod] = true;
		// a ≡ -aLow (mod 2^512)
		// Apply constraints for bits i in [0..min(511, maxAllowedA)] into knownOne0/knownZero0.
		// NOTE: now windowSize >= 512, so wordCount is enough to store these bits.
		Span<byte> aLowBytes = stackalloc byte[128];
		if (!aLow.TryWriteBytes(aLowBytes, out int aLowBytesWritten, isUnsigned: true, isBigEndian: false))
		{
			aLowBytesWritten = 0;
		}
		int maxKnownA;
		int maxFixed = maxAllowedA < ForcedBitsMinusOne ? maxAllowedA : ForcedBitsMinusOne;
		int word;
		if (maxFixed >= 0)
		{
			int lastWord = maxFixed >> 6;
			int lastBit = maxFixed & 63;
			int fullWords = lastBit == 63 ? lastWord + 1 : lastWord;
			for (int w = 0; w < fullWords; w++)
			{
				int byteIndex = w << 3;
				ulong wordValue = 0;
				int limit = aLowBytesWritten - byteIndex;
				if (limit > 0)
				{
					if (limit > 8)
					{
						limit = 8;
					}
					for (int b = 0; b < limit; b++)
					{
						wordValue |= (ulong)aLowBytes[byteIndex + b] << (b << 3);
					}
				}
				knownOne0[w] = wordValue;
				knownZero0[w] = ~wordValue;
			}

			if (fullWords <= lastWord)
			{
				int byteIndex = lastWord << 3;
				ulong wordValue = 0;
				int limit = aLowBytesWritten - byteIndex;
				if (limit > 0)
				{
					if (limit > 8)
					{
						limit = 8;
					}
					for (int b = 0; b < limit; b++)
					{
						wordValue |= (ulong)aLowBytes[byteIndex + b] << (b << 3);
					}
				}
				ulong mask64 = (bitMask64[lastBit + 1]) - 1UL;
				knownOne0[lastWord] = wordValue & mask64;
				knownZero0[lastWord] = (~wordValue) & mask64;
			}
		}

		// We have learned bits up to maxFixed in the current segment.
		maxKnownA = -1;
		if (maxFixed > maxKnownA)
		{
			maxKnownA = maxFixed;
		}

		// ------------------------------------------------------------
		// Cheap low-column prefilter (prefix-style):
		// For the first N columns, all a-indices that contribute to the column are within
		// the forced low-bit region (aLow). When every contributing bit is known (remaining==0),
		// we can propagate carry deterministically and reject immediately on contradiction.
		// This is significantly cheaper than entering the full row-aware window DP.
		// ------------------------------------------------------------
		const int PrefilterColumns = 512; // 256..512 is a good trade-off; keep <= 1024 forced bits
		int preMax = PrefilterColumns - 1;
		if (preMax > maxFixed)
		{
			preMax = maxFixed;
		}

		if (preMax > maxAllowedA)
		{
			preMax = maxAllowedA;
		}
		// Do NOT clamp to maxOffset: for columns above maxOffset the full set of q=1 positions
		// still contributes, and aLow still fixes the corresponding a-bits.

		// Only run if we have at least a few columns to check.
		int col;
		if (preMax >= 8)
		{
			CarryRange preCarry = CarryRange.Zero;
			for (col = 0; col <= preMax; col++)
			{
				// Required result bit for M_p in columns [0..p-1] is 1.
				const int requiredBit = 1;

				// Compute forced ones exactly: sum of a_{col - t} for all q_t=1 with t<=col.
				int forced = 0;
				for (i = 0; i < qOneOffsetsLength; i++)
				{
					t = qOneOffsets[i];
					if (t > col)
					{
						break;
					}

					int aIndex = col - t;
					// For this prefilter we only run where aIndex is guaranteed within [0..maxFixed].
					word = aIndex >> 6;
					ulong m64 = bitMask64[aIndex & 63];
					if ((knownOne0[word] & m64) != 0) forced++;
				}

				// remaining==0 here, so possible==forced. Propagate carry range deterministically.
				if (!TryPropagateCarry(ref preCarry, forced, forced, requiredBit))
				{
#if DETAILED_LOG
						reason = ContradictionReason.ParityUnreachable;
#endif
					divides = false;
					return true;
				}
			}
		}

		int topIndex0 = maxAllowedA;
		int topIndex1 = maxAllowedA - 1;
		bool top0KnownOne = topIndex0 == 0;
		bool top1KnownOne = topIndex1 == 0;
		// bool top0KnownZero = false;
		// bool top1KnownZero = false;
		int zeroTailStart = maxAllowedA + 1;

		// long qHash = ComputeQOffsetsHash(qOneOffsets);


		// if (!TryModularPrefilter(qOneOffsets, qHash, p, maxAllowedA))
		// {
		// 	reason = ContradictionReason.ParityUnreachable;
		// 	return true;
		// }

		// Ultra-cheap high-bit feasibility check based on the forced top carry-out (no bit beyond p-1).
		// This is a tiny reverse-carry DP over the last few columns (p-1, p-2, ...), using only the
		// geometric availability of contributions implied by qOneOffsets and maxAllowedA.
		// It can reject entire classes of q with sparse/awkward high-bit structure before the full DP runs.
#if DETAILED_LOG
		if (!TryRunHighBitAndBorrowPrefiltersCombined(qOneOffsets, qOneOffsetsLength, pLong, maxAllowedA, out _lastTopDownFailure))
#else
		if (!TryRunHighBitAndBorrowPrefiltersCombined(qOneOffsets, qOneOffsetsLength, pLong, maxAllowedA))
#endif
		{
#if DETAILED_LOG
			reason = ContradictionReason.ParityUnreachable;
#endif
			return true;
		}

		// Row-aware masks for q and a-window (supports q > 64 bits)
		ulong lastWordMask = (qBitLen & 63) == 0 ? ulong.MaxValue : ((bitMask64[qBitLen & 63]) - 1UL);
		// Initialize window for column 0: mark negative a indices as zero (t>0).
		// Sliding window over a: bit t corresponds to a_{column - t}.
		ulong[] aOneWin = workSet.AOneWin;
		ulong[] aZeroWin = workSet.AZeroWin;
		Array.Clear(aOneWin, 0, qWordCount);
		Array.Clear(aZeroWin, 0, qWordCount);
		int windowColumn = 0;

		// Insert a_0 status into bit0
		int st = GetAKnownStateRowAware(0, segmentBase, windowSize, knownOne0, knownZero0, knownOne1, knownZero1);
		if (st == 1)
		{
			aOneWin[0] |= 1UL;
		}
		else if (st == 2)
		{
			aZeroWin[0] |= 1UL;
		}

		int lowColumn = 0;
		int highColumn = pLong - 1;
		CarryRange carryLow = CarryRange.Zero;
		// int bottomRunStart = 0;
		// int bottomRunEnd = 0;

		// -----------------------------
		// Option B mini ping-pong prefilter (chunk + reseed)
		// -----------------------------
		// int chunk = MiniPingPongPrefilterColumns; // Currently 60
		// int maxDepth = 60;                       // Set it to how many you want to test (e.g. 960)
		// if (maxDepth > highColumn) maxDepth = highColumn;

		// long seedMin = 0;
		// long seedMax = 0;
		// int currentStartColumn = pLong - 1; // We start from the very top (column p-1)

		// for (int depth = chunk; depth <= maxDepth; depth += chunk)
		// {
		// 	int targetJ = pLong - depth;
		// 	if (targetJ <= 0 || targetJ > highColumn) break;

		// 	// 1) Bottom-up DP only up to column J (process columns lowColumn..J-1).
		// 	while (lowColumn < targetJ)
		// 	{
		// 		int remaining = targetJ - lowColumn;
		// 		int blockSize = remaining < PingPongBlockColumns ? remaining : PingPongBlockColumns;
		// 		if (!TryProcessBottomUpBlockRowAware(
		// 			qOneOffsets, qMaskWords, qWordCount, lastWordMask, maxAllowedA,
		// 			lowColumn, blockSize,
		// 			ref carryLow, ref maxKnownA, ref segmentBase,
		// 			ref knownOne0, ref knownOne1, ref knownZero0, ref knownZero1,
		// 			windowSize,
		// 			aOneWin, aZeroWin, ref windowColumn,
		// 			out reason))
		// 		{
		// 			return true;
		// 		}

		// 		lowColumn += blockSize;
		// 	}

		// 	// 2) Top-down for exactly 'chunk' columns, seeded from previous intersection.
		// 	if (!TryHighBitCarryRangeToColumnJ(
		// 			qOneOffsets, qOneOffsetsLength, pLong, maxAllowedA,
		// 			startColumn: currentStartColumn,
		// 			steps: chunk,
		// 			startCarryOutMin: seedMin,
		// 			startCarryOutMax: seedMax,
		// 			out long topMin, out long topMax))
		// 	{
		// 		Console.WriteLine("Pruned with TryHighBitCarryRangeToColumnJ");
		// 		reason = ContradictionReason.ParityUnreachable;
		// 		return true;
		// 	}

		// 	// 3) Intersect with bottom-up carry at the SAME targetJ.
		// 	long interMin = carryLow.Min > topMin ? carryLow.Min : topMin;
		// 	long interMax = carryLow.Max < topMax ? carryLow.Max : topMax;
		// 	if (interMin > interMax)
		// 	{
		// 		Console.WriteLine("Pruned with interMin > interMax");
		// 		reason = ContradictionReason.ParityUnreachable;
		// 		return true;
		// 	}

		// 	// 4) Reseed for the next chunk: carryOut above next block is our intersection.
		// 	seedMin = interMin;
		// 	seedMax = interMax;

		// 	// Next chunk starts right above the current targetJ.
		// 	currentStartColumn = targetJ - 1;

		// 	// Also tighten carryLow (optional but consistent).
		// 	carryLow = new CarryRange(interMin, interMax);

		// 	// Safety: if we fell off the bottom, stop.
		// 	if (currentStartColumn < 0) break;
		// }

		// -----------------------------
		// Continue full bottom-up DP for remaining columns
		// -----------------------------
		while (lowColumn <= highColumn)
		{
			int remaining = highColumn - lowColumn + 1;
			int blockSize = remaining < PingPongBlockColumns ? remaining : PingPongBlockColumns;
#if DETAILED_LOG
			if (!TryProcessBottomUpBlockRowAware(
				qOneOffsets, qMaskWords, qWordCount, lastWordMask, maxAllowedA,
				lowColumn, blockSize,
				ref carryLow, ref maxKnownA, ref segmentBase,
				ref knownOne0, ref knownOne1, ref knownZero0, ref knownZero1,
				windowSize, wordCount, aOneWin, aZeroWin, ref windowColumn,
				out reason))
#else
			if (!TryProcessBottomUpBlockRowAware(
				qOneOffsets, qMaskWords, qWordCount, lastWordMask, maxAllowedA,
				lowColumn, blockSize,
				ref carryLow, ref maxKnownA, ref segmentBase,
				ref knownOne0, ref knownOne1, ref knownZero0, ref knownZero1,
				windowSize, aWordCount, aOneWin, aZeroWin, ref windowColumn))
#endif
			{
#if DETAILED_LOG
					PrintStats();
#endif

				return true;
			}

			lowColumn += blockSize;
		}

		// long carryMin = carryLow.Min > carryHigh.Min ? carryLow.Min : carryHigh.Min;
		// long carryMax = carryLow.Max < carryHigh.Max ? carryLow.Max : carryHigh.Max;
		// if (carryMin > carryMax)
		// {
		// 	divides = false;
		// 	reason = ContradictionReason.ParityUnreachable;
		// 	return true;
		// }

		// if (_debugEnabled)
		// {
		// 	Console.WriteLine($"[bit-contradiction] AMultiplier dynamic-mod checks={_dynamicModChecks} failures={_dynamicModFailures}");
		// }

		divides = true;
#if DETAILED_LOG
		reason = ContradictionReason.None;
		PrintStats();
#endif
		return true;
	}

#if DETAILED_LOG
	internal static bool TryCheckDivisibilityFromOneOffsets(
		ReadOnlySpan<int> qOneOffsets,
		ulong p,
		out bool divides) => TryCheckDivisibilityFromOneOffsets(qOneOffsets, p, BigInteger.Zero, false, out divides, out _);
#endif
#if !DETAILED_LOG
	internal static bool TryCheckDivisibilityFromOneOffsets(
		ReadOnlySpan<int> qOneOffsets,
		ulong p,
		out bool divides) => TryCheckDivisibilityFromOneOffsets(qOneOffsets, p, BigInteger.Zero, false, out divides);
#endif

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ulong BuildQLow64(ReadOnlySpan<int> qOneOffsets)
	{
		ulong q = 0;
		for (int i = 0; i < qOneOffsets.Length; i++)
		{
			int bit = qOneOffsets[i];
			if ((uint)bit >= 64u) break;
			q |= 1UL << bit;
		}
		return q;
	}

	/// <summary>
	/// Modular inverse of odd q modulo 2^64.
	/// Uses Newton/Hensel lifting: x_{k+1} = x_k * (2 - q*x_k) mod 2^64.
	/// </summary>
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ulong ModInversePow2_64(ulong qOdd)
	{
		// qOdd must be odd.
		// Start with x ≡ 1 (mod 2). A common fast seed is x = qOdd.
		// Each iteration doubles correct bits. 6 iterations are enough for 64 bits: 1->2->4->8->16->32->64.
		unchecked
		{
			qOdd *= 2UL - qOdd * qOdd;
			qOdd *= 2UL - qOdd * qOdd;
			qOdd *= 2UL - qOdd * qOdd;
			qOdd *= 2UL - qOdd * qOdd;
			qOdd *= 2UL - qOdd * qOdd;
			qOdd *= 2UL - qOdd * qOdd;
		}

		return qOdd;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static bool TryAdvanceZeroColumns(ref CarryRange carry, int columnCount)
	{
		const int Step = 30;
		long carryMin = carry.Min;
		long carryMax = carry.Max;
		int add;
		while (columnCount >= Step)
		{
			add = (1 << Step) - 1;
			carryMin = (carryMin + add) >> Step;
			carryMax >>= Step;
			if (carryMin > carryMax)
			{
				return false;
			}

			columnCount -= Step;
		}

		if (columnCount > 0)
		{
			add = (1 << columnCount) - 1;
			carryMin = (carryMin + add) >> columnCount;
			carryMax >>= columnCount;
			if (carryMin > carryMax)
			{
				return false;
			}
		}

		carry = new CarryRange(carryMin, carryMax);
		return true;
	}

	// [ThreadStatic]
	// private static Dictionary<int, int[]> _stableUnknownDelta8;

	// Indexed by "unknown" (usually qOneOffsetsLength). Elements are lazy initialized.
	private static volatile int[][]? _stableUnknownDelta8ByUnknown;

	private static int[] GetStableUnknownDelta(int unknown)
	{
#if DETAILED_LOG
		_statDeltaRequests++;
#endif

		// Lazily allocate the outer array. Keep it volatile to publish safely.
		int[][] cache = _stableUnknownDelta8ByUnknown!;

		// Ensure capacity for this "unknown".
		int cacheLength = cache.Length,
			parity;

		if ((uint)unknown >= (uint)cacheLength)
		{
			// Grow to at least unknown+1, doubling for amortized O(1). parity means newLength here - we're reusing variable to limit registry pressure.
			parity = cacheLength;
			while (parity <= unknown) parity <<= 1;

			var newCache = new int[parity][];
			Array.Copy(cache, newCache, cacheLength);

			// Publish the grown cache (racy publish is OK; contents are immutable after creation).
			_stableUnknownDelta8ByUnknown = newCache;
			cache = newCache;
		}

		// Fast path: already computed.
		var cached = cache[unknown];
		if (cached != null)
		{
#if DETAILED_LOG
			_statDeltaHits++;
#endif
			return cached;
		}

		// Compute.
		const int PrefixLength = 1 << ColumnsAtOnce;
		int[] delta = new int[PrefixLength];

		for (int low = 0; low < PrefixLength; low++)
		{
			long value = low;

			// 16 iterations correspond to processing 16 columns at once. It was slower with: 32, 128, 1024 and rolled.
			for (int i = 0; i < ColumnsAtOnce; i += 4)
			{
				parity = (int)(value & 1L);
				int sMax = (((unknown ^ parity) & 1) != 0) ? unknown : (unknown - 1);
				value = (value + sMax - 1) >> 1;

				parity = (int)(value & 1L);
				sMax = (((unknown ^ parity) & 1) != 0) ? unknown : (unknown - 1);
				value = (value + sMax - 1) >> 1;

				parity = (int)(value & 1L);
				sMax = (((unknown ^ parity) & 1) != 0) ? unknown : (unknown - 1);
				value = (value + sMax - 1) >> 1;

				parity = (int)(value & 1L);
				sMax = (((unknown ^ parity) & 1) != 0) ? unknown : (unknown - 1);
				value = (value + sMax - 1) >> 1;
			}

			delta[low] = (int)((value << ColumnsAtOnce) - low);
		}

#if DETAILED_LOG
		_statDeltaBuilds++;
#endif
		// Publish into cache. If a race computes it twice, that's acceptable (same deterministic result).
		cache[unknown] = delta;
		return delta;
	}

	// private static int[] GetStableUnknownDelta8(int unknown)
	// {
	// 	var stableUnknownDelta8Cache = _stableUnknownDelta8;
	// 	if (stableUnknownDelta8Cache.TryGetValue(unknown, out int[]? cached))
	// 	{
	// 		return cached!;
	// 	}

	// 	const int PrefixLength = 131072;
	// 	int[] delta = new int[PrefixLength];
	// 	for (int low = 0; low < PrefixLength; low++)
	// 	{
	// 		long value = low;
	// 		for (int i = 0; i < 8; i++)
	// 		{
	// 			int sMax = (((unknown ^ (int)value) & 1) != 0) ? unknown : unknown - 1;
	// 			value = (value + sMax - 1) >> 1;
	// 		}

	// 		delta[low] = (int)((value << 8) - low);
	// 	}

	// 	stableUnknownDelta8Cache[unknown] = delta;
	// 	return delta;
	// }

	const int ColumnsAtOnce = 24;

#if DETAILED_LOG
	private static bool TryAdvanceStableUnknown(ref CarryRange carry, int unknown, int columnCount, out ContradictionReason reason)
#else
	private static bool TryAdvanceStableUnknown(ref CarryRange carry, int unknown, int columnCount)
#endif
	{
#if DETAILED_LOG
		_statStableAdvanceCalls++;
		_statStableAdvanceCols += columnCount;
		if (columnCount >= 64) _statStableAdvanceBigShift++;
#endif

		if (columnCount <= 0)
		{
#if DETAILED_LOG
			reason = ContradictionReason.None;
			_statStableAdvanceColsBatched += columnCount;
#endif
			return true;
		}

		if (unknown <= 0)
		{
			if (!TryAdvanceZeroColumns(ref carry, columnCount))
			{
#if DETAILED_LOG
				reason = ContradictionReason.ParityUnreachable;
#endif
				return false;
			}

#if DETAILED_LOG
			_statStableAdvanceColsBatched += columnCount;
			reason = ContradictionReason.None;
#endif
			return true;
		}

		long carryMin = columnCount >= 63 ? 0 : (carry.Min >> columnCount);
		long carryMax = carry.Max;
		int remaining = columnCount;
		int[] delta = GetStableUnknownDelta(unknown);

		const long CarryDecimalMask = (1L << ColumnsAtOnce) - 1L;
		int carryMask;
		while (remaining >= ColumnsAtOnce)
		{
			carryMask = (int)(carryMax & CarryDecimalMask);
			carryMax = (carryMax + delta[carryMask]) >> ColumnsAtOnce;
			remaining -= ColumnsAtOnce;
		}

		while (remaining > 0)
		{
			// carryMask is sMax here. We're reusing variable to limit registry pressure.
			carryMask = (((unknown ^ (int)carryMax) & 1) != 0) ? unknown : unknown - 1;
			carryMax = (carryMax + carryMask - 1) >> 1;
			remaining--;
		}

		if (carryMax < carryMin)
		{
#if DETAILED_LOG
			reason = ContradictionReason.ParityUnreachable;
#endif
			return false;
		}

#if DETAILED_LOG
		_statStableAdvanceColsBatched += columnCount;
		reason = ContradictionReason.None;
#endif
		carry = new CarryRange(carryMin, carryMax);
		return true;
	}


	// -----------------------------
	// Row-aware bottom-up block processing
	// -----------------------------
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void ShiftLeft1InPlace(ulong[] wordsOne, ulong[] wordsZero, int wordCount, ulong lastWordMask)
	{
		if (wordCount > 0)
		{
			ulong carryOne = 0,
				  carryZero = 0;

			for (int i = 0; i < wordCount; i++)
			{
				ulong w = wordsOne[i];
				wordsOne[i] = (w << 1) | carryOne;
				carryOne = w >> 63;

				w = wordsZero[i];
				wordsZero[i] = (w << 1) | carryZero;
				carryZero = w >> 63;
			}

			wordCount--;
			wordsOne[wordCount] &= lastWordMask;
			wordsZero[wordCount] &= lastWordMask;
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool AnyAndNonZero(ulong[] a, ulong[] b, int wordCount)
	{
		for (int i = 0; i < wordCount; i++)
		{
			if ((a[i] & b[i]) != 0) return true;
		}
		return false;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static int PopCountAnd(ulong[] a, ulong[] b, int wordCount)
	{
		int c = 0;
		for (int i = 0; i < wordCount; i++)
		{
			c += BitOperations.PopCount(a[i] & b[i]);
		}
		return c;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static int PopCountUnknownInQ(ulong[] qMask, ulong[] oneWin, ulong[] zeroWin, int wordCount)
	{
		int c = 0;
		for (int i = 0; i < wordCount; i++)
		{
			ulong known = oneWin[i] | zeroWin[i];
			ulong unkQ = qMask[i] & ~known;
			c += BitOperations.PopCount(unkQ);
		}
		return c;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool TryFindSingleUnknownInQ(ulong[] qMask, ulong[] oneWin, ulong[] zeroWin, int wordCount, out int offset)
	{
		// Returns true if exactly one unknown bit among positions where q has 1.
		offset = -1;
		int found = 0;
		for (int w = 0; w < wordCount; w++)
		{
			ulong known = oneWin[w] | zeroWin[w];
			ulong unk = qMask[w] & ~known;
			if (unk == 0)
			{
				continue;
			}

			int pc = BitOperations.PopCount(unk);
			found += pc;
			if (found > 1)
			{
				return false;
			}

			int bit = BitOperations.TrailingZeroCount(unk);
			offset = (w << 6) + bit;
		}

		return found == 1;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static int GetAKnownStateRowAware(
		int aIndex,
		int segmentBase,
		int windowSize,
		in ulong[] knownOne0,
		in ulong[] knownZero0,
		in ulong[] knownOne1,
		in ulong[] knownZero1)
	{
		// 0 unknown, 1 one, 2 zero
		if (aIndex < 0)
		{
			return 2;
		}

		int rel, w;
		ulong m;
		if (aIndex >= segmentBase && aIndex < segmentBase + windowSize)
		{
			rel = aIndex - segmentBase;
			w = rel >> 6;
			m = 1UL << (rel & 63);
			if ((knownOne0[w] & m) != 0)
			{
				return 1;
			}

			if ((knownZero0[w] & m) != 0)
			{
				return 2;
			}

			return 0;
		}

		w = segmentBase - windowSize;
		if (aIndex >= w && aIndex < segmentBase)
		{
			rel = aIndex - w;
			w = rel >> 6;
			m = 1UL << (rel & 63);
			if ((knownOne1[w] & m) != 0)
			{
				return 1;
			}

			if ((knownZero1[w] & m) != 0)
			{
				return 2;
			}

			return 0;
		}

		return 0;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void SetAKnownRowAware(
		int aIndex,
		bool valueOne,
		int segmentBase,
		int windowSize,
		ulong[] knownOne0,
		ulong[] knownZero0,
		ulong[] knownOne1,
		ulong[] knownZero1)
	{
		if (aIndex < 0) return;

		int rel, w;
		ulong m;
		if (aIndex >= segmentBase && aIndex < segmentBase + windowSize)
		{
			rel = aIndex - segmentBase;
			w = rel >> 6;
			m = 1UL << (rel & 63);
			if (valueOne)
				knownOne0[w] |= m;
			else
				knownZero0[w] |= m;
			return;
		}

		// windowSize is seg1Start here. We're reusing the variable to limit registry pressure.
		windowSize = segmentBase - windowSize;
		if (aIndex >= windowSize && aIndex < segmentBase)
		{
			rel = aIndex - windowSize;
			w = rel >> 6;
			m = 1UL << (rel & 63);
			if (valueOne)
				knownOne1[w] |= m;
			else
				knownZero1[w] |= m;
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool TryFindSingleUnknownInQ_Prefix(
		in ulong[] qMask, in ulong[] oneWin, in ulong[] zeroWin,
		int words, ulong lastMask,
		out int offset)
	{
		offset = -1;
		int found = 0;

		for (int w = 0; w < words; w++)
		{
			ulong q = qMask[w];
			if (w == words - 1) q &= lastMask;

			ulong unknown = oneWin[w] | zeroWin[w];
			unknown = q & ~unknown;
			if (unknown == 0) continue;

			found += BitOperations.PopCount(unknown);
			if (found > 1) return false;

			int bit = BitOperations.TrailingZeroCount(unknown);
			offset = (w << 6) + bit;
		}

		return found == 1;
	}

#if DETAILED_LOG
	private static bool TryProcessBottomUpBlockRowAware(
		ReadOnlySpan<int> qOneOffsets,
		in ulong[] qMaskWords,
		int qWordCount,
		ulong lastWordMask,
		int maxAllowedA,
		int startColumn,
		int columnCount,
		ref CarryRange carry,
		ref int maxKnownA,
		ref int segmentBase,
		ref ulong[] knownOne0,
		ref ulong[] knownOne1,
		ref ulong[] knownZero0,
		ref ulong[] knownZero1,
		int windowSize,
		int wordCount,
		ulong[] aOneWin,
		ulong[] aZeroWin,
		ref int windowColumn,
		out ContradictionReason reason)
#else
	private static bool TryProcessBottomUpBlockRowAware(
		ReadOnlySpan<int> qOneOffsets,
		in ulong[] qMaskWords,
		int qWordCount,
		ulong lastWordMask,
		int maxAllowedA,
		int startColumn,
		int columnCount,
		ref CarryRange carry,
		ref int maxKnownA,
		ref int segmentBase,
		ref ulong[] knownOne0,
		ref ulong[] knownOne1,
		ref ulong[] knownZero0,
		ref ulong[] knownZero1,
		int windowSize,
		int wordCount,
		ulong[] aOneWin,
		ulong[] aZeroWin,
		ref int windowColumn)
#endif
	{
		bool needRebuildWindow = false;
		// columnCount is endColumn from here. We're reusing the variable to limit registry pressure.
		columnCount += startColumn;
		int stableUnknown = qOneOffsets.Length;
		int maxOffset = qOneOffsets[stableUnknown - 1];
		ulong[] temp;

		// Ensure window is aligned.
		ArgumentOutOfRangeException.ThrowIfNotEqual(windowColumn, startColumn);

		int maxOffsetPlusOne = maxOffset + 1,
			qWordCountMinusOne = qWordCount - 1,
			endColumnMinusOne = columnCount - 1;

		while (startColumn < columnCount)
		{
			int aIndex, remaining, state, step;
			// Segment rollover based on current column (because we may set aIndex within [startColumn-maxOffset..startColumn]).
			int bound = (startColumn / windowSize) * windowSize;

			if (bound != segmentBase)
			{
				if (bound == segmentBase + windowSize)
				{
					temp = knownOne0; knownOne0 = knownOne1; knownOne1 = temp;
					temp = knownZero0; knownZero0 = knownZero1; knownZero1 = temp;
					Array.Clear(knownOne0, 0, wordCount);
					Array.Clear(knownZero0, 0, wordCount);
				}
				else
				{
					Array.Clear(knownOne0, 0, wordCount);
					Array.Clear(knownOne1, 0, wordCount);
					Array.Clear(knownZero0, 0, wordCount);
					Array.Clear(knownZero1, 0, wordCount);
					// Also clear window because it may reference bits outside cache.
					needRebuildWindow = true;
				}

				segmentBase = bound;
				if (needRebuildWindow)
				{
					Array.Clear(aOneWin, 0, qWordCount);
					Array.Clear(aZeroWin, 0, qWordCount);

					// Build window for current startColumn: bit t corresponds to a_{startColumn - t}
					for (step = 0; step <= maxOffset; step++)
					{
						aIndex = startColumn - step;
						state = aIndex < 0 || aIndex > maxAllowedA
							? 2
							: GetAKnownStateRowAware(aIndex, segmentBase, windowSize, knownOne0, knownZero0, knownOne1, knownZero1);

						if (state == 1)
						{
							aOneWin[step >> 6] |= 1UL << (step & 63);
						}
						else if (state == 2)
						{
							aZeroWin[step >> 6] |= 1UL << (step & 63);
						}
					}

					if (qWordCount > 0)
					{
						aOneWin[qWordCountMinusOne] &= lastWordMask;
						aZeroWin[qWordCountMinusOne] &= lastWordMask;
					}

					needRebuildWindow = false;
				}
			}

			int qWordsEff = qWordCount;
			ulong qLastEff = lastWordMask;
			if (startColumn < maxOffset)
			{
				GetQPrefixMask(startColumn, maxOffsetPlusOne, qWordCount, out qWordsEff, out qLastEff);
			}

			// StableUnknown batching is valid only if the window contains no known bits in the q-active positions.
			if (startColumn >= maxOffset && startColumn <= maxAllowedA)
			{
				remaining = endColumnMinusOne;
				if (remaining > maxAllowedA)
				{
					remaining = maxAllowedA;
				}

				remaining = remaining - startColumn + 1;
				if (remaining > 0)
				{
					// Additionally, ensure current window has no known bits (should be true if maxKnownA condition holds).
					// Conservative batching: ignore row-aware window state and assume all q positions are unknown.
					// This cannot create false negatives; it may only reduce pruning power.
#if DETAILED_LOG
					_statBatchEntry++;
					_statBatchCols += remainingCols;
#endif

					while (remaining >= TailCarryBatchColumns)
					{
#if DETAILED_LOG
						if (!TryAdvanceStableUnknown(ref carry, stableUnknown, TailCarryBatchColumns, out var propagateReason))
#else
						if (!TryAdvanceStableUnknown(ref carry, stableUnknown, TailCarryBatchColumns))
#endif
						{
#if DETAILED_LOG
							reason = propagateReason;
#endif
							return false;
						}

						startColumn += TailCarryBatchColumns;
						windowColumn = startColumn;
						remaining -= TailCarryBatchColumns;
					}

					if (remaining > 0)
					{
#if DETAILED_LOG
						if (!TryAdvanceStableUnknown(ref carry, stableUnknown, remainingCols, out var propagateReason))
#else
						if (!TryAdvanceStableUnknown(ref carry, stableUnknown, remaining))
#endif
						{
#if DETAILED_LOG
							reason = propagateReason;
#endif
							return false;
						}

						startColumn += remaining;
						windowColumn = startColumn;
					}

					// Reset window to unknown (no known bits) because we skipped updating it across columns.
					Array.Clear(aOneWin, 0, qWordCount);
					Array.Clear(aZeroWin, 0, qWordCount);

					continue;
				}
			}

			// Mark out-of-range (negative) a indices as zero in this column.
			// if (startColumn < maxOffset)
			// {
			// 	int firstOut = startColumn + 1;
			// 	for (int t = firstOut; t <= maxOffset; t++)
			// 	{
			// 		int w = t >> 6;
			// 		int b = t & 63;
			// 		aZeroWin[w] |= 1UL << b;
			// 	}
			// 	if (qWordCount > 0) aZeroWin[qWordCount - 1] &= lastWordMask;
			// }

			ComputeForcedRemainingAndClean_Prefix(qMaskWords, aOneWin, aZeroWin, qWordsEff, qLastEff, out int forced, out remaining);

			// If exactly one unknown contribution among q=1 positions AND carry parity is fixed,
			// we can force the corresponding a bit to satisfy the required result bit parity.
			if (remaining == 1 && ((carry.Min ^ carry.Max) & 1L) == 0)
			{
				if (TryFindSingleUnknownInQ_Prefix(qMaskWords, aOneWin, aZeroWin, qWordsEff, qLastEff, out step))
				{
					aIndex = startColumn - step;

					// Required result bit for M_p in columns [0..p-1] is 1.
					const int requiredBit = 1;

					// We need (carryIn + forced + x) % 2 == requiredBit, where x ∈ {0,1}.
					int b = (int)(carry.Min & 1L);   // fixed parity because Min/Max same parity
													 // We're reusing qWordsEff as w from here to limit registry pressure
					qWordsEff = (requiredBit ^ b ^ (forced & 1)) & 1;
					bool needOne = qWordsEff != 0;

					// Check contradiction with already-known value.
					state = GetAKnownStateRowAware(aIndex, segmentBase, windowSize, knownOne0, knownZero0, knownOne1, knownZero1);
					if (needOne)
					{
						if (state == 2)
						{
#if DETAILED_LOG
							reason = ContradictionReason.ParityUnreachable;
#endif
							return false;
						}

						SetAKnownRowAware(aIndex, true, segmentBase, windowSize, knownOne0, knownZero0, knownOne1, knownZero1);

						// Reflect in window at offset t
						qWordsEff = step >> 6;
						b = step & 63;
						aOneWin[qWordsEff] |= 1UL << b;
						forced++;
					}
					else
					{
						if (state == 1)
						{
#if DETAILED_LOG
							reason = ContradictionReason.ParityUnreachable;
#endif
							return false;
						}

						SetAKnownRowAware(aIndex, false, segmentBase, windowSize, knownOne0, knownZero0, knownOne1, knownZero1);

						qWordsEff = step >> 6;
						b = step & 63;
						aZeroWin[qWordsEff] |= 1UL << b;
					}

					remaining = 0;
					if (aIndex > maxKnownA)
					{
						maxKnownA = aIndex;
					}
				}
			}

			// Now propagate carry using forced and possible (=forced+remaining).
			remaining += forced;
			if (!TryPropagateCarry(ref carry, forced, remaining, 1))
			{
#if DETAILED_LOG
				reason = ContradictionReason.ParityUnreachable;
#endif
				return false;
			}

			// Advance to next column: shift window, then insert status of a_{startColumn+1} into bit0.
			// remaining is nextA from here. We're reusing variable to limit registry pressure.
			remaining = startColumn + 1;
			ShiftLeft1InPlace(aOneWin, aZeroWin, qWordCount, lastWordMask);

			// Insert bit0 for nextA
			if (remaining > maxAllowedA)
			{
				state = 2;
			}
			else
			{
				state = GetAKnownStateRowAware(remaining, segmentBase, windowSize, knownOne0, knownZero0, knownOne1, knownZero1);
			}

			if (state == 1)
			{
				aOneWin[0] |= 1UL;
			}
			else if (state == 2)
			{
				aZeroWin[0] |= 1UL;
			}

			startColumn++;
			windowColumn = startColumn;
#if DETAILED_LOG
			_statScalarCols++;
#endif
		}

#if DETAILED_LOG
		reason = ContradictionReason.None;
#endif
		return true;
	}

	private static bool TryProcessBottomUpBlock(
		ulong prime,
		ReadOnlySpan<int> qOneOffsets,
		in int[] runStarts,
		in int[] runLengths,
		int runCount,
		int maxAllowedA,
		int startColumn,
		int columnCount,
		ref CarryRange carry,
		ref int maxKnownA,
		ref int segmentBase,
		ref int runStartIdx,
		ref int runEndIdx,
		ulong[] knownOne0,
		ulong[] knownOne1,
		ulong[] knownZero0,
		ulong[] knownZero1,
		int windowSize,
		int topIndex0,
		int topIndex1,
		ref bool top0KnownOne,
		ref bool top1KnownOne,
		ref bool top0KnownZero,
		ref bool top1KnownZero,
		ref int zeroTailStart
#if DETAILED_LOG
		, out ContradictionReason reason
#endif
	)
	{
		int endColumn = startColumn + columnCount;
		int qOneOffsetsLength = qOneOffsets.Length;
		int maxOffset = qOneOffsets[qOneOffsetsLength - 1];
		int stableUnknown = qOneOffsetsLength;
		int bound, runStart, runEnd;
		ulong[] temp;
		// Span<ulong> temp;
		while (startColumn < endColumn)
		{
			int remaining;
			if (startColumn >= maxOffset && startColumn <= maxAllowedA && maxKnownA < startColumn - maxOffset)
			{
				remaining = endColumn - 1;
				if (remaining > maxAllowedA)
				{
					remaining = maxAllowedA;
				}

				remaining = remaining - startColumn + 1;
				if (remaining >= 16)
				{
					runStartIdx = 0;
					runEndIdx = runCount;
					while (remaining >= TailCarryBatchColumns)
					{
#if DETAILED_LOG
						if (!TryAdvanceStableUnknown(ref carry, stableUnknown, TailCarryBatchColumns, out var propagateReason))
#else
						if (!TryAdvanceStableUnknown(ref carry, stableUnknown, TailCarryBatchColumns))
#endif
						{
#if DETAILED_LOG
							reason = propagateReason;
#endif
							return false;
						}

						startColumn += TailCarryBatchColumns;
						remaining -= TailCarryBatchColumns;
					}

					if (remaining > 0)
					{
#if DETAILED_LOG
						if (!TryAdvanceStableUnknown(ref carry, stableUnknown, remaining, out var propagateReason))
#else
						if (!TryAdvanceStableUnknown(ref carry, stableUnknown, remaining))
#endif
						{
#if DETAILED_LOG
							reason = propagateReason;
#endif
							return false;
						}

						remaining = 0;
					}

					bound = (startColumn / windowSize) * windowSize;
					if (bound != segmentBase)
					{
						if (bound == segmentBase + windowSize)
						{
							// rollover forward by one segment: swap and clear only the new "current" segment
							temp = knownOne0; knownOne0 = knownOne1; knownOne1 = temp;
							temp = knownZero0; knownZero0 = knownZero1; knownZero1 = temp;

							Array.Clear(knownOne0);
							Array.Clear(knownZero0);
						}
						else
						{
							// jumped more than one segment: we genuinely lost locality; clear both
							Array.Clear(knownOne0);
							Array.Clear(knownOne1);
							Array.Clear(knownZero0);
							Array.Clear(knownZero1);
						}

						segmentBase = bound;
					}

					continue;
				}
			}

			int forced = 0;
			remaining = 0;
			int chosenIndex = -1;
			// bool topChanged = false;
			bound = (startColumn / windowSize) * windowSize;
			if (bound != segmentBase)
			{
				if (bound == segmentBase + windowSize)
				{
					temp = knownOne0;
					knownOne0 = knownOne1;
					knownOne1 = temp;

					temp = knownZero0;
					knownZero0 = knownZero1;
					knownZero1 = temp;

					// knownOne0.Clear();
					// knownZero0.Clear();
					Array.Clear(knownOne0);
					Array.Clear(knownZero0);
				}
				else
				{
					// knownOne0.Clear();
					// knownOne1.Clear();
					// knownZero0.Clear();
					// knownZero1.Clear();
					Array.Clear(knownOne0);
					Array.Clear(knownOne1);
					Array.Clear(knownZero0);
					Array.Clear(knownZero1);
				}

				segmentBase = bound;
			}

			bound = startColumn - maxAllowedA;
			while (runEndIdx < runCount && runStarts[runEndIdx] <= startColumn)
			{
				runEndIdx++;
			}

			while (runStartIdx < runEndIdx)
			{
				runStart = runStarts[runStartIdx];
				runEnd = runStart + runLengths[runStartIdx] - 1;
				if (runEnd < bound)
				{
					runStartIdx++;
					continue;
				}

				break;
			}

			int known0Len, runIndex;
			int seg0Start = segmentBase;
			for (runIndex = runStartIdx; runIndex < runEndIdx; runIndex++)
			{
				runStart = runStarts[runIndex];
				if (runStart > startColumn)
				{
					break;
				}

				runEnd = runStart + runLengths[runIndex] - 1;
				if (runEnd > startColumn)
				{
					runEnd = startColumn;
				}

				if (runStart < bound)
				{
					runStart = bound;
				}

				if (runStart > runEnd)
				{
					continue;
				}

				int aStart = startColumn - runEnd;
				runEnd = startColumn - runStart;
				if (aStart > maxAllowedA)
				{
					continue;
				}

				if (runEnd > maxAllowedA)
				{
					runEnd = maxAllowedA;
				}

				int rangeLength = runEnd - aStart + 1;
				if (rangeLength <= 0)
				{
					continue;
				}

				int seg0End = seg0Start + windowSize - 1;
				int seg1Start = seg0Start - windowSize;
				int seg1End = seg0Start - 1;

				int overlapStart = aStart > seg0Start ? aStart : seg0Start;
				int overlap0End = runEnd < seg0End ? runEnd : seg0End;
				// seg1End is now overlap1End to limit registry pressure
				seg1End = runEnd < seg1End ? runEnd : seg1End;

				known0Len = 0;
				int ones;
				if (overlapStart <= overlap0End)
				{
					runStart = overlapStart - seg0Start;
					// overlap0Start is now len to limit registry pressure
					overlapStart = overlap0End - overlapStart + 1;
					ones = CountBitsInRange(knownOne0, runStart, overlapStart);
					// overlap0End is now zeros to limit registry pressure
					overlap0End = CountBitsInRange(knownZero0, runStart, overlapStart);
					forced += ones;
					remaining += overlapStart - ones - overlap0End;
					known0Len = overlapStart;

					if (chosenIndex == -1 && overlapStart > ones + overlap0End && TryFindFirstUnknown(knownOne0, knownZero0, runStart, overlapStart, out int idx))
					{
						chosenIndex = seg0Start + idx;
					}
				}

				// overlap0End is now knownTotal to limit registry pressure
				overlap0End = 0;
				overlapStart = aStart > seg1Start ? aStart : seg1Start;
				if (overlapStart <= seg1End)
				{
					runStart = overlapStart - seg1Start;
					// overlap1Start is now len to limit registry pressure
					overlapStart = seg1End - overlapStart + 1;
					ones = CountBitsInRange(knownOne1, runStart, overlapStart);
					// seg1End is now zeros to limit registry pressure
					seg1End = CountBitsInRange(knownZero1, runStart, overlapStart);
					forced += ones;
					remaining += overlapStart - ones - seg1End;
					overlap0End = overlapStart;

					if (chosenIndex == -1 && overlapStart > ones + seg1End && TryFindFirstUnknown(knownOne1, knownZero1, runStart, overlapStart, out int idx))
					{
						chosenIndex = seg1Start + idx;
					}
				}

				overlap0End = known0Len + overlap0End;
				if (overlap0End < rangeLength)
				{
					remaining += rangeLength - overlap0End;
				}

				if (zeroTailStart <= runEnd)
				{
					// overlap0End is now tailStart to limit registry pressure
					overlap0End = aStart > zeroTailStart ? aStart : zeroTailStart;
					if (overlap0End <= runEnd)
					{
						rangeLength = runEnd - overlap0End + 1;
						if (rangeLength > 0)
						{
							remaining -= rangeLength;
							if (remaining < 0)
							{
								remaining = 0;
							}
						}
					}
				}
			}

			long minSum = carry.Min;
			long maxSum = carry.Max;
			ulong mask;
			if (remaining == 1 && ((minSum ^ maxSum) & 1) == 0)
			{
				// known0Len is now requiredUnknownParity to limit registry pressure
				known0Len = 1 ^ ((int)minSum & 1) ^ (forced & 1);
				if (known0Len != 0)
				{
					if (chosenIndex >= 0)
					{
						if (chosenIndex >= seg0Start)
						{
							bound = chosenIndex - seg0Start;
							// runIndex is now word to limit registry pressure
							runIndex = bound >> 6;
							mask = 1UL << (bound & 63);
							knownOne0[runIndex] |= mask;
							knownZero0[runIndex] &= ~mask;
						}
						else
						{
							bound = chosenIndex - (seg0Start - windowSize);
							// runIndex is now word to limit registry pressure
							runIndex = bound >> 6;
							mask = 1UL << (bound & 63);
							knownOne1[runIndex] |= mask;
							knownZero1[runIndex] &= ~mask;
						}

						forced++;
						remaining = 0;
						if (chosenIndex > maxKnownA)
						{
							maxKnownA = chosenIndex;
						}
						if (chosenIndex == topIndex0)
						{
							if (top0KnownZero)
							{
#if DETAILED_LOG
								reason = ContradictionReason.ParityUnreachable;
#endif
								return false;
							}

							top0KnownOne = true;
							// topChanged = true;
						}
						else if (chosenIndex == topIndex1)
						{
							if (top1KnownZero)
							{
#if DETAILED_LOG
								reason = ContradictionReason.ParityUnreachable;
#endif
								return false;
							}

							top1KnownOne = true;
							// topChanged = true;
						}
					}
				}
				else
				{
					// if (_debugEnabled && _parityZeroLogCount < 8)
					// {
					//     _parityZeroLogCount++;
					//     Console.WriteLine($"[bit-contradiction] unknown==1 parity forces zero at column={column} carry=[{minSum},{maxSum}] forced={forced} chosenIndex={chosenIndex}");
					// }

					// Treat the single unknown as 0 for parity.
					remaining = 0;

					if (chosenIndex == topIndex0)
					{
						if (top0KnownOne)
						{
#if DETAILED_LOG
							reason = ContradictionReason.ParityUnreachable;
#endif
							return false;
						}

						top0KnownZero = true;
					}
					else if (chosenIndex == topIndex1)
					{
						if (top1KnownOne)
						{
#if DETAILED_LOG
							reason = ContradictionReason.ParityUnreachable;
#endif
							return false;
						}

						top1KnownZero = true;
					}

					UpdateZeroTailFromTop(ref zeroTailStart, topIndex0, topIndex1, top0KnownZero, top1KnownZero);
				}
			}

			// 			if (topChanged)
			// 			{
			// 				var qHash = ComputeQOffsetsHash(qOneOffsets);
			// 				if (!TryDynamicModularRangePrune(qOneOffsets, qHash, prime, maxAllowedA, top0KnownOne, top0KnownZero, top1KnownOne, zeroTailStart))
			// 				{
			// #if DETAILED_LOG
			// 								reason = ContradictionReason.ParityUnreachable;
			// #endif
			// 					return false;
			// 				}
			// 			}

			if ((top0KnownZero && top0KnownOne) || (top1KnownZero && top1KnownOne))
			{
#if DETAILED_LOG
				reason = ContradictionReason.ParityUnreachable;
#endif
				return false;
			}

			if (topIndex1 < 0)
			{
				if (top0KnownZero)
				{
#if DETAILED_LOG
					reason = ContradictionReason.ParityUnreachable;
#endif
					return false;
				}
			}
			else if (top0KnownZero && top1KnownZero)
			{
#if DETAILED_LOG
				reason = ContradictionReason.ParityUnreachable;
#endif
				return false;
			}

			if (remaining == 0)
			{
				// known0Len is now requiredCarryParity to limit registry pressure
				known0Len = 1 ^ (forced & 1);
				long carryMin = AlignUpToParity(carry.Min, known0Len);
				long carryMax = AlignDownToParity(carry.Max, known0Len);
				if (carryMin > carryMax)
				{
#if DETAILED_LOG
					reason = ContradictionReason.ParityUnreachable;
#endif
					return false;
				}

				carry = new CarryRange(carryMin, carryMax);
				minSum = carryMin;
				maxSum = carryMax;
			}

			minSum += forced;
			maxSum += forced + remaining;
			const int requiredParity = 1;

			if (minSum == maxSum)
			{
				if ((minSum & 1) != requiredParity)
				{
#if DETAILED_LOG
					reason = ContradictionReason.ParityUnreachable;
#endif
					return false;
				}

				minSum = (minSum - requiredParity) >> 1;
				carry = CarryRange.Single(minSum);
			}
			else
			{
				if ((minSum & 1) != requiredParity && remaining == 0)
				{
#if DETAILED_LOG
					reason = ContradictionReason.ParityUnreachable;
#endif
					return false;
				}

				if (!TryPropagateCarry(
						ref carry,
						forced,
						forced + remaining,
						requiredParity))
				{
#if DETAILED_LOG
					reason = ContradictionReason.ParityUnreachable;
#endif
					return false;
				}
			}

			startColumn++;
#if DETAILED_LOG
			_statScalarCols++;
#endif
		}

#if DETAILED_LOG
		reason = ContradictionReason.None;
#endif
		return true;
	}

#if DETAILED_LOG
	private static bool TryProcessTopDownBlock(
		ReadOnlySpan<int> qOneOffsets,
		int qOneOffsetsLength,
		int maxAllowedA,
		int startHighColumn,
		int columnCount,
		CarryRange carryOut,
		ref int windowStart,
		ref int windowEnd,
		out CarryRange nextCarryOut,
		out TopDownPruneFailure? failure)
#else
	private static bool TryProcessTopDownBlock(
		ReadOnlySpan<int> qOneOffsets,
		int qOneOffsetsLength,
		int maxAllowedA,
		int startHighColumn,
		int columnCount,
		CarryRange carryOut,
		ref int windowStart,
		ref int windowEnd,
		out CarryRange nextCarryOut)
#endif
	{
#if DETAILED_LOG
		failure = null;
#endif
		long carryOutMin = carryOut.Min,
			 carryOutMax = carryOut.Max;
		int value;
		int endColumn = startHighColumn - columnCount + 1;

		int column = startHighColumn;
		while (column >= endColumn)
		{
			// value is used as high column index
			value = column;
			while (windowEnd > 0 && qOneOffsets[windowEnd - 1] > value)
			{
				windowEnd--;
			}

			// value is used as low column index
			value = column - maxAllowedA;
			if (value < 0)
			{
				value = 0;
			}

			while (windowStart > 0 && qOneOffsets[windowStart - 1] >= value)
			{
				windowStart--;
			}

			if (windowStart > windowEnd)
			{
				windowStart = windowEnd;
			}

			// value is unknown bits count here
			value = windowEnd - windowStart;
			if (value == 0)
			{
				int lastEnd = windowEnd > 0 ? qOneOffsets[windowEnd - 1] : int.MinValue;
				int lastStart = windowStart > 0 ? qOneOffsets[windowStart - 1] + maxAllowedA : int.MinValue;
				int chunkEnd = lastEnd > lastStart ? lastEnd : lastStart;
				if (chunkEnd < endColumn)
				{
					chunkEnd = endColumn;
				}

				int steps = column - chunkEnd + 1;
				if (steps > 1)
				{
					if (!AdvanceTopDownUnknown0(ref carryOutMin, ref carryOutMax, steps))
					{
#if DETAILED_LOG
						failure = new TopDownPruneFailure(column, carryOutMin, carryOutMax, value);
#endif
						nextCarryOut = default;
						return false;
					}

					column = chunkEnd - 1;
					continue;
				}
			}
			else if (value == 1)
			{
				int lastEnd = windowEnd > 0 ? qOneOffsets[windowEnd - 1] : int.MinValue;
				int lastStart = windowStart > 0 ? qOneOffsets[windowStart - 1] + maxAllowedA : int.MinValue;
				int chunkEnd = lastEnd > lastStart ? lastEnd : lastStart;
				if (chunkEnd < endColumn)
				{
					chunkEnd = endColumn;
				}

				int steps = (int)(column - chunkEnd + 1);
				if (steps > 1)
				{
					if (!AdvanceTopDownUnknown1(ref carryOutMin, ref carryOutMax, steps))
					{
#if DETAILED_LOG
						failure = new TopDownPruneFailure(column, carryOutMin, carryOutMax, value);
#endif
						nextCarryOut = default;
						return false;
					}

					column = chunkEnd - 1;
					continue;
				}
			}

			// carryOutMin now becomes nextCarryMin to limit registry pressure
			carryOutMin = (carryOutMin << 1) + 1 - value;
			// carryOutMax now becomes nextCarryMax to limit registry pressure
			carryOutMax = (carryOutMax << 1) + 1;
			if (carryOutMin < 0)
			{
				carryOutMin = 0;
			}

			if (carryOutMax < 0)
			{
				carryOutMax = 0;
			}

			if (carryOutMin > carryOutMax)
			{
#if DETAILED_LOG
				failure = new TopDownPruneFailure(column, carryOutMin, carryOutMax, value);
#endif
				nextCarryOut = default;
				return false;
			}

			column--;
		}

		nextCarryOut = new CarryRange(carryOutMin, carryOutMax);
		return true;
	}

	private static long AlignUpToParity(long value, int parity) => (value & 1L) == parity ? value : value + 1L;
	private static long AlignDownToParity(long value, int parity) => (value & 1L) == parity ? value : value - 1L;

#if DETAILED_LOG
	private static bool TryTopDownCarryPrune(
		ReadOnlySpan<int> qOneOffsets,
		int qOneOffsetsLength,
		int pLong,
		int maxAllowedA,
		out TopDownPruneFailure? failure)
#else
	private static bool TryTopDownCarryPrune(
		ReadOnlySpan<int> qOneOffsets,
		int qOneOffsetsLength,
		int pLong,
		int maxAllowedA)
#endif
	{
#if DETAILED_LOG
		failure = null;
#endif
		int windowStart = qOneOffsetsLength;
		int windowEnd = qOneOffsetsLength;
		long carryOutMin = 0,
			 carryOutMax = 0;
		int value;

		int column = pLong - 1;
		while (column >= 0)
		{
			// value is used as high column index
			value = column;
			while (windowEnd > 0 && qOneOffsets[windowEnd - 1] > value)
			{
				windowEnd--;
			}

			// value is used as low column index
			value = column - maxAllowedA;
			if (value < 0)
			{
				value = 0;
			}

			while (windowStart > 0 && qOneOffsets[windowStart - 1] >= value)
			{
				windowStart--;
			}

			if (windowStart > windowEnd)
			{
				windowStart = windowEnd;
			}

			// value is used as unknown bits count here
			value = windowEnd - windowStart;
			if (value == 0)
			{
				int lastEnd = windowEnd > 0 ? qOneOffsets[windowEnd - 1] : int.MinValue;
				int lastStart = windowStart > 0 ? qOneOffsets[windowStart - 1] + maxAllowedA : int.MinValue;
				int chunkEnd = lastEnd > lastStart ? lastEnd : lastStart;
				if (chunkEnd < 0)
				{
					chunkEnd = 0;
				}

				int steps = (int)(column - chunkEnd + 1);
				if (steps > 1)
				{
					if (!AdvanceTopDownUnknown0(ref carryOutMin, ref carryOutMax, steps))
					{
#if DETAILED_LOG
						failure = new TopDownPruneFailure(column, carryOutMin, carryOutMax, value);
#endif
						return false;
					}

					column = chunkEnd - 1;
					continue;
				}
			}
			else if (value == 1)
			{
				int lastEnd = windowEnd > 0 ? qOneOffsets[windowEnd - 1] : int.MinValue;
				int lastStart = windowStart > 0 ? qOneOffsets[windowStart - 1] + maxAllowedA : int.MinValue;
				int chunkEnd = lastEnd > lastStart ? lastEnd : lastStart;
				if (chunkEnd < 0)
				{
					chunkEnd = 0;
				}

				int steps = (int)(column - chunkEnd + 1);
				if (steps > 1)
				{
					if (!AdvanceTopDownUnknown1(ref carryOutMin, ref carryOutMax, steps))
					{
#if DETAILED_LOG
						failure = new TopDownPruneFailure(column, carryOutMin, carryOutMax, value);
#endif
						return false;
					}

					column = chunkEnd - 1;
					continue;
				}
			}

			// carryOutMin now becomes nextCarryMin to limit registry pressure
			carryOutMin = (carryOutMin << 1) + 1 - value;
			// carryOutMax now becomes nextCarryMax to limit registry pressure
			carryOutMax = (carryOutMax << 1) + 1;
			if (carryOutMin < 0)
			{
				carryOutMin = 0;
			}

			if (carryOutMax < 0)
			{
				carryOutMax = 0;
			}

			if (carryOutMin > carryOutMax)
			{
#if DETAILED_LOG
				failure = new TopDownPruneFailure(column, carryOutMin, carryOutMax, value);
#endif
				return false;
			}

			column--;
		}

		if (carryOutMin > 0)
		{
#if DETAILED_LOG
			failure = new TopDownPruneFailure(0, carryOutMin, carryOutMax, 0);
#endif
			return false;
		}

		return true;
	}

#if DETAILED_LOG
	private static bool TryTopDownBorrowPrefilter(
		ReadOnlySpan<int> qOneOffsets,
		int qOneOffsetsLength,
		int pLong,
		int maxAllowedA,
		out TopDownPruneFailure? failure)
#else
	private static bool TryTopDownBorrowPrefilter(
		ReadOnlySpan<int> qOneOffsets,
		int qOneOffsetsLength,
		int pLong,
		int maxAllowedA)
#endif
	{
#if DETAILED_LOG
		failure = null;
#endif
		CarryRange borrow = CarryRange.Zero;
		int windowStart = qOneOffsetsLength;
		int windowEnd = qOneOffsetsLength;

		int column = pLong - 1;
		int endColumn = column - (TopDownBorrowColumns - 1);
		if (endColumn < 0)
		{
			endColumn = 0;
		}

		while (column >= endColumn)
		{
			int value = column;
			while (windowEnd > 0 && qOneOffsets[windowEnd - 1] > value)
			{
				windowEnd--;
			}

			value = column - maxAllowedA;
			if (value < 0)
			{
				value = 0;
			}

			while (windowStart > 0 && qOneOffsets[windowStart - 1] >= value)
			{
				windowStart--;
			}

			if (windowStart > windowEnd)
			{
				windowStart = windowEnd;
			}

			int unknown = windowEnd - windowStart;
			if (!TryPropagateCarry(ref borrow, 0, unknown, 1))
			{
#if DETAILED_LOG
				failure = new TopDownPruneFailure(column, borrow.Min, borrow.Max, unknown);
#endif
				return false;
			}

			column--;
		}

		return true;
	}

	private static bool ShouldRunTopDownBorrowPrefilter(
		ReadOnlySpan<int> qOneOffsets,
		int qOneOffsetsLength,
		long pLong,
		long maxAllowedA)
	{
		if (qOneOffsetsLength < TopDownBorrowMinUnknown)
		{
			return false;
		}

		long column = pLong - 1;
		long start = column - maxAllowedA;
		if (start < 0)
		{
			start = 0;
		}

		FindBounds(qOneOffsets, (int)start, (int)column, out int lower, out int upper);
		return upper - lower >= TopDownBorrowMinUnknown;
	}

	private static bool AdvanceTopDownUnknown1(ref long carryOutMin, ref long carryOutMax, int steps)
	{
		while (steps > 0)
		{
			int step = steps > 30 ? 30 : steps;
			int pow = 1 << step;

			if (carryOutMin > (int.MaxValue >> step) || carryOutMax > ((int.MaxValue - (pow - 1)) >> step))
			{
				carryOutMin = 0;
				carryOutMax = int.MaxValue;
				return true;
			}

			carryOutMin <<= step;
			carryOutMax = (carryOutMax << step) + (pow - 1);
			steps -= step;
		}

		return true;
	}

	private static bool AdvanceTopDownUnknown0(ref long carryOutMin, ref long carryOutMax, int steps)
	{
		while (steps > 0)
		{
			int step = steps > 30 ? 30 : steps;
			int add = (1 << step) - 1;
			if (carryOutMin > ((int.MaxValue - add) >> step) || carryOutMax > ((int.MaxValue - add) >> step))
			{
				carryOutMin = 0;
				carryOutMax = int.MaxValue;
				return true;
			}

			carryOutMin = (carryOutMin << step) + add;
			carryOutMax = (carryOutMax << step) + add;
			steps -= step;
		}

		return true;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static int CountBitsInRange(in ulong[] bits, int start, int length)
	{
		if (length <= 0)
		{
			return 0;
		}

		length += start - 1;
		int firstWord = start >> 6;
		int lastWord = length >> 6;
		start &= 63;
		length &= 63;
		ulong mask;

		if (firstWord == lastWord)
		{
			mask = (length == 63 ? ulong.MaxValue : ((1UL << (length + 1)) - 1UL)) & (ulong.MaxValue << start);
			return BitOperations.PopCount(bits[firstWord] & mask);
		}

		mask = ulong.MaxValue << start;
		// start is count from here. We're reusing the variable to limit registry pressure.
		start = BitOperations.PopCount(bits[firstWord] & mask);

		firstWord++;
		for (; firstWord < lastWord; firstWord++)
		{
			start += BitOperations.PopCount(bits[firstWord]);
		}

		mask = length == 63 ? ulong.MaxValue : ((1UL << (length + 1)) - 1UL);
		start += BitOperations.PopCount(bits[lastWord] & mask);
		return start;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool TryFindFirstUnknown(in ulong[] knownOne, in ulong[] knownZero, int start, int length, out int index)
	{
		if (length <= 0)
		{
			index = -1;
			return false;
		}

		int end = start + length - 1;
		int firstWord = start >> 6;
		int lastWord = end >> 6;
		int startBit = start & 63;
		int endBit = end & 63;

		ulong firstMask = ulong.MaxValue << startBit;
		ulong lastMask = endBit == 63 ? ulong.MaxValue : ((1UL << (endBit + 1)) - 1UL);

		for (int word = firstWord; word <= lastWord; word++)
		{
			ulong mask = word == firstWord ? firstMask : ulong.MaxValue;
			if (word == lastWord)
			{
				mask &= lastMask;
			}

			ulong known = (knownOne[word] | knownZero[word]) & mask;
			ulong unknown = mask & ~known;
			if (unknown != 0)
			{
				int bit = BitOperations.TrailingZeroCount(unknown);
				index = (word << 6) + bit;
				return true;
			}
		}

		index = -1;
		return false;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void UpdateZeroTailFromTop(ref int zeroTailStart, int topIndex0, int topIndex1, bool top0KnownZero, bool top1KnownZero)
	{
		if (!top0KnownZero)
		{
			return;
		}

		if (zeroTailStart > topIndex0)
		{
			zeroTailStart = topIndex0;
		}

		if (top1KnownZero && topIndex1 == topIndex0 - 1 && zeroTailStart > topIndex1)
		{
			zeroTailStart = topIndex1;
		}
	}

#if DETAILED_LOG
	private static bool TryRunHighBitAndBorrowPrefiltersCombined(
		ReadOnlySpan<int> qOneOffsets,
		int qOneOffsetsLength,
		int pLong,
		int maxAllowedA,
		out TopDownPruneFailure? failure)
#else
	private static bool TryRunHighBitAndBorrowPrefiltersCombined(
		ReadOnlySpan<int> qOneOffsets,
		int qOneOffsetsLength,
		int pLong,
		int maxAllowedA)
#endif
	{
#if DETAILED_LOG
		failure = null;
#endif
		int column = pLong - 1,
			columnPlusOne = column + 1;

		int highBitColumns = HighBitCarryPrefilterColumns;
		if (highBitColumns > columnPlusOne)
		{
			highBitColumns = columnPlusOne;
		}

		int borrowColumns = 0;
		bool runBorrowPrefilter = qOneOffsetsLength >= TopDownBorrowMinUnknown;
		if (runBorrowPrefilter)
		{
			borrowColumns = TopDownBorrowColumns;
			if (borrowColumns > columnPlusOne)
			{
				borrowColumns = columnPlusOne;
			}
		}

		int sequentialColumns = highBitColumns > borrowColumns ? highBitColumns : borrowColumns;
		// Given the above, sequentialColumns will never <= 0, because highBitColumns is always > 0.
		// if (sequentialColumns <= 0)
		// {
		// 	return true;
		// }

		Span<long> lo0 = stackalloc long[64];
		Span<long> hi0 = stackalloc long[64];
		Span<long> lo1 = stackalloc long[64];
		Span<long> hi1 = stackalloc long[64];

		Span<long> outLo = lo0, outHi = hi0;
		Span<long> inLo = lo1, inHi = hi1;

		int outCount = 1;
		outLo[0] = 0;
		outHi[0] = 0;

		int inCount = 0;
		bool flip = false;
		CarryRange borrow = CarryRange.Zero;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static void AddInterval(ref int count, Span<long> loArr, Span<long> hiArr, long lo, long hi)
		{
			if (hi < lo) return;

			if (lo < 0) lo = 0;
			if (hi < 0) return;

			if (count == 0)
			{
				loArr[0] = lo;
				hiArr[0] = hi;
				count = 1;
				return;
			}

			long lastHi = hiArr[count - 1];
			if (lo <= lastHi + 1)
			{
				if (hi > lastHi)
					hiArr[count - 1] = hi;
				return;
			}

			if ((uint)count >= (uint)loArr.Length)
			{
				count = 0;
				return;
			}

			loArr[count] = lo;
			hiArr[count] = hi;
			count++;
		}

		int step = 0;
		int upperT;
		int lowerT = column - maxAllowedA;
		if (lowerT < 0)
		{
			lowerT = 0;
		}

		FindBounds(qOneOffsets, lowerT, column, out lowerT, out upperT);
		int n = upperT - lowerT;
		if (n < 0)
		{
			return false;
		}

		if (runBorrowPrefilter && n < TopDownBorrowMinUnknown)
		{
			runBorrowPrefilter = false;
			if (highBitColumns < sequentialColumns)
			{
				sequentialColumns = highBitColumns;
			}
		}

		while (true)
		{
			if (step < highBitColumns)
			{
				// Build carryIn set into inLo/inHi
				inCount = 0;

				for (int i = 0; i < outCount; i++)
				{
					long rMin = outLo[i];
					long rMax = outHi[i];

					if (rMax < 0) continue;
					if (rMin < 0) rMin = 0;
					if (rMax < rMin) continue;

					const long TwicePlusOneOverflowLimit = (long.MaxValue - 1) >> 1;
					if (rMin > TwicePlusOneOverflowLimit || rMax > TwicePlusOneOverflowLimit)
					{
						return true;
					}

					rMin = (rMin << 1) + 1;
					rMax = (rMax << 1) + 1;

					if (n <= 0)
					{
						AddInterval(ref inCount, inLo, inHi, rMin, rMax);
						if (inCount == 0) return true;
						continue;
					}

					if (n == 1)
					{
						AddInterval(ref inCount, inLo, inHi, rMin - 1, rMax);
						if (inCount == 0) return true;
						continue;
					}

					if (rMax <= n)
					{
						AddInterval(ref inCount, inLo, inHi, 0, rMax);
						if (inCount == 0) return true;
					}
					else if (rMin > n)
					{
						AddInterval(ref inCount, inLo, inHi, rMin - n, rMax);
						if (inCount == 0) return true;
					}
					else
					{
						AddInterval(ref inCount, inLo, inHi, 0, rMax);
						if (inCount == 0) return true;
					}
				}

				if (inCount <= 0)
				{
					return false;
				}

				// Swap buffers for next step.
				outCount = inCount;
				flip = !flip;
				if (!flip)
				{
					outLo = lo0; outHi = hi0;
					inLo = lo1; inHi = hi1;
				}
				else
				{
					outLo = lo1; outHi = hi1;
					inLo = lo0; inHi = hi0;
				}
			}

			if (runBorrowPrefilter && step < borrowColumns)
			{
				if (!TryPropagateCarry(ref borrow, 0, n, 1))
				{
#if DETAILED_LOG
					failure = new TopDownPruneFailure(column, borrow.Min, borrow.Max, n);
#endif
					return false;
				}
			}

			step++;
			column--;
			if (step >= sequentialColumns || column < 0)
			{
				break;
			}

			lowerT = column - maxAllowedA;
			if (lowerT < 0)
			{
				lowerT = 0;
			}

			FindBounds(qOneOffsets, lowerT, column, out lowerT, out upperT);
			n = upperT - lowerT;
			if (n < 0)
			{
				return false;
			}
		}

		return true;
	}
}
