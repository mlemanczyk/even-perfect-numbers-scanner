using System;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core.ByDivisor;

public readonly struct ColumnBounds(int forcedOnes, int possibleOnes)
{
	public readonly int ForcedOnes = forcedOnes;
	public readonly int PossibleOnes = possibleOnes;
}

public readonly struct CarryRange(long min, long max)
{
	public readonly long Min = min;
	public readonly long Max = max;

	public static CarryRange Single(long value) => new(value, value);
	public static readonly CarryRange Zero = Single(0);
}

public enum ContradictionReason
{
	None,
	ParityUnreachable,
	TruncatedLength,
	TailNotZero,
}

/// <summary>
/// Helper utilities for the BitContradiction divisor search. These methods implement the
/// interval-based carry propagation described in the BitContradictions plan document.
/// The solver itself will build on top of these primitives.
/// </summary>
internal static class BitContradictionSolverWithQMultiplier
{
	private const int PingPongBlockColumns = 64;
	private const int TailCarryBatchColumns = 1024;
	private const int TailCarryBatchRepeats = 1;
	private const int TopDownCacheCapacity = 16;

#if DETAILED_LOG
    public readonly struct TopDownPruneFailure(int column, int carryMin, int carryMax, int unknown)
    {
        public readonly int Column = column;
        public readonly int CarryMin = carryMin;
        public readonly int CarryMax = carryMax;
        public readonly int Unknown = unknown;
    }

    private readonly struct TopDownCacheValue(bool success, TopDownPruneFailure? failure)
    {
        public readonly bool Success = success;
        public readonly TopDownPruneFailure? Failure = failure;
    }

    internal readonly struct BottomUpFailure(int column, int carryMin, int carryMax, int forced, int unknown)
    {
        public readonly int Column = column;
        public readonly int CarryMin = carryMin;
        public readonly int CarryMax = carryMax;
        public readonly int Forced = forced;
        public readonly int Unknown = unknown;
    }

    [ThreadStatic]
    private static TopDownPruneFailure? _lastTopDownFailure;
    [ThreadStatic]
    private static BottomUpFailure? _lastBottomUpFailure;
    [ThreadStatic]
    private static Dictionary<TopDownCacheKey, TopDownCacheValue>? _topDownCache;

    internal static TopDownPruneFailure? LastTopDownFailure => _lastTopDownFailure;
    internal static BottomUpFailure? LastBottomUpFailure => _lastBottomUpFailure;
#else
	[ThreadStatic]
	private static Dictionary<TopDownCacheKey, bool>? _topDownCache;
#endif

	[ThreadStatic]
	private static List<TopDownCacheKey>? _topDownCacheOrder;

	private readonly struct TopDownCacheKey(long hash, int length, int maxAllowedA, int pLong)
	{
		public readonly long Hash = hash;
		public readonly int Length = length;
		public readonly int MaxAllowedA = maxAllowedA;
		public readonly int PLong = pLong;

		public bool Equals(TopDownCacheKey other) => Hash == other.Hash && Length == other.Length && MaxAllowedA == other.MaxAllowedA && PLong == other.PLong;
		public override bool Equals(object? obj) => obj is TopDownCacheKey other && Equals(other);
		public override int GetHashCode() => HashCode.Combine(Hash, Length, MaxAllowedA, PLong);
	}

	private static long ComputeQOffsetsHash(ReadOnlySpan<int> qOneOffsets)
	{
		const long fnvOffset = unchecked((long)1469598103934665603);
		const long fnvPrime = 1099511628211;
		long hash = fnvOffset;
		for (int i = 0; i < qOneOffsets.Length; i++)
		{
			hash ^= qOneOffsets[i];
			hash *= fnvPrime;
		}

		return hash;
	}

#if DETAILED_LOG
    private static bool TryGetTopDownCache(in TopDownCacheKey key, out TopDownCacheValue value)
#else
	private static bool TryGetTopDownCache(in TopDownCacheKey key, out bool value)
#endif
	{
		var cache = _topDownCache;
		if (cache == null)
		{
			cache = [];
			_topDownCache = cache;
			value = default;
			return false;
		}

		return cache.TryGetValue(key, out value);
	}

#if DETAILED_LOG
    private static void StoreTopDownCache(in TopDownCacheKey key, in TopDownCacheValue value)
#else
	private static void StoreTopDownCache(in TopDownCacheKey key, bool value)
#endif
	{
		// _topDownCache is always non-null here on EvenPerfectBitScanner execution path.
		var topDownCache = _topDownCache!;
		if (topDownCache.ContainsKey(key))
		{
			topDownCache[key] = value;
			return;
		}

		List<TopDownCacheKey>? topDownCacheOrder = _topDownCacheOrder;
		if (topDownCacheOrder == null)
		{
			topDownCacheOrder = new List<TopDownCacheKey>(TopDownCacheCapacity + 1);
			_topDownCacheOrder = topDownCacheOrder;
		}
		else if (topDownCacheOrder.Count >= TopDownCacheCapacity)
		{
			var evict = topDownCacheOrder[0];
			topDownCacheOrder.RemoveAt(0);
			topDownCache.Remove(evict);
		}

		topDownCache[key] = value;
		topDownCacheOrder.Add(key);
	}

	internal static ColumnBounds ComputeColumnBounds(
		in ReadOnlySpan<bool> multiplicand,
		in ReadOnlySpan<bool> multiplier,
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

#if DETAILED_LOG
    internal static bool TryPropagateCarry(
        CarryRange currentCarry,
        long forcedOnes,
        long possibleOnes,
        int requiredResultBit,
        out CarryRange nextCarry,
        out ContradictionReason reason)
#else
	internal static bool TryPropagateCarry(
		CarryRange currentCarry,
		int forcedOnes,
		int possibleOnes,
		int requiredResultBit,
		out CarryRange nextCarry)
#endif
	{
		long minSum = currentCarry.Min + forcedOnes;
		long maxSum = currentCarry.Max + possibleOnes;
		int parity = requiredResultBit & 1;

		minSum = AlignUpToParity(minSum, parity);
		maxSum = AlignDownToParity(maxSum, parity);

		if (minSum > maxSum)
		{
			nextCarry = default;
#if DETAILED_LOG
            reason = ContradictionReason.ParityUnreachable;
#endif
			return false;
		}

		nextCarry = new CarryRange((minSum - requiredResultBit) >> 1, (maxSum - requiredResultBit) >> 1);
#if DETAILED_LOG
        reason = ContradictionReason.None;
#endif
		return true;
	}

	/// <summary>
	/// Full column-wise propagation for a given set of q one-bit offsets (LSB=0) and exponent p.
	/// Returns true if the solver could decide locally (divides or contradiction).
	/// </summary>
	internal static bool TryCheckDivisibilityFromOneOffsets(
		ReadOnlySpan<int> qOneOffsets,
		ulong p,
		out bool divides,
		out ContradictionReason reason)
	{
		divides = false;
#if DETAILED_LOG
		_lastTopDownFailure = null;
		_lastBottomUpFailure = null;
#endif

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
		if ((ulong)(maxOffset + 1) >= p)
		{
			reason = ContradictionReason.TruncatedLength;
			return true;
		}

		// q from EvenPerfectBitScanner will always have the correct 2kp+1 form.
		// if (qOneOffsets[0] != 0)
		// {
		//     reason = ContradictionReason.ParityUnreachable;
		//     return true;
		// }

		int windowSize = maxOffset + 1;
		var segmentState0 = new sbyte[windowSize];
		var segmentState1 = new sbyte[windowSize];
		long segmentBase = 0;
		int pLong = (int)p;
		int maxAllowedA = pLong - (maxOffset + 1);
		if (maxAllowedA < 0)
		{
			reason = ContradictionReason.TruncatedLength;
			return true;
		}

		long qHash = ComputeQOffsetsHash(qOneOffsets);
		var cacheKey = new TopDownCacheKey(qHash, qOneOffsetsLength, maxAllowedA, pLong);
		if (TryGetTopDownCache(in cacheKey, out var cached))
		{
#if DETAILED_LOG
			_lastTopDownFailure = cached.Failure;
			if (!cached.Success)
#else
			if (!cached)
#endif
			{
				reason = ContradictionReason.ParityUnreachable;
#if DETAILED_LOG
				Console.WriteLine($"[bit-contradiction] parity unreachable in top-down prune (failure={_lastTopDownFailure.HasValue})");
#endif
				return true;
			}
		}
		else
		{
#if DETAILED_LOG
			if (!TryTopDownCarryPrune(qOneOffsets, qOneOffsetsLength, pLong, maxAllowedA, out _lastTopDownFailure))
#else
			if (!TryTopDownCarryPrune(qOneOffsets, qOneOffsetsLength, pLong, maxAllowedA))
#endif
			{
#if DETAILED_LOG
				StoreTopDownCache(in cacheKey, new TopDownCacheValue(false, _lastTopDownFailure));
				Console.WriteLine($"[bit-contradiction] parity unreachable in top-down prune (failure={_lastTopDownFailure.HasValue})");
#else
				StoreTopDownCache(in cacheKey, false);
#endif
				reason = ContradictionReason.ParityUnreachable;
				return true;
			}

#if DETAILED_LOG
			StoreTopDownCache(in cacheKey, new TopDownCacheValue(true, null));
#else
			StoreTopDownCache(in cacheKey, true);
#endif
		}

		int lowColumn = 0;
		int highColumn = pLong - 1;
		CarryRange carryLow = CarryRange.Zero;
		CarryRange carryHigh = CarryRange.Zero;
		int bottomWindowStart = 0;
		int bottomWindowEnd = 0;
		int topWindowStart = qOneOffsetsLength;
		int topWindowEnd = qOneOffsetsLength;

		bool processTop = true;

		while (lowColumn <= highColumn)
		{
			int remaining = (int)(highColumn - lowColumn + 1);
			int blockSize = remaining < PingPongBlockColumns ? remaining : PingPongBlockColumns;

			if (processTop)
			{
#if DETAILED_LOG
				if (!TryProcessTopDownBlock(qOneOffsets, qOneOffsetsLength, maxAllowedA, highColumn, blockSize, carryHigh, ref topWindowStart, ref topWindowEnd, out carryHigh, out _lastTopDownFailure))
#else
				if (!TryProcessTopDownBlock(qOneOffsets, qOneOffsetsLength, maxAllowedA, highColumn, blockSize, carryHigh, ref topWindowStart, ref topWindowEnd, out carryHigh))
#endif
				{
					reason = ContradictionReason.ParityUnreachable;
					return true;
				}

				highColumn -= blockSize;
			}
			else
			{
				if (!TryProcessBottomUpBlock(qOneOffsets, maxAllowedA, lowColumn, blockSize, ref carryLow, ref segmentBase, ref bottomWindowStart, ref bottomWindowEnd, segmentState0, segmentState1, out reason))
				{
					return true;
				}

				lowColumn += blockSize;
			}

			processTop = !processTop;
		}

		long carryMin = carryLow.Min > carryHigh.Min ? carryLow.Min : carryHigh.Min;
		long carryMax = carryLow.Max < carryHigh.Max ? carryLow.Max : carryHigh.Max;
		if (carryMin > carryMax)
		{
#if DETAILED_LOG
			Console.WriteLine($"[bit-contradiction] parity unreachable after ping-pong carry merge (low=[{carryLow.Min},{carryLow.Max}] high=[{carryHigh.Min},{carryHigh.Max}])");
#endif
			reason = ContradictionReason.ParityUnreachable;
			divides = false;
			return true;
		}

		divides = true;
		reason = ContradictionReason.None;
		return true;
	}

	internal static bool TryCheckDivisibilityFromOneOffsets(
		ReadOnlySpan<int> qOneOffsets,
		ulong p,
		out bool divides) => TryCheckDivisibilityFromOneOffsets(qOneOffsets, p, out divides, out _);

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static bool TryAdvanceZeroColumns(ref CarryRange carry, int columnCount)
	{
		long carryMin = carry.Min;
		long carryMax = carry.Max;
		while (columnCount > 0)
		{
			int step = columnCount > 30 ? 30 : columnCount;
			int add = (1 << step) - 1;
			long nextMin = (carryMin + add) >> step;
			long nextMax = carryMax >> step;
			if (nextMin > nextMax)
			{
				return false;
			}

			carryMin = nextMin;
			carryMax = nextMax;
			columnCount -= step;
		}

		carry = new CarryRange(carryMin, carryMax);
		return true;
	}

	private static bool TryProcessBottomUpBlock(
		ReadOnlySpan<int> qOneOffsets,
		int maxAllowedA,
		int startColumn,
		int columnCount,
		ref CarryRange carry,
		ref long segmentBase,
		ref int windowStartIdx,
		ref int windowEndIdx,
		sbyte[] segmentState0,
		sbyte[] segmentState1,
		out ContradictionReason reason)
	{
		long endColumn = startColumn + columnCount;
		int qOneOffsetsLength = qOneOffsets.Length;
		int windowSize = segmentState0.Length;
		for (int column = startColumn; column < endColumn; column++)
		{
			int forced = 0;
			int unknown = 0;
			int chosenIndex = -1;
			int newBase = (column / windowSize) * windowSize;
			if (newBase != segmentBase)
			{
				if (newBase == segmentBase + windowSize)
				{
					(segmentState0, segmentState1) = (segmentState1, segmentState0);
					Array.Clear(segmentState0);
				}
				else
				{
					Array.Clear(segmentState0);
					Array.Clear(segmentState1);
				}

				segmentBase = newBase;
			}

			while (windowEndIdx < qOneOffsetsLength && qOneOffsets[windowEndIdx] <= column)
			{
				windowEndIdx++;
			}

			int lowerBound = column - maxAllowedA;
			while (windowStartIdx < windowEndIdx && qOneOffsets[windowStartIdx] < lowerBound)
			{
				windowStartIdx++;
			}

			for (int offsetIndex = windowStartIdx; offsetIndex < windowEndIdx; offsetIndex++)
			{
				int offset = qOneOffsets[offsetIndex];
				int aIndex = column - offset;
				if (aIndex < 0)
				{
					break;
				}

				if (aIndex > maxAllowedA)
				{
					continue; // a must be zero here
				}

				if (aIndex >= segmentBase)
				{
					int slot = (int)(aIndex - segmentBase);
					sbyte state = segmentState0[slot];
					if (state == 1)
					{
						forced++;
					}
					else if (state == 0)
					{
						unknown++;
						if (chosenIndex == -1L)
						{
							chosenIndex = aIndex;
						}
					}
				}
				else if (aIndex >= segmentBase - windowSize)
				{
					int slot = (int)(aIndex - (segmentBase - windowSize));
					sbyte state = segmentState1[slot];
					if (state == 1)
					{
						forced++;
					}
					else if (state == 0)
					{
						unknown++;
						if (chosenIndex == -1L)
						{
							chosenIndex = aIndex;
						}
					}
				}
				else
				{
					unknown++;
					if (chosenIndex == -1L)
					{
						chosenIndex = aIndex;
					}
				}
			}

			long minSum = carry.Min;
			long maxSum = carry.Max;
			if (unknown == 1 && ((minSum ^ maxSum) & 1) == 0)
			{
				int requiredUnknownParity = 1 ^ ((int)minSum & 1) ^ ((int)forced & 1);
				if (requiredUnknownParity != 0)
				{
					if (chosenIndex >= 0)
					{
						if (chosenIndex >= segmentBase)
						{
							int slot = (int)(chosenIndex - segmentBase);
							segmentState0[slot] = 1;
						}
						else
						{
							int slot = (int)(chosenIndex - (segmentBase - windowSize));
							segmentState1[slot] = 1;
						}
						forced++;
						unknown = 0;
					}
				}
				else
				{
					unknown = 0;
				}
			}

			if (unknown == 0)
			{
				int requiredCarryParity = 1 ^ ((int)forced & 1);
				long carryMin = AlignUpToParity(carry.Min, requiredCarryParity);
				long carryMax = AlignDownToParity(carry.Max, requiredCarryParity);
				if (carryMin > carryMax)
				{
#if DETAILED_LOG
					_lastBottomUpFailure = new BottomUpFailure(column, carryMin, carryMax, forced, unknown);
#endif
					reason = ContradictionReason.ParityUnreachable;
					return false;
				}

				carry = new CarryRange(carryMin, carryMax);
				minSum = carry.Min;
				maxSum = carry.Max;
			}

			minSum += forced;
			maxSum += forced + unknown;
			const int requiredParity = 1;

			if (minSum == maxSum)
			{
				if ((minSum & 1) != requiredParity)
				{
#if DETAILED_LOG
					_lastBottomUpFailure = new BottomUpFailure(column, carry.Min, carry.Max, forced, unknown);
#endif
					reason = ContradictionReason.ParityUnreachable;
					return false;
				}

				long nextCarry = (minSum - requiredParity) >> 1;
				carry = CarryRange.Single(nextCarry);
			}
			else
			{
				if ((minSum & 1) != requiredParity && unknown == 0)
				{
#if DETAILED_LOG
					_lastBottomUpFailure = new BottomUpFailure(column, carry.Min, carry.Max, forced, unknown);
#endif
					reason = ContradictionReason.ParityUnreachable;
					return false;
				}

#if DETAILED_LOG
				if (!TryPropagateCarry(
						carry,
						forced,
						forced + unknown,
						requiredParity,
						out var next,
						out var propagateReason))
#else
				if (!TryPropagateCarry(
						carry,
						forced,
						forced + unknown,
						requiredParity,
						out var next))
#endif
				{
#if DETAILED_LOG
					_lastBottomUpFailure = new BottomUpFailure(column, carry.Min, carry.Max, forced, unknown);
					reason = propagateReason;
#else
					reason = ContradictionReason.ParityUnreachable;
#endif
					return false;
				}

				carry = next;
			}

		}

		reason = ContradictionReason.None;
		return true;
	}

#if DETAILED_LOG
	private static bool TryProcessTopDownBlock(
		ReadOnlySpan<int> qOneOffsets,
		int qOneOffsetsLength,
		long maxAllowedA,
		long startHighColumn,
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

				int steps = (int)(column - chunkEnd + 1);
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
		long pLong,
		long maxAllowedA,
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
		long carryOutMin = 0,
			 carryOutMax = 0;
		int value;

		int windowStart = qOneOffsetsLength;
		int windowEnd = qOneOffsetsLength;

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

	private static bool AdvanceTopDownUnknown1(ref long carryOutMin, ref long carryOutMax, int steps)
	{
		while (steps > 0)
		{
			int step = steps > 30 ? 30 : steps;
			int pow = 1 << step;

			if (carryOutMin > (long.MaxValue >> step) || carryOutMax > ((long.MaxValue - (pow - 1)) >> step))
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
}
