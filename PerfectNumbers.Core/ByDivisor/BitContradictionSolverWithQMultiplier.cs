using System;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core.ByDivisor;

    public readonly struct TopDownPruneFailure(long column, long carryMin, long carryMax, long unknown)
    {
        public readonly long Column = column;
        public readonly long CarryMin = carryMin;
        public readonly long CarryMax = carryMax;
        public readonly long Unknown = unknown;
    }

    public readonly struct ColumnBounds(long forcedOnes, long possibleOnes)
	{
        public readonly long ForcedOnes = forcedOnes;
        public readonly long PossibleOnes = possibleOnes;
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

    [ThreadStatic]
    private static TopDownPruneFailure? _lastTopDownFailure;
    [ThreadStatic]
    private static BottomUpFailure? _lastBottomUpFailure;
    [ThreadStatic]
    private static bool _debugEnabled;

    internal static TopDownPruneFailure? LastTopDownFailure => _lastTopDownFailure;
    internal static BottomUpFailure? LastBottomUpFailure => _lastBottomUpFailure;

    internal readonly struct BottomUpFailure(long column, long carryMin, long carryMax, long forced, long unknown)
    {
        public readonly long Column = column;
        public readonly long CarryMin = carryMin;
        public readonly long CarryMax = carryMax;
        public readonly long Forced = forced;
        public readonly long Unknown = unknown;
    }

    internal static void SetDebugEnabled(bool enabled) => _debugEnabled = enabled;

    internal static ColumnBounds ComputeColumnBounds(
        ReadOnlySpan<bool> multiplicand,
        ReadOnlySpan<bool> multiplier,
        int columnIndex)
    {
        long forced = 0;
        long possible = 0;

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
        CarryRange currentCarry,
        long forcedOnes,
        long possibleOnes,
        int requiredResultBit,
        out CarryRange nextCarry,
        out ContradictionReason reason)
    {
        long minSum = currentCarry.Min + forcedOnes;
        long maxSum = currentCarry.Max + possibleOnes;
        int parity = requiredResultBit & 1;

        minSum = AlignUpToParity(minSum, parity);
        maxSum = AlignDownToParity(maxSum, parity);

        if (minSum > maxSum)
        {
            nextCarry = default;
            reason = ContradictionReason.ParityUnreachable;
            return false;
        }

        nextCarry = new CarryRange((minSum - requiredResultBit) >> 1, (maxSum - requiredResultBit) >> 1);
        reason = ContradictionReason.None;
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
        _lastTopDownFailure = null;
        _lastBottomUpFailure = null;

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
		long pLong = (long)p;
		long maxAllowedA = pLong - (maxOffset + 1L);
        if (maxAllowedA < 0)
        {
            reason = ContradictionReason.TruncatedLength;
            return true;
        }

        if (!TryTopDownCarryPrune(qOneOffsets, qOneOffsetsLength, pLong, maxAllowedA, out _lastTopDownFailure))
        {
            reason = ContradictionReason.ParityUnreachable;
            Console.WriteLine($"[bit-contradiction] parity unreachable in top-down prune (failure={_lastTopDownFailure.HasValue})");
            return true;
        }

        long lowColumn = 0;
        long highColumn = pLong - 1;
        CarryRange carryLow = CarryRange.Zero;
        CarryRange carryHigh = CarryRange.Zero;
        bool processTop = true;

        while (lowColumn <= highColumn)
        {
            int remaining = (int)(highColumn - lowColumn + 1);
            int blockSize = remaining < PingPongBlockColumns ? remaining : PingPongBlockColumns;

            if (processTop)
            {
                if (!TryProcessTopDownBlock(qOneOffsets, qOneOffsetsLength, maxAllowedA, highColumn, blockSize, carryHigh, out carryHigh, out _lastTopDownFailure))
                {
                    reason = ContradictionReason.ParityUnreachable;
                    return true;
                }

                highColumn -= blockSize;
            }
            else
            {
                if (!TryProcessBottomUpBlock(qOneOffsets, maxAllowedA, lowColumn, blockSize, ref carryLow, ref segmentBase, segmentState0, segmentState1, out reason))
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
            Console.WriteLine($"[bit-contradiction] parity unreachable after ping-pong carry merge (low=[{carryLow.Min},{carryLow.Max}] high=[{carryHigh.Min},{carryHigh.Max}])");
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
            int step = columnCount > 62 ? 62 : columnCount;
            long add = (1L << step) - 1L;
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
        long maxAllowedA,
        long startColumn,
        int columnCount,
        ref CarryRange carry,
        ref long segmentBase,
        sbyte[] segmentState0,
        sbyte[] segmentState1,
        out ContradictionReason reason)
    {
        long endColumn = startColumn + columnCount;
		int qOneOffsetsLength = qOneOffsets.Length;
        int windowSize = segmentState0.Length;
        for (long column = startColumn; column < endColumn; column++)
        {
            long forced = 0L;
            long unknown = 0L;
            long chosenIndex = -1L;
            long newBase = (column / windowSize) * windowSize;
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

            for (int offsetIndex = 0; offsetIndex < qOneOffsetsLength; offsetIndex++)
            {
			int offset = qOneOffsets[offsetIndex];
                long aIndex = column - offset;
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
					if (_debugEnabled)
					{
						_lastBottomUpFailure = new BottomUpFailure(column, carryMin, carryMax, forced, unknown);
					}
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
                    if (_debugEnabled)
                    {
                        _lastBottomUpFailure = new BottomUpFailure(column, carry.Min, carry.Max, forced, unknown);
                    }
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
                    if (_debugEnabled)
                    {
                        _lastBottomUpFailure = new BottomUpFailure(column, carry.Min, carry.Max, forced, unknown);
                    }
                    reason = ContradictionReason.ParityUnreachable;
                    return false;
                }

                if (!TryPropagateCarry(
                        carry,
                        forced,
                        forced + unknown,
                        requiredParity,
                        out var next,
                        out var propagateReason))
                {
                    if (_debugEnabled)
                    {
                        _lastBottomUpFailure = new BottomUpFailure(column, carry.Min, carry.Max, forced, unknown);
                    }
                    reason = propagateReason;
                    return false;
                }

                carry = next;
            }

        }

        reason = ContradictionReason.None;
        return true;
    }

    private static bool TryProcessTopDownBlock(
        ReadOnlySpan<int> qOneOffsets,
        int qOneOffsetsLength,
        long maxAllowedA,
        long startHighColumn,
        int columnCount,
        CarryRange carryOut,
        out CarryRange nextCarryOut,
        out TopDownPruneFailure? failure)
    {
        failure = null;
        long carryOutMin = carryOut.Min,
			 carryOutMax = carryOut.Max,
			 value;
        int windowStart = qOneOffsetsLength;
        int windowEnd = qOneOffsetsLength;
        long endColumn = startHighColumn - columnCount + 1;

        long column = startHighColumn;
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
                long lastEnd = windowEnd > 0 ? qOneOffsets[windowEnd - 1] : long.MinValue;
                long lastStart = windowStart > 0 ? qOneOffsets[windowStart - 1] + maxAllowedA : long.MinValue;
                long chunkEnd = lastEnd > lastStart ? lastEnd : lastStart;
                if (chunkEnd < endColumn)
                {
                    chunkEnd = endColumn;
                }

                int steps = (int)(column - chunkEnd + 1);
                if (steps > 1)
                {
                    if (!AdvanceTopDownUnknown0(ref carryOutMin, ref carryOutMax, steps))
                    {
                        failure = new TopDownPruneFailure(column, carryOutMin, carryOutMax, value);
                        nextCarryOut = default;
                        return false;
                    }

                    column = chunkEnd - 1;
                    continue;
                }
            }
            else if (value == 1)
            {
                long lastEnd = windowEnd > 0 ? qOneOffsets[windowEnd - 1] : long.MinValue;
                long lastStart = windowStart > 0 ? qOneOffsets[windowStart - 1] + maxAllowedA : long.MinValue;
                long chunkEnd = lastEnd > lastStart ? lastEnd : lastStart;
                if (chunkEnd < endColumn)
                {
                    chunkEnd = endColumn;
                }

                int steps = (int)(column - chunkEnd + 1);
                if (steps > 1)
                {
                    if (!AdvanceTopDownUnknown1(ref carryOutMin, ref carryOutMax, steps))
                    {
                        failure = new TopDownPruneFailure(column, carryOutMin, carryOutMax, value);
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
                failure = new TopDownPruneFailure(column, carryOutMin, carryOutMax, value);
                nextCarryOut = default;
                return false;
            }

            column--;
        }

        nextCarryOut = new CarryRange(carryOutMin, carryOutMax);
        return true;
    }

    private static long AlignUpToParity(long value, int parity) => (value & 1) == parity ? value : value + 1;

	private static long AlignDownToParity(long value, int parity) => (value & 1) == parity ? value : value - 1;

    private static bool TryTopDownCarryPrune(
        ReadOnlySpan<int> qOneOffsets,
        int qOneOffsetsLength,
        long pLong,
        long maxAllowedA,
        out TopDownPruneFailure? failure)
    {
        failure = null;
        long carryOutMin = 0,
			 carryOutMax = 0,
			 value;
        int windowStart = qOneOffsetsLength;
        int windowEnd = qOneOffsetsLength;

        long column = pLong - 1;
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
                long lastEnd = windowEnd > 0 ? qOneOffsets[windowEnd - 1] : long.MinValue;
                long lastStart = windowStart > 0 ? qOneOffsets[windowStart - 1] + maxAllowedA : long.MinValue;
                long chunkEnd = lastEnd > lastStart ? lastEnd : lastStart;
                if (chunkEnd < 0)
                {
                    chunkEnd = 0;
                }

                int steps = (int)(column - chunkEnd + 1);
                if (steps > 1)
                {
                    if (!AdvanceTopDownUnknown0(ref carryOutMin, ref carryOutMax, steps))
                    {
                        failure = new TopDownPruneFailure(column, carryOutMin, carryOutMax, value);
                        return false;
                    }

                    column = chunkEnd - 1;
                    continue;
                }
            }
            else if (value == 1)
            {
                long lastEnd = windowEnd > 0 ? qOneOffsets[windowEnd - 1] : long.MinValue;
                long lastStart = windowStart > 0 ? qOneOffsets[windowStart - 1] + maxAllowedA : long.MinValue;
                long chunkEnd = lastEnd > lastStart ? lastEnd : lastStart;
                if (chunkEnd < 0)
                {
                    chunkEnd = 0;
                }

                int steps = (int)(column - chunkEnd + 1);
                if (steps > 1)
                {
                    if (!AdvanceTopDownUnknown1(ref carryOutMin, ref carryOutMax, steps))
                    {
                        failure = new TopDownPruneFailure(column, carryOutMin, carryOutMax, value);
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
                failure = new TopDownPruneFailure(column, carryOutMin, carryOutMax, value);
                return false;
            }

            column--;
        }

        if (carryOutMin > 0)
        {
            failure = new TopDownPruneFailure(0, carryOutMin, carryOutMax, 0);
            return false;
        }

        return true;
    }

    private static bool AdvanceTopDownUnknown1(ref long carryOutMin, ref long carryOutMax, int steps)
    {
        while (steps > 0)
        {
            int step = steps > 62 ? 62 : steps;
            long pow = 1L << step;

            if (carryOutMin > (long.MaxValue >> step) || carryOutMax > ((long.MaxValue - (pow - 1)) >> step))
            {
                carryOutMin = 0;
                carryOutMax = long.MaxValue;
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
            int step = steps > 62 ? 62 : steps;
            long add = (1L << step) - 1L;
            if (carryOutMin > ((long.MaxValue - add) >> step) || carryOutMax > ((long.MaxValue - add) >> step))
            {
                carryOutMin = 0;
                carryOutMax = long.MaxValue;
                return true;
            }

            carryOutMin = (carryOutMin << step) + add;
            carryOutMax = (carryOutMax << step) + add;
            steps -= step;
        }

        return true;
    }
}
