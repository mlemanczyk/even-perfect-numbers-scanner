using System;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core.ByDivisor;

/// <summary>
/// Helper utilities for the BitContradiction divisor search. These methods implement the
/// interval-based carry propagation described in the BitContradictions plan document.
/// The solver itself will build on top of these primitives.
/// </summary>
internal static class BitContradictionSolver
{
    private const int PingPongBlockColumns = 64;
    private const int TailCarryBatchColumns = 1024;
    private const int TailCarryBatchRepeats = 1;

    [ThreadStatic]
    private static Dictionary<long, sbyte>? _aBitsBuffer;
    [ThreadStatic]
    private static TopDownPruneFailure? _lastTopDownFailure;

    internal readonly struct TopDownPruneFailure(long column, long carryMin, long carryMax, long unknown)
    {
        public readonly long Column = column;
        public readonly long CarryMin = carryMin;
        public readonly long CarryMax = carryMax;
        public readonly long Unknown = unknown;
    }

    internal static TopDownPruneFailure? LastTopDownFailure => _lastTopDownFailure;

    internal readonly struct ColumnBounds(long forcedOnes, long possibleOnes)
	{
        public readonly long ForcedOnes = forcedOnes;
        public readonly long PossibleOnes = possibleOnes;
	}

    internal readonly struct CarryRange(long min, long max)
	{
        public readonly long Min = min;
        public readonly long Max = max;

		public static CarryRange Single(long value) => new(value, value);
		public static readonly CarryRange Zero = Single(0);
    }

    internal enum ContradictionReason
    {
        None,
        ParityUnreachable,
        TruncatedLength,
        TailNotZero,
    }

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

		Dictionary<long, sbyte> aBits = _aBitsBuffer ??= new Dictionary<long, sbyte>(capacity: qOneOffsetsLength << 2);
        // aBits.EnsureCapacity(qOneOffsetsLength << 2);
        aBits.Clear();
        long maxKnownA = -1L;
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
                if (!TryProcessBottomUpBlock(qOneOffsets, aBits, maxAllowedA, lowColumn, blockSize, ref carryLow, ref maxKnownA, out reason))
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
            divides = false;
            reason = ContradictionReason.ParityUnreachable;
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
        Dictionary<long, sbyte> aBits,
        long maxAllowedA,
        long startColumn,
        int columnCount,
        ref CarryRange carry,
        ref long maxKnownA,
        out ContradictionReason reason)
    {
        long endColumn = startColumn + columnCount;
		int qOneOffsetsLength = qOneOffsets.Length;
        for (long column = startColumn; column < endColumn; column++)
        {
            long forced = 0L;
            long unknown = 0L;
            long chosenIndex = -1L;

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

                if (aBits.TryGetValue(aIndex, out sbyte aVal))
                {
                    if (aVal == 1)
                    {
                        forced++;
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
                        aBits[chosenIndex] = 1;
                        forced++;
                        unknown = 0;
                        if (chosenIndex > maxKnownA)
                        {
                            maxKnownA = chosenIndex;
                        }
                    }
                }
                else
                {
                    unknown = 0;
                }
            }

            minSum += forced;
            maxSum += forced + unknown;
            const int requiredParity = 1;

            if (minSum == maxSum)
            {
                if ((minSum & 1) != requiredParity)
                {
                    reason = ContradictionReason.ParityUnreachable;
                    return false;
                }

                long nextCarry = (minSum - requiredParity) >> 1;
                carry = CarryRange.Single(nextCarry);
            }
            else
            {
                if ((minSum & 1) != requiredParity)
                {
                    if (unknown == 0)
                    {
                        reason = ContradictionReason.ParityUnreachable;
                        return false;
                    }

                    if (chosenIndex >= 0)
                    {
                        aBits[chosenIndex] = 1;
                        forced++;
                        unknown--;
                        if (chosenIndex > maxKnownA)
                        {
                            maxKnownA = chosenIndex;
                        }
                    }
                }

                if (!TryPropagateCarry(
                        carry,
                        forced,
                        forced + unknown,
                        requiredParity,
                        out var next,
                        out var propagateReason))
                {
                    reason = propagateReason;
                    return false;
                }

                carry = next;
            }

            if (chosenIndex >= 0 && !aBits.TryAdd(chosenIndex, 1))
            {
                if (chosenIndex > maxKnownA)
                {
                    maxKnownA = chosenIndex;
                }
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

        for (long column = startHighColumn; column >= endColumn; column--)
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

        for (long column = pLong - 1; column >= 0; column--)
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
        }

        if (carryOutMin > 0)
        {
            failure = new TopDownPruneFailure(0, carryOutMin, carryOutMax, 0);
            return false;
        }

        return true;
    }
}
