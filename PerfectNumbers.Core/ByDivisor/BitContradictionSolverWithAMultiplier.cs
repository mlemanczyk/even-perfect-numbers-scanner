using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Numerics;

namespace PerfectNumbers.Core.ByDivisor;

/// <summary>
/// Helper utilities for the BitContradiction divisor search. These methods implement the
/// interval-based carry propagation described in the BitContradictions plan document.
/// The solver itself will build on top of these primitives.
/// </summary>
internal static class BitContradictionSolverWithAMultiplier
{
    private const int PingPongBlockColumns = 64;
    private const int TailCarryBatchColumns = 1024;
    private const int TailCarryBatchRepeats = 1;

    private const int ModLocalPrefilterMaxABits = 20;
    private static readonly int[] ModLocalPrefilterModuli = { 65537, 65521 };

    [ThreadStatic]
    private static TopDownPruneFailure? _lastTopDownFailure;
    private static bool _debugEnabled = true;


    private const int TopDownCacheCapacity = 16;

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

    private readonly struct TopDownCacheValue(bool success, TopDownPruneFailure? failure)
    {
        public readonly bool Success = success;
        public readonly TopDownPruneFailure? Failure = failure;
    }

    [ThreadStatic]
    private static Dictionary<TopDownCacheKey, TopDownCacheValue>? _topDownCache;
    [ThreadStatic]
    private static List<TopDownCacheKey>? _topDownCacheOrder;

    private static long ComputeQOffsetsHash(ReadOnlySpan<int> qOneOffsets)
    {
        const long fnvOffset = unchecked(1469598103934665603);
        const long fnvPrime = 1099511628211;
        long hash = fnvOffset;
        for (int i = 0; i < qOneOffsets.Length; i++)
        {
            hash ^= qOneOffsets[i];
            hash *= fnvPrime;
        }

        return hash;
    }

    private static bool TryGetTopDownCache(in TopDownCacheKey key, out TopDownCacheValue value)
    {
        if (_topDownCache == null)
        {
            value = default;
            return false;
        }

        return _topDownCache.TryGetValue(key, out value);
    }

    private static void StoreTopDownCache(in TopDownCacheKey key, in TopDownCacheValue value)
    {
        _topDownCache ??= new Dictionary<TopDownCacheKey, TopDownCacheValue>();
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
    private static int _parityZeroLogCount;

    internal static TopDownPruneFailure? LastTopDownFailure => _lastTopDownFailure;

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

    private static bool TryModularPrefilter(
        ReadOnlySpan<int> qOneOffsets,
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
            int qMod = ComputeQMod(qOneOffsets, mod);
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

    [ThreadStatic]
    private static Dictionary<int, int[]>? _pow2DiffModCache;

    private static int[] GetPow2DiffLut(int mod)
    {
        _pow2DiffModCache ??= new Dictionary<int, int[]>();
        if (_pow2DiffModCache.TryGetValue(mod, out int[]? lut))
        {
            return lut;
        }

        int[] values = new int[256];
        values[0] = 1 % mod;
        for (int i = 1; i < values.Length; i++)
        {
            values[i] = (values[i - 1] * 2) % mod;
        }

        _pow2DiffModCache[mod] = values;
        return values;
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

    private static int ModPow2Minus1(ulong p, int mod)
    {
        int pow = PowMod2(p, mod);
        int result = pow - 1;
        return result < 0 ? result + mod : result;
    }

    private static int PowMod2(ulong exp, int mod)
    {
        long result = 1 % mod;
        long baseVal = 2 % mod;
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

        int windowSize = maxOffset + 1;
        int wordCount = (windowSize + 63) >> 6;
        var knownOne0 = new ulong[wordCount];
        var knownOne1 = new ulong[wordCount];
        var knownZero0 = new ulong[wordCount];
        var knownZero1 = new ulong[wordCount];
        long segmentBase = 0;
        long maxKnownA = -1L;
		long pLong = (long)p;
		long maxAllowedA = pLong - (maxOffset + 1L);
        if (maxAllowedA < 0)
        {
            reason = ContradictionReason.TruncatedLength;
            return true;
        }

        long topIndex0 = maxAllowedA;
        long topIndex1 = maxAllowedA - 1L;
        bool top0KnownOne = topIndex0 == 0L;
        bool top1KnownOne = topIndex1 == 0L;
        bool top0KnownZero = false;
        bool top1KnownZero = false;
        long zeroTailStart = maxAllowedA + 1L;

        if (!TryModularPrefilter(qOneOffsets, p, maxAllowedA))
        {
            reason = ContradictionReason.ParityUnreachable;
            return true;
        }

        long qHash = ComputeQOffsetsHash(qOneOffsets);
        Span<int> runStarts = qOneOffsetsLength <= 256 ? stackalloc int[qOneOffsetsLength] : new int[qOneOffsetsLength];
        Span<int> runLengths = qOneOffsetsLength <= 256 ? stackalloc int[qOneOffsetsLength] : new int[qOneOffsetsLength];
        int runCount = 0;
        {
            int prev = qOneOffsets[0];
            int currentStart = prev;
            int currentLength = 1;
            for (int i = 1; i < qOneOffsetsLength; i++)
            {
                int offset = qOneOffsets[i];
                if (offset == prev + 1)
                {
                    currentLength++;
                }
                else
                {
                    runStarts[runCount] = currentStart;
                    runLengths[runCount] = currentLength;
                    runCount++;
                    currentStart = offset;
                    currentLength = 1;
                }

                prev = offset;
            }

            runStarts[runCount] = currentStart;
            runLengths[runCount] = currentLength;
            runCount++;
        }
        var cacheKey = new TopDownCacheKey(qHash, qOneOffsetsLength, maxAllowedA, pLong);
        if (TryGetTopDownCache(in cacheKey, out var cached))
        {
            _lastTopDownFailure = cached.Failure;
            if (!cached.Success)
            {
                reason = ContradictionReason.ParityUnreachable;
                return true;
            }
        }
        else
        {
            if (!TryTopDownCarryPrune(qOneOffsets, qOneOffsetsLength, pLong, maxAllowedA, out _lastTopDownFailure))
            {
                StoreTopDownCache(in cacheKey, new TopDownCacheValue(false, _lastTopDownFailure));
                reason = ContradictionReason.ParityUnreachable;
                return true;
            }

            StoreTopDownCache(in cacheKey, new TopDownCacheValue(true, null));
        }

        long lowColumn = 0;
        long highColumn = pLong - 1;
        CarryRange carryLow = CarryRange.Zero;
        CarryRange carryHigh = CarryRange.Zero;
        int bottomRunStart = 0;
        int bottomRunEnd = 0;
        int topWindowStart = qOneOffsetsLength;
        int topWindowEnd = qOneOffsetsLength;

        bool processTop = true;

        while (lowColumn <= highColumn)
        {
            int remaining = (int)(highColumn - lowColumn + 1);
            int blockSize = remaining < PingPongBlockColumns ? remaining : PingPongBlockColumns;

            if (processTop)
            {
                if (!TryProcessTopDownBlock(qOneOffsets, qOneOffsetsLength, maxAllowedA, highColumn, blockSize, carryHigh, ref topWindowStart, ref topWindowEnd, out carryHigh, out _lastTopDownFailure))
                {
                    reason = ContradictionReason.ParityUnreachable;
                    return true;
                }

                highColumn -= blockSize;
            }
            else
            {
                if (!TryProcessBottomUpBlock(qOneOffsets, runStarts, runLengths, runCount, maxAllowedA, lowColumn, blockSize, ref carryLow, ref maxKnownA, ref segmentBase, ref bottomRunStart, ref bottomRunEnd, knownOne0, knownOne1, knownZero0, knownZero1, windowSize, topIndex0, topIndex1, ref top0KnownOne, ref top1KnownOne, ref top0KnownZero, ref top1KnownZero, ref zeroTailStart, out reason))
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

    [ThreadStatic]
    private static Dictionary<int, int[]>? _stableUnknownDelta8;

    private static int[] GetStableUnknownDelta8(int unknown)
    {
        _stableUnknownDelta8 ??= new Dictionary<int, int[]>();
        if (_stableUnknownDelta8.TryGetValue(unknown, out int[]? cached))
        {
            return cached;
        }

        int[] delta = new int[256];
        for (int low = 0; low < 256; low++)
        {
            long value = low;
            for (int i = 0; i < 8; i++)
            {
                int sMax = (((unknown ^ (int)value) & 1) != 0) ? unknown : unknown - 1;
                value = (value + sMax - 1) >> 1;
            }

            delta[low] = (int)((value << 8) - low);
        }

        _stableUnknownDelta8[unknown] = delta;
        return delta;
    }

    private static bool TryAdvanceStableUnknown(ref CarryRange carry, int unknown, int columnCount, out ContradictionReason reason)
    {
        if (columnCount <= 0)
        {
            reason = ContradictionReason.None;
            return true;
        }

        if (unknown <= 0)
        {
            if (!TryAdvanceZeroColumns(ref carry, columnCount))
            {
                reason = ContradictionReason.ParityUnreachable;
                return false;
            }

            reason = ContradictionReason.None;
            return true;
        }

        long carryMin = carry.Min >> columnCount;
        long carryMax = carry.Max;
        int remaining = columnCount;
        int[] delta = GetStableUnknownDelta8(unknown);

        while (remaining >= 8)
        {
            int low = (int)(carryMax & 255);
            carryMax = (carryMax + delta[low]) >> 8;
            remaining -= 8;
        }

        while (remaining > 0)
        {
            int sMax = (((unknown ^ (int)carryMax) & 1) != 0) ? unknown : unknown - 1;
            carryMax = (carryMax + sMax - 1) >> 1;
            remaining--;
        }

        if (carryMax < carryMin)
        {
            reason = ContradictionReason.ParityUnreachable;
            return false;
        }

        carry = new CarryRange(carryMin, carryMax);
        reason = ContradictionReason.None;
        return true;
    }

    private static bool TryProcessBottomUpBlock(
        ReadOnlySpan<int> qOneOffsets,
        ReadOnlySpan<int> runStarts,
        ReadOnlySpan<int> runLengths,
        int runCount,
        long maxAllowedA,
        long startColumn,
        int columnCount,
        ref CarryRange carry,
        ref long maxKnownA,
        ref long segmentBase,
        ref int runStartIdx,
        ref int runEndIdx,
        ulong[] knownOne0,
        ulong[] knownOne1,
        ulong[] knownZero0,
        ulong[] knownZero1,
        int windowSize,
        long topIndex0,
        long topIndex1,
        ref bool top0KnownOne,
        ref bool top1KnownOne,
        ref bool top0KnownZero,
        ref bool top1KnownZero,
        ref long zeroTailStart,
        out ContradictionReason reason)
    {
        long endColumn = startColumn + columnCount;
		int qOneOffsetsLength = qOneOffsets.Length;
        int maxOffset = qOneOffsets[qOneOffsetsLength - 1];
        long stableUnknown = qOneOffsetsLength;
        long column = startColumn;
        while (column < endColumn)
        {
            if (column >= maxOffset && column <= maxAllowedA && maxKnownA < column - maxOffset)
            {
                long batchEnd = endColumn - 1;
                if (batchEnd > maxAllowedA)
                {
                    batchEnd = maxAllowedA;
                }

                long remaining = batchEnd - column + 1;
                if (remaining > 0)
                {
                    runStartIdx = 0;
                    runEndIdx = runCount;
                    while (remaining > 0)
                    {
                        int step = remaining > TailCarryBatchColumns ? TailCarryBatchColumns : (int)remaining;
                        if (!TryAdvanceStableUnknown(ref carry, (int)stableUnknown, step, out var propagateReason))
                        {
                            reason = propagateReason;
                            return false;
                        }

                        column += step;
                        remaining -= step;
                    }

                    long batchBase = (column / windowSize) * windowSize;
                    if (batchBase != segmentBase)
                    {
                        Array.Clear(knownOne0);
                        Array.Clear(knownOne1);
                        Array.Clear(knownZero0);
                        Array.Clear(knownZero1);
                        segmentBase = batchBase;
                    }

                    continue;
                }
            }
            long forced = 0L;
            long unknown = 0L;
            long chosenIndex = -1L;
            long newBase = (column / windowSize) * windowSize;
            if (newBase != segmentBase)
            {
                if (newBase == segmentBase + windowSize)
                {
                    (knownOne0, knownOne1) = (knownOne1, knownOne0);
                    (knownZero0, knownZero1) = (knownZero1, knownZero0);
                    Array.Clear(knownOne0);
                    Array.Clear(knownZero0);
                }
                else
                {
                    Array.Clear(knownOne0);
                    Array.Clear(knownOne1);
                    Array.Clear(knownZero0);
                    Array.Clear(knownZero1);
                }

                segmentBase = newBase;
            }

            long lowerBound = column - maxAllowedA;
            while (runEndIdx < runCount && runStarts[runEndIdx] <= column)
            {
                runEndIdx++;
            }

            while (runStartIdx < runEndIdx)
            {
                int runStart = runStarts[runStartIdx];
                int runEnd = runStart + runLengths[runStartIdx] - 1;
                if (runEnd < lowerBound)
                {
                    runStartIdx++;
                    continue;
                }

                break;
            }

            for (int runIndex = runStartIdx; runIndex < runEndIdx; runIndex++)
            {
                int runStart = runStarts[runIndex];
                int runEnd = runStart + runLengths[runIndex] - 1;
                if (runStart > column)
                {
                    break;
                }

                if (runEnd > column)
                {
                    runEnd = (int)column;
                }

                if (runStart < lowerBound)
                {
                    runStart = (int)lowerBound;
                }

                if (runStart > runEnd)
                {
                    continue;
                }

                long aStart = column - runEnd;
                long aEnd = column - runStart;
                if (aStart > maxAllowedA)
                {
                    continue;
                }

                if (aEnd > maxAllowedA)
                {
                    aEnd = maxAllowedA;
                }

                long rangeLength = aEnd - aStart + 1;
                if (rangeLength <= 0)
                {
                    continue;
                }

                long seg0Start = segmentBase;
                long seg0End = segmentBase + windowSize - 1;
                long seg1Start = segmentBase - windowSize;
                long seg1End = segmentBase - 1;

                long overlap0Start = aStart > seg0Start ? aStart : seg0Start;
                long overlap0End = aEnd < seg0End ? aEnd : seg0End;
                long overlap1Start = aStart > seg1Start ? aStart : seg1Start;
                long overlap1End = aEnd < seg1End ? aEnd : seg1End;

                long known0Len = 0;
                if (overlap0Start <= overlap0End)
                {
                    int relStart = (int)(overlap0Start - seg0Start);
                    int len = (int)(overlap0End - overlap0Start + 1);
                    long ones = CountBitsInRange(knownOne0, relStart, len);
                    long zeros = CountBitsInRange(knownZero0, relStart, len);
                    forced += ones;
                    unknown += len - ones - zeros;
                    known0Len = len;

                    if (chosenIndex == -1L && len > ones + zeros && TryFindFirstUnknown(knownOne0, knownZero0, relStart, len, out int idx))
                    {
                        chosenIndex = seg0Start + idx;
                    }
                }

                long known1Len = 0;
                if (overlap1Start <= overlap1End)
                {
                    int relStart = (int)(overlap1Start - seg1Start);
                    int len = (int)(overlap1End - overlap1Start + 1);
                    long ones = CountBitsInRange(knownOne1, relStart, len);
                    long zeros = CountBitsInRange(knownZero1, relStart, len);
                    forced += ones;
                    unknown += len - ones - zeros;
                    known1Len = len;

                    if (chosenIndex == -1L && len > ones + zeros && TryFindFirstUnknown(knownOne1, knownZero1, relStart, len, out int idx))
                    {
                        chosenIndex = seg1Start + idx;
                    }
                }

                long knownTotal = known0Len + known1Len;
                if (knownTotal < rangeLength)
                {
                    unknown += rangeLength - knownTotal;
                }

                if (zeroTailStart <= aEnd)
                {
                    long tailStart = aStart > zeroTailStart ? aStart : zeroTailStart;
                    if (tailStart <= aEnd)
                    {
                        long tailZeros = aEnd - tailStart + 1;
                        if (tailZeros > 0)
                        {
                            unknown -= tailZeros;
                            if (unknown < 0)
                            {
                                unknown = 0;
                            }
                        }
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
                            int word = slot >> 6;
                            ulong mask = 1UL << (slot & 63);
                            knownOne0[word] |= mask;
                            knownZero0[word] &= ~mask;
                        }
                        else
                        {
                            int slot = (int)(chosenIndex - (segmentBase - windowSize));
                            int word = slot >> 6;
                            ulong mask = 1UL << (slot & 63);
                            knownOne1[word] |= mask;
                            knownZero1[word] &= ~mask;
                        }
                        forced++;
                        unknown = 0;
                        if (chosenIndex > maxKnownA)
                        {
                            maxKnownA = chosenIndex;
                        }
                        if (chosenIndex == topIndex0)
                        {
                            if (top0KnownZero)
                            {
                                reason = ContradictionReason.ParityUnreachable;
                                return false;
                            }

                            top0KnownOne = true;
                        }
                        else if (chosenIndex == topIndex1)
                        {
                            if (top1KnownZero)
                            {
                                reason = ContradictionReason.ParityUnreachable;
                                return false;
                            }

                            top1KnownOne = true;
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
                    unknown = 0;

                    if (chosenIndex == topIndex0)
                    {
                        if (top0KnownOne)
                        {
                            reason = ContradictionReason.ParityUnreachable;
                            return false;
                        }

                        top0KnownZero = true;
                        UpdateZeroTailFromTop(ref zeroTailStart, topIndex0, topIndex1, top0KnownZero, top1KnownZero);
                    }
                    else if (chosenIndex == topIndex1)
                    {
                        if (top1KnownOne)
                        {
                            reason = ContradictionReason.ParityUnreachable;
                            return false;
                        }

                        top1KnownZero = true;
                        UpdateZeroTailFromTop(ref zeroTailStart, topIndex0, topIndex1, top0KnownZero, top1KnownZero);
                    }
                }
            }

            if ((top0KnownZero && top0KnownOne) || (top1KnownZero && top1KnownOne))
            {
                reason = ContradictionReason.ParityUnreachable;
                return false;
            }

            if (topIndex1 < 0)
            {
                if (top0KnownZero)
                {
                    reason = ContradictionReason.ParityUnreachable;
                    return false;
                }
            }
            else if (top0KnownZero && top1KnownZero)
            {
                reason = ContradictionReason.ParityUnreachable;
                return false;
            }

			if (unknown == 0)
			{
				int requiredCarryParity = 1 ^ ((int)forced & 1);
				long carryMin = AlignUpToParity(carry.Min, requiredCarryParity);
				long carryMax = AlignDownToParity(carry.Max, requiredCarryParity);
				if (carryMin > carryMax)
				{
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
                    reason = propagateReason;
                    return false;
                }

                carry = next;
            }

            column++;
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
        ref int windowStart,
        ref int windowEnd,
        out CarryRange nextCarryOut,
        out TopDownPruneFailure? failure)
    {
        failure = null;
        long carryOutMin = carryOut.Min,
			 carryOutMax = carryOut.Max,
			 value;
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
        int windowStart = qOneOffsetsLength;
        int windowEnd = qOneOffsetsLength;
        long carryOutMin = 0,
			 carryOutMax = 0,
			 value;

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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static long CountBitsInRange(ulong[] bits, int start, int length)
    {
        if (length <= 0)
        {
            return 0;
        }

        int end = start + length - 1;
        int firstWord = start >> 6;
        int lastWord = end >> 6;
        int startBit = start & 63;
        int endBit = end & 63;

        if (firstWord == lastWord)
        {
            ulong mask = (endBit == 63 ? ulong.MaxValue : ((1UL << (endBit + 1)) - 1UL)) & (ulong.MaxValue << startBit);
            return BitOperations.PopCount(bits[firstWord] & mask);
        }

        long count = 0;
        ulong firstMask = ulong.MaxValue << startBit;
        count += BitOperations.PopCount(bits[firstWord] & firstMask);

        for (int word = firstWord + 1; word < lastWord; word++)
        {
            count += BitOperations.PopCount(bits[word]);
        }

        ulong lastMask = endBit == 63 ? ulong.MaxValue : ((1UL << (endBit + 1)) - 1UL);
        count += BitOperations.PopCount(bits[lastWord] & lastMask);
        return count;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool TryFindFirstUnknown(ulong[] knownOne, ulong[] knownZero, int start, int length, out int index)
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
    private static void UpdateZeroTailFromTop(ref long zeroTailStart, long topIndex0, long topIndex1, bool top0KnownZero, bool top1KnownZero)
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
}
