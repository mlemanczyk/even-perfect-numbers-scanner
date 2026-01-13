using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

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
        for (int i = 0; i < ModLocalPrefilterModuli.Length; i++)
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
            long x = (long)((long)target * inverse % mod);
            if (x > maxAValue)
            {
                return false;
            }
        }

        return true;
    }

    private static int ComputeQMod(ReadOnlySpan<int> qOneOffsets, int mod)
    {
        long sum = 0;
        for (int i = 0; i < qOneOffsets.Length; i++)
        {
            sum += PowMod2((ulong)qOneOffsets[i], mod);
            if (sum >= mod)
            {
                sum %= mod;
            }
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
        var segmentState0 = new sbyte[windowSize];
        var segmentState1 = new sbyte[windowSize];
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

        if (!TryModularPrefilter(qOneOffsets, p, maxAllowedA))
        {
            reason = ContradictionReason.ParityUnreachable;
            return true;
        }

        long qHash = ComputeQOffsetsHash(qOneOffsets);
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
                if (!TryProcessTopDownBlock(qOneOffsets, qOneOffsetsLength, maxAllowedA, highColumn, blockSize, carryHigh, ref topWindowStart, ref topWindowEnd, out carryHigh, out _lastTopDownFailure))
                {
                    reason = ContradictionReason.ParityUnreachable;
                    return true;
                }

                highColumn -= blockSize;
            }
            else
            {
                if (!TryProcessBottomUpBlock(qOneOffsets, maxAllowedA, lowColumn, blockSize, ref carryLow, ref maxKnownA, ref segmentBase, ref bottomWindowStart, ref bottomWindowEnd, segmentState0, segmentState1, topIndex0, topIndex1, ref top0KnownOne, ref top1KnownOne, ref top0KnownZero, ref top1KnownZero, out reason))
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
        long maxAllowedA,
        long startColumn,
        int columnCount,
        ref CarryRange carry,
        ref long maxKnownA,
        ref long segmentBase,
        ref int windowStartIdx,
        ref int windowEndIdx,
        sbyte[] segmentState0,
        sbyte[] segmentState1,
        long topIndex0,
        long topIndex1,
        ref bool top0KnownOne,
        ref bool top1KnownOne,
        ref bool top0KnownZero,
        ref bool top1KnownZero,
        out ContradictionReason reason)
    {
        long endColumn = startColumn + columnCount;
		int qOneOffsetsLength = qOneOffsets.Length;
        int windowSize = segmentState0.Length;
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
                    windowStartIdx = 0;
                    windowEndIdx = qOneOffsetsLength;
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
                        Array.Clear(segmentState0);
                        Array.Clear(segmentState1);
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

            long lowerBound = column - maxAllowedA;
            while (windowStartIdx < windowEndIdx && qOneOffsets[windowStartIdx] < lowerBound)
            {
                windowStartIdx++;
            }

            for (int offsetIndex = windowStartIdx; offsetIndex < windowEndIdx; offsetIndex++)
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
                    }
                    else if (chosenIndex == topIndex1)
                    {
                        if (top1KnownOne)
                        {
                            reason = ContradictionReason.ParityUnreachable;
                            return false;
                        }

                        top1KnownZero = true;
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
}
