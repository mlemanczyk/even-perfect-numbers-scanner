
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using PerfectNumbers.Core;

namespace PerfectNumbers.Core.Gpu.Kernels;
	
internal static class BitContradictionKernels
{
    public const int BatchCount = PerfectNumberConstants.BitContradictionGpuBatchCount;
    public const int ForcedALowBits = 1024;
    public const int ForcedALowWords = 16;
    public const int MaxQBitLength = 4096;
    public const int MaxQWordCount = MaxQBitLength / 64;
    public const int MaxQOffsets = 4096;
    public const int MaxAWordCount = MaxQBitLength / 64;

    private const int InverseIterations = 10;
    private const int HighBitCarryPrefilterColumns = 16;
    private const int TopDownBorrowColumns = 4096;
    private const int TopDownBorrowMinUnknown = 12;
    private const int TailCarryBatchColumns = 16384;
    private const int DeltaColumnsAtOnce = 16;
    private const int DeltaLength = 1 << DeltaColumnsAtOnce;
    internal const int DeltaCacheSlots = 500;
    public const int DebugWordCountPerSlot = 80;
    private const int IntervalBufferLength = 500;

    private struct CarryRange(long min, long max)
	{
        public long Min = min;
        public long Max = max;

		public static CarryRange Zero => new(0, 0);
    }


    /// <summary>
    /// Required number of int entries for the per-thread delta cache buffer.
    /// Layout: [thread][slot][DeltaLength]
    /// </summary>
    public static long GetDeltaCacheIntLength(int threadCount) =>
        (long)threadCount * DeltaCacheSlots * DeltaLength;

    /// <summary>
    /// Required number of int entries for the per-thread delta cache key buffer.
    /// Layout: [thread][slot] -> stored 'unknown' value for the slot.
    /// </summary>
    public static int GetDeltaCacheKeyLength(int threadCount) =>
        checked(threadCount * DeltaCacheSlots);


    public static void BitContradictionKernelScanWithDeltaCache64(
        Index1D index,
        ulong exponent,
        ArrayView1D<ulong, Stride1D.Dense> batchIndexWords,
        int countQ,
        ArrayView1D<int, Stride1D.Dense> foundOut,
#if DETAILED_LOG
        ArrayView1D<ulong, Stride1D.Dense> debugOut,
#endif
        ArrayView1D<int, Stride1D.Dense> deltaCacheKeys,
        ArrayView1D<int, Stride1D.Dense> deltaCache)
    {
		// firstQWordsEff is reused as slot here to limit registry pressure
        int firstQWordsEff = index;
        foundOut[firstQWordsEff] = -1;

        int keyBase = firstQWordsEff * DeltaCacheSlots;
        long deltaBase = (long)firstQWordsEff * DeltaCacheSlots * DeltaLength;
        // Initialize keys for this thread to avoid false hits across launches.
		int i;
        for (i = 0; i < DeltaCacheSlots; i++)
        {
            deltaCacheKeys[keyBase + i] = int.MinValue;
        }

        if (countQ <= 0)
        {
            return;
        }

        ulong twoP = exponent << 1;
		// firstQWordsEff is reused as startOffset here to limit registry pressure
        firstQWordsEff *= MaxQWordCount;
        int localCountQ = countQ;

        var baseKWords = LocalMemory.Allocate<ulong>(MaxQWordCount);
        var kWords = LocalMemory.Allocate<ulong>(MaxQWordCount);
        var qWords = LocalMemory.Allocate<ulong>(MaxQWordCount);
        var qOffsets = LocalMemory.Allocate<int>(MaxQOffsets);
        var qPrefix = LocalMemory.Allocate<ulong>(ForcedALowWords);

        var invWords = LocalMemory.Allocate<ulong>(ForcedALowWords);
        var tmpWords = LocalMemory.Allocate<ulong>(ForcedALowWords);
        var tmpWords2 = LocalMemory.Allocate<ulong>(ForcedALowWords);
        var aLowWords = LocalMemory.Allocate<ulong>(ForcedALowWords);
        var knownOne0 = LocalMemory.Allocate<ulong>(MaxAWordCount);
        var knownOne1 = LocalMemory.Allocate<ulong>(MaxAWordCount);
        var knownZero0 = LocalMemory.Allocate<ulong>(MaxAWordCount);
        var knownZero1 = LocalMemory.Allocate<ulong>(MaxAWordCount);
        var qMaskWords = LocalMemory.Allocate<ulong>(MaxQWordCount);
        var aOneWin = LocalMemory.Allocate<ulong>(MaxQWordCount);
        var aZeroWin = LocalMemory.Allocate<ulong>(MaxQWordCount);

        var lo0 = LocalMemory.Allocate<long>(IntervalBufferLength);
        var hi0 = LocalMemory.Allocate<long>(IntervalBufferLength);
        var lo1 = LocalMemory.Allocate<long>(IntervalBufferLength);
        var hi1 = LocalMemory.Allocate<long>(IntervalBufferLength);

        for (i = 0; i < MaxQWordCount; i++)
        {
            baseKWords[i] = batchIndexWords[firstQWordsEff + i];
        }

        if (MultiplyWordsByUInt64(baseKWords, kWords, MaxQWordCount, (ulong)BatchCount) != 0UL)
        {
            return;
        }

        if (AddUInt64ToWords(kWords, MaxQWordCount, 1UL) != 0UL)
        {
            return;
        }

        CopyWords(kWords, baseKWords, MaxQWordCount);

        for (int batch = 0; batch < localCountQ; batch++)
        {
            CopyWords(baseKWords, kWords, MaxQWordCount);
            if (batch != 0)
            {
                if (AddUInt64ToWords(kWords, MaxQWordCount, (ulong)batch) != 0UL)
                {
                    break;
                }
            }

            if (MultiplyWordsByUInt64(kWords, qWords, MaxQWordCount, twoP) != 0UL)
            {
                continue;
            }

            if (AddUInt64ToWords(qWords, MaxQWordCount, 1UL) != 0UL)
            {
                continue;
            }

            int qBitLen = GetBitLength(qWords, MaxQWordCount);
            if (qBitLen <= 0 || qBitLen > MaxQBitLength)
            {
                continue;
            }

            if ((ulong)qBitLen >= exponent)
            {
                continue;
            }

            int pLong = (int)exponent;
            int maxAllowedA = pLong - qBitLen;
            if (maxAllowedA < 0)
            {
                continue;
            }

            int qWordCount = (qBitLen + 63) >> 6;
            int offsetCount;
            if (!BuildQOneOffsetsWords(qWords, qWordCount, qOffsets, out offsetCount))
            {
                continue;
            }

            if (offsetCount <= 0)
            {
                continue;
            }

			int maxOffsetValue = qOffsets[offsetCount - 1];
            int windowSize = qBitLen < ForcedALowBits ? ForcedALowBits : qBitLen;
            windowSize = (windowSize + 63) & ~63;
            int aWordCount = (windowSize + 63) >> 6;

            ClearWords(knownOne0, aWordCount);
            ClearWords(knownOne1, aWordCount);
            ClearWords(knownZero0, aWordCount);
            ClearWords(knownZero1, aWordCount);
            ClearWords(qMaskWords, MaxQWordCount);
            ClearWords(aOneWin, MaxQWordCount);
            ClearWords(aZeroWin, MaxQWordCount);

            ClearWords(qPrefix, ForcedALowWords);
            CopyWords(qWords, qPrefix, ForcedALowWords);
            CopyWords(qPrefix, invWords, ForcedALowWords);
            ComputeInverseMod2k(qPrefix, invWords, tmpWords, tmpWords2, ForcedALowWords);
            NegateMod2k(invWords, aLowWords, ForcedALowWords);

            for (i = 0; i < qWordCount; i++)
            {
                qMaskWords[i] = qWords[i];
            }

			i = qBitLen & 63;
            ulong lastWordMask = i == 0 ? ulong.MaxValue : ((1UL << i) - 1UL);
            if (qWordCount > 0)
            {
                qMaskWords[qWordCount - 1] &= lastWordMask;
            }

			int firstRemaining;
			ulong firstQLastEff, wordValue;
            int maxFixed = maxAllowedA < (ForcedALowBits - 1) ? maxAllowedA : (ForcedALowBits - 1);
            if (maxFixed >= 0)
            {
				// firstQWordsEff is reused as fullWords here to limit registry pressure
                firstQWordsEff = maxFixed >> 6;
				// firstRemaining is reused as lastBit here to limit registry pressure
                firstRemaining = maxFixed & 63;
				// i is reused as word to limit registry pressure
                for (i = 0; i < firstQWordsEff; i++)
                {
                    wordValue = aLowWords[i];
                    knownOne0[i] = wordValue;
                    knownZero0[i] = ~wordValue;
                }

                if (firstQWordsEff < aWordCount)
                {
                    wordValue = aLowWords[firstQWordsEff];
					// firstQLastEff is reused as mask here to limit registry pressure
                    firstQLastEff = firstRemaining == 63 ? ulong.MaxValue : ((1UL << (firstRemaining + 1)) - 1UL);
                    knownOne0[firstQWordsEff] = wordValue & firstQLastEff;
                    knownZero0[firstQWordsEff] = (~wordValue) & firstQLastEff;
                }
            }

            int maxKnownA = maxFixed >= 0 ? maxFixed : -1;
            int firstForced = -1;
            firstRemaining = -1;
            firstQWordsEff = 0;
            firstQLastEff = 0;
            long firstCarryMin = 0;
            long firstCarryMax = 0;

            int maxFixedPreFilter = 512 - 1;
            maxFixedPreFilter = maxFixedPreFilter > maxKnownA ? maxKnownA : maxFixedPreFilter;
            maxFixedPreFilter = maxFixedPreFilter > maxAllowedA ? maxAllowedA : maxFixedPreFilter;

			int column;
			int word;
			int aIndex;
			int forced;
			int bitShift;
            if (maxFixedPreFilter >= 8)
            {
                // tmpWords is reused for reversePrefix
                ReverseBits1024(qPrefix, tmpWords);

                bool prefixOk = true;
                CarryRange carryLow = CarryRange.Zero;
				for (column = 0; column <= maxFixedPreFilter; column++)
                {
                    forced = 0;
                    if (column <= 1023)
                    {
						// i is reused as shift here to limit registry pressure
                        i = 1023 - column;
						// word is reused as wordShift here to limit registry pressure
                        word = i >> 6;
                        bitShift = i & 63;
						// i is reused as word to limit registry pressure
                        for (i = 0; i < ForcedALowWords; i++)
                        {
							// aIndex is reused as src here to limit registry pressure
                            aIndex = i + word;
                            wordValue = 0UL;
                            if (aIndex < ForcedALowWords)
                            {
                                wordValue = tmpWords[aIndex] >> bitShift;
                                if (bitShift != 0 && aIndex + 1 < ForcedALowWords)
                                {
                                    wordValue |= tmpWords[aIndex + 1] << (64 - bitShift);
                                }
                            }

                            // tmpWords2 is reused for prefixMask
                            tmpWords2[i] = wordValue;
                        }

						// i is reused as word to limit registry pressure
                        for (i = 0; i < ForcedALowWords; i++)
                        {
                            forced += XMath.PopCount(knownOne0[i] & tmpWords2[i]);
                        }
                    }
                    else
                    {
                        for (i = 0; i < offsetCount; i++)
                        {
							// bitShift is reused as t here to limit registry pressure
                            bitShift = qOffsets[i];
                            if (bitShift > column)
                            {
                                break;
                            }

                            aIndex = column - bitShift;
                            word = aIndex >> 6;
							// wordValue is reused as mask here to limit registry pressure
                            wordValue = 1UL << (aIndex & 63);
                            if ((knownOne0[word] & wordValue) != 0UL)
                            {
                                forced++;
                            }
                        }
                    }

                    if (!TryPropagateCarry(ref carryLow, forced, forced, 1))
                    {
                        prefixOk = false;
                        break;
                    }
                }

                if (!prefixOk)
                {
                    continue;
                }
            }

			// word is reused as segmentBase from here to limit registry pressure
            word = 0;
			// column is reused as windowColumn from here to limit registry pressure
            column = 0;

			// aIndex is reused as initialState from here to limit registry pressure
            aIndex = GetAKnownStateRowAware(0, word, windowSize, knownOne0, knownZero0, knownOne1, knownZero1);
            aOneWin[0] |= aIndex == 1 ? 1UL : 0UL;
            aZeroWin[0] |= aIndex == 2 ? 1UL : 0UL;

            if (!TryRunHighBitAndBorrowPrefiltersCombinedGpu(qOffsets, offsetCount, pLong, maxAllowedA, lo0, hi0, lo1, hi1))
            {
                continue;
            }

			// forced is reused as unknown from here to limit registry pressure
            forced = offsetCount;
			// i is reused as cacheSlot from here to limit registry pressure
            i = forced & (DeltaCacheSlots - 1);
			// bitShift is reused as keyIndex from here to limit registry pressure
            bitShift = keyBase + i;
			// i is reused as cacheSlotBase from here to limit registry pressure
            i = (int)(deltaBase + (long)i * DeltaLength);
            var delta8 = deltaCache.SubView(i, DeltaLength);
            if (deltaCacheKeys[bitShift] != forced)
            {
                BuildStableUnknownDelta8(forced, delta8);
                deltaCacheKeys[bitShift] = forced;
            }

            CarryRange carry = CarryRange.Zero;
			#if DETAILED_LOG
				int preWindowColumn = windowColumn;
				int preSegmentBase = segmentBase;
				ulong preKnownZero0 = knownZero0[0];
				ulong preKnownOne0 = knownOne0[0];
				ulong preAOneWin0 = aOneWin[0];
				ulong preAZeroWin0 = aZeroWin[0];
				long preCarryMin = 0L;
				long preCarryMax = 0L;
			#endif

            bool ok = TryProcessBottomUpBlockRowAwareGpu(
                qOffsets,
                offsetCount,
                qMaskWords,
                qWordCount,
                lastWordMask,
                maxAllowedA,
                0,
                pLong,
                ref carry,
                ref maxKnownA,
                ref word,
                knownOne0,
                knownOne1,
                knownZero0,
                knownZero1,
                windowSize,
                aWordCount,
                aOneWin,
                aZeroWin,
                ref column,
                delta8,
                offsetCount,
                ref firstForced,
                ref firstRemaining,
                ref firstQWordsEff,
                ref firstQLastEff,
                ref firstCarryMin,
                ref firstCarryMax);

			#if DETAILED_LOG
				if (ok)
				{
					int debugBase = slot * DebugWordCountPerSlot;
					for (int w = 0; w < ForcedALowWords; w++)
					{
						debugOut[debugBase + w] = qWords[w];
					}
					for (int w = 0; w < ForcedALowWords; w++)
					{
						debugOut[debugBase + ForcedALowWords + w] = invWords[w];
					}
					for (int w = 0; w < ForcedALowWords; w++)
					{
						debugOut[debugBase + (ForcedALowWords * 2) + w] = aLowWords[w];
					}
					debugOut[debugBase + (ForcedALowWords * 3)] = (ulong)qBitLen;
					int extraBase = debugBase + (ForcedALowWords * 3) + 1;
					debugOut[extraBase] = (ulong)qWordCount;
					debugOut[extraBase + 1] = qMaskWords[0];
					debugOut[extraBase + 2] = qMaskWords[1];
					debugOut[extraBase + 3] = knownOne0[0];
					debugOut[extraBase + 4] = knownZero0[0];
					debugOut[extraBase + 5] = aOneWin[0];
					debugOut[extraBase + 6] = aZeroWin[0];
					debugOut[extraBase + 7] = (ulong)carry.Min;
					debugOut[extraBase + 8] = (ulong)carry.Max;
					debugOut[extraBase + 9] = (ulong)maxAllowedA;
					debugOut[extraBase + 10] = (ulong)maxFixed;
					debugOut[extraBase + 11] = (ulong)windowSize;
					debugOut[extraBase + 12] = (ulong)aWordCount;
					debugOut[extraBase + 13] = (ulong)maxKnownA;
					debugOut[extraBase + 14] = (ulong)maxFixedPrefilter;
					debugOut[extraBase + 15] = preKnownOne0;
					debugOut[extraBase + 16] = preKnownZero0;
					debugOut[extraBase + 17] = preAOneWin0;
					debugOut[extraBase + 18] = preAZeroWin0;
					debugOut[extraBase + 19] = (ulong)preCarryMin;
					debugOut[extraBase + 20] = (ulong)preCarryMax;
					debugOut[extraBase + 21] = (ulong)preSegmentBase;
					debugOut[extraBase + 22] = (ulong)preWindowColumn;
					debugOut[extraBase + 23] = (ulong)offsetCount;
					debugOut[extraBase + 24] = (ulong)maxOffsetValue;
					debugOut[extraBase + 25] = (ulong)firstForced;
					debugOut[extraBase + 26] = (ulong)firstRemaining;
					debugOut[extraBase + 27] = (ulong)firstQWordsEff;
					debugOut[extraBase + 28] = firstQLastEff;
					debugOut[extraBase + 29] = (ulong)firstCarryMin;
					debugOut[extraBase + 30] = (ulong)firstCarryMax;
					foundOut[slot] = batch;
					return;
				}
			#endif
        }
    }

    public static void BitContradictionKernelScan(
        Index1D index,
        ulong exponent,
        ArrayView1D<ulong, Stride1D.Dense> batchIndexWords,
        int countQ,
        ArrayView1D<int, Stride1D.Dense> foundOut,
        ArrayView1D<ulong, Stride1D.Dense> debugOut)
    {
        int slot = index;
        foundOut[slot] = -1;

        if (countQ <= 0)
        {
            return;
        }

        int localCountQ = countQ;
        ulong twoP = exponent << 1;
        int startOffset = slot * MaxQWordCount;

        var baseKWords = LocalMemory.Allocate<ulong>(MaxQWordCount);
        var kWords = LocalMemory.Allocate<ulong>(MaxQWordCount);
        var qWords = LocalMemory.Allocate<ulong>(MaxQWordCount);
        var qOffsets = LocalMemory.Allocate<int>(MaxQOffsets);
        var qPrefix = LocalMemory.Allocate<ulong>(ForcedALowWords);


        var invWords = LocalMemory.Allocate<ulong>(ForcedALowWords);
        var tmpWords = LocalMemory.Allocate<ulong>(ForcedALowWords);
        var tmpWords2 = LocalMemory.Allocate<ulong>(ForcedALowWords);
        var aLowWords = LocalMemory.Allocate<ulong>(ForcedALowWords);
        var delta8 = LocalMemory.Allocate<int>(DeltaLength);

        var knownOne0 = LocalMemory.Allocate<ulong>(MaxAWordCount);
        var knownOne1 = LocalMemory.Allocate<ulong>(MaxAWordCount);
        var knownZero0 = LocalMemory.Allocate<ulong>(MaxAWordCount);
        var knownZero1 = LocalMemory.Allocate<ulong>(MaxAWordCount);
        var qMaskWords = LocalMemory.Allocate<ulong>(MaxQWordCount);
        var aOneWin = LocalMemory.Allocate<ulong>(MaxQWordCount);
        var aZeroWin = LocalMemory.Allocate<ulong>(MaxQWordCount);

        var lo0 = LocalMemory.Allocate<long>(IntervalBufferLength);
        var hi0 = LocalMemory.Allocate<long>(IntervalBufferLength);
        var lo1 = LocalMemory.Allocate<long>(IntervalBufferLength);
        var hi1 = LocalMemory.Allocate<long>(IntervalBufferLength);

        for (int i = 0; i < MaxQWordCount; i++)
        {
            baseKWords[i] = batchIndexWords[startOffset + i];
        }

        if (MultiplyWordsByUInt64(baseKWords, kWords, MaxQWordCount, (ulong)BatchCount) != 0UL)
        {
            return;
        }

        if (AddUInt64ToWords(kWords, MaxQWordCount, 1UL) != 0UL)
        {
            return;
        }

        CopyWords(kWords, baseKWords, MaxQWordCount);

        for (int batch = 0; batch < localCountQ; batch++)
        {
            CopyWords(baseKWords, kWords, MaxQWordCount);
            if (batch != 0)
            {
                if (AddUInt64ToWords(kWords, MaxQWordCount, (ulong)batch) != 0UL)
                {
                    break;
                }
            }

            if (MultiplyWordsByUInt64(kWords, qWords, MaxQWordCount, twoP) != 0UL)
            {
                continue;
            }

            if (AddUInt64ToWords(qWords, MaxQWordCount, 1UL) != 0UL)
            {
                continue;
            }

            int qBitLen = GetBitLength(qWords, MaxQWordCount);
            if (qBitLen <= 0 || qBitLen > MaxQBitLength)
            {
                continue;
            }

            if ((ulong)qBitLen >= exponent)
            {
                continue;
            }

            int pLong = (int)exponent;
            int maxAllowedA = pLong - qBitLen;
            if (maxAllowedA < 0)
            {
                continue;
            }

            int qWordCount = (qBitLen + 63) >> 6;
            int offsetCount;
            int maxOffsetValue = 0;
            if (!BuildQOneOffsetsWords(qWords, qWordCount, qOffsets, out offsetCount))
            {
                continue;
            }

            if (offsetCount <= 0)
            {
				maxOffsetValue = qOffsets[offsetCount - 1];
                continue;
            }

            int windowSize = qBitLen < ForcedALowBits ? ForcedALowBits : qBitLen;
            windowSize = (windowSize + 63) & ~63;
            int aWordCount = (windowSize + 63) >> 6;

            ClearWords(knownOne0, aWordCount);
            ClearWords(knownOne1, aWordCount);
            ClearWords(knownZero0, aWordCount);
            ClearWords(knownZero1, aWordCount);
            ClearWords(qMaskWords, MaxQWordCount);
            ClearWords(aOneWin, MaxQWordCount);
            ClearWords(aZeroWin, MaxQWordCount);

            ClearWords(qPrefix, ForcedALowWords);
            CopyWords(qWords, qPrefix, ForcedALowWords);
            CopyWords(qPrefix, invWords, ForcedALowWords);
            ComputeInverseMod2k(qPrefix, invWords, tmpWords, tmpWords2, ForcedALowWords);
            NegateMod2k(invWords, aLowWords, ForcedALowWords);

            for (int i = 0; i < qWordCount; i++)
            {
                qMaskWords[i] = qWords[i];
            }

            ulong lastWordMask = (qBitLen & 63) == 0 ? ulong.MaxValue : ((1UL << (qBitLen & 63)) - 1UL);
            if (qWordCount > 0)
            {
                qMaskWords[qWordCount - 1] &= lastWordMask;
            }

            int maxFixed = maxAllowedA < (ForcedALowBits - 1) ? maxAllowedA : (ForcedALowBits - 1);
            if (maxFixed >= 0)
            {
                int fullWords = maxFixed >> 6;
                int lastBit = maxFixed & 63;
                for (int w = 0; w < fullWords; w++)
                {
                    ulong wordValue = aLowWords[w];
                    knownOne0[w] = wordValue;
                    knownZero0[w] = ~wordValue;
                }

                if (fullWords < aWordCount)
                {
                    ulong wordValue = aLowWords[fullWords];
                    ulong mask = lastBit == 63 ? ulong.MaxValue : ((1UL << (lastBit + 1)) - 1UL);
                    knownOne0[fullWords] = wordValue & mask;
                    knownZero0[fullWords] = (~wordValue) & mask;
                }
            }

            int maxKnownA = maxFixed >= 0 ? maxFixed : -1;
            int firstForced = -1;
            int firstRemaining = -1;
            int firstQWordsEff = 0;
            ulong firstQLastEff = 0;
            long firstCarryMin = 0;
            long firstCarryMax = 0;

            int maxFixedPrefilter = 512 - 1;
            maxFixedPrefilter = maxFixedPrefilter > maxKnownA ? maxKnownA : maxFixedPrefilter;
            maxFixedPrefilter = maxFixedPrefilter > maxAllowedA ? maxAllowedA : maxFixedPrefilter;

            if (maxFixedPrefilter >= 8)
            {
                // tmpWords is reused for reversePrefix
                ReverseBits1024(qPrefix, tmpWords);

                bool prefixOk = true;
                CarryRange carryLow = CarryRange.Zero;
                for (int column = 0; column <= maxFixedPrefilter; column++)
                {
                    int forced = 0;
                    if (column <= 1023)
                    {
                        int shift = 1023 - column;
                        int wordShift = shift >> 6;
                        int bitShift = shift & 63;
                        for (int word = 0; word < ForcedALowWords; word++)
                        {
                            int src = word + wordShift;
                            ulong value = 0UL;
                            if (src < ForcedALowWords)
                            {
                                value = tmpWords[src] >> bitShift;
                                if (bitShift != 0 && src + 1 < ForcedALowWords)
                                {
                                    value |= tmpWords[src + 1] << (64 - bitShift);
                                }
                            }

                            // tmpWords2 is reused for prefixMask
                            tmpWords2[word] = value;
                        }

                        for (int word = 0; word < ForcedALowWords; word++)
                        {
                            forced += XMath.PopCount(knownOne0[word] & tmpWords2[word]);
                        }
                    }
                    else
                    {
                        for (int i = 0; i < offsetCount; i++)
                        {
                            int t = qOffsets[i];
                            if (t > column)
                            {
                                break;
                            }

                            int aIndex = column - t;
                            int word = aIndex >> 6;
                            ulong mask = 1UL << (aIndex & 63);
                            if ((knownOne0[word] & mask) != 0UL)
                            {
                                forced++;
                            }
                        }
                    }

                    if (!TryPropagateCarry(ref carryLow, forced, forced, 1))
                    {
                        prefixOk = false;
                        break;
                    }
                }

                if (!prefixOk)
                {
                    continue;
                }
            }

            int segmentBase = 0;
            int windowColumn = 0;

            int initialState = GetAKnownStateRowAware(0, segmentBase, windowSize, knownOne0, knownZero0, knownOne1, knownZero1);
            aOneWin[0] |= initialState == 1 ? 1UL : 0UL;
            aZeroWin[0] |= initialState == 2 ? 1UL : 0UL;

            if (!TryRunHighBitAndBorrowPrefiltersCombinedGpu(qOffsets, offsetCount, pLong, maxAllowedA, lo0, hi0, lo1, hi1))
            {
                continue;
            }

            BuildStableUnknownDelta8(offsetCount, delta8);

            CarryRange carry = CarryRange.Zero;
            ulong preKnownOne0 = knownOne0[0];
            ulong preKnownZero0 = knownZero0[0];
            ulong preAOneWin0 = aOneWin[0];
            ulong preAZeroWin0 = aZeroWin[0];
            long preCarryMin = 0L;
            long preCarryMax = 0L;
            int preSegmentBase = segmentBase;
            int preWindowColumn = windowColumn;
            bool ok = TryProcessBottomUpBlockRowAwareGpu(
                qOffsets,
                offsetCount,
                qMaskWords,
                qWordCount,
                lastWordMask,
                maxAllowedA,
                0,
                pLong,
                ref carry,
                ref maxKnownA,
                ref segmentBase,
                knownOne0,
                knownOne1,
                knownZero0,
                knownZero1,
                windowSize,
                aWordCount,
                aOneWin,
                aZeroWin,
                ref windowColumn,
                delta8,
                offsetCount,
                ref firstForced,
                ref firstRemaining,
                ref firstQWordsEff,
                ref firstQLastEff,
                ref firstCarryMin,
                ref firstCarryMax);

            if (ok)
            {
                int debugBase = slot * DebugWordCountPerSlot;
                for (int w = 0; w < ForcedALowWords; w++)
                {
                    debugOut[debugBase + w] = qWords[w];
                }
                for (int w = 0; w < ForcedALowWords; w++)
                {
                    debugOut[debugBase + ForcedALowWords + w] = invWords[w];
                }
                for (int w = 0; w < ForcedALowWords; w++)
                {
                    debugOut[debugBase + (ForcedALowWords * 2) + w] = aLowWords[w];
                }
                debugOut[debugBase + (ForcedALowWords * 3)] = (ulong)qBitLen;
                int extraBase = debugBase + (ForcedALowWords * 3) + 1;
                debugOut[extraBase] = (ulong)qWordCount;
                debugOut[extraBase + 1] = qMaskWords[0];
                debugOut[extraBase + 2] = qMaskWords[1];
                debugOut[extraBase + 3] = knownOne0[0];
                debugOut[extraBase + 4] = knownZero0[0];
                debugOut[extraBase + 5] = aOneWin[0];
                debugOut[extraBase + 6] = aZeroWin[0];
                debugOut[extraBase + 7] = (ulong)carry.Min;
                debugOut[extraBase + 8] = (ulong)carry.Max;
                debugOut[extraBase + 9] = (ulong)maxAllowedA;
                debugOut[extraBase + 10] = (ulong)maxFixed;
                debugOut[extraBase + 11] = (ulong)windowSize;
                debugOut[extraBase + 12] = (ulong)aWordCount;
                debugOut[extraBase + 13] = (ulong)maxKnownA;
                debugOut[extraBase + 14] = (ulong)maxFixedPrefilter;
                debugOut[extraBase + 15] = preKnownOne0;
                debugOut[extraBase + 16] = preKnownZero0;
                debugOut[extraBase + 17] = preAOneWin0;
                debugOut[extraBase + 18] = preAZeroWin0;
                debugOut[extraBase + 19] = (ulong)preCarryMin;
                debugOut[extraBase + 20] = (ulong)preCarryMax;
                debugOut[extraBase + 21] = (ulong)preSegmentBase;
                debugOut[extraBase + 22] = (ulong)preWindowColumn;
                debugOut[extraBase + 23] = (ulong)offsetCount;
                debugOut[extraBase + 24] = (ulong)maxOffsetValue;
                debugOut[extraBase + 25] = (ulong)firstForced;
                debugOut[extraBase + 26] = (ulong)firstRemaining;
                debugOut[extraBase + 27] = (ulong)firstQWordsEff;
                debugOut[extraBase + 28] = firstQLastEff;
                debugOut[extraBase + 29] = (ulong)firstCarryMin;
                debugOut[extraBase + 30] = (ulong)firstCarryMax;
                foundOut[slot] = batch;
                return;
            }
        }
    }

	/// <summary>
	/// This method is used for testing purposes, only
	/// </summary>
    internal static bool TryBuildQWordsFromBatchIndex(
        ulong exponent,
        ReadOnlySpan<ulong> batchIndexWords,
        int wordCount,
        int batchOffset,
        Span<ulong> qWords,
        out int qBitLen)
    {
        Span<ulong> baseKWords = stackalloc ulong[wordCount];
        Span<ulong> kWords = stackalloc ulong[wordCount];

        for (int i = 0; i < wordCount; i++)
        {
            baseKWords[i] = batchIndexWords[i];
        }

        if (MultiplyWordsByUInt64(baseKWords, kWords, wordCount, (ulong)BatchCount) != 0UL)
        {
            qBitLen = 0;
            return false;
        }

        if (AddUInt64ToWords(kWords, wordCount, 1UL) != 0UL)
        {
            qBitLen = 0;
            return false;
        }

        if (batchOffset != 0)
        {
            if (AddUInt64ToWords(kWords, wordCount, (ulong)batchOffset) != 0UL)
            {
                qBitLen = 0;
                return false;
            }
        }

        if (MultiplyWordsByUInt64(kWords, qWords, wordCount, exponent << 1) != 0UL)
        {
            qBitLen = 0;
            return false;
        }

        if (AddUInt64ToWords(qWords, wordCount, 1UL) != 0UL)
        {
            qBitLen = 0;
            return false;
        }

        qBitLen = GetBitLength(qWords, wordCount);
        return true;
    }

    private static ulong AddUInt64ToWords(Span<ulong> words, int wordCount, ulong value)
    {
        ulong carry = value;
        for (int i = 0; i < wordCount; i++)
        {
            if (carry == 0UL)
            {
                return 0UL;
            }

            ulong sum = words[i] + carry;
            words[i] = sum;
            carry = sum < carry ? 1UL : 0UL;
        }

        return carry;
    }

    private static ulong MultiplyWordsByUInt64(ReadOnlySpan<ulong> source, Span<ulong> destination, int wordCount, ulong factor)
    {
        ulong carry = 0UL;
        ulong carryExtra = 0UL;
        for (int i = 0; i < wordCount; i++)
        {
            ulong high;
            ulong low;
            MultiplyPartsGpu(source[i], factor, out high, out low);

            ulong sum = low + carry;
            ulong carryOut = sum < low ? 1UL : 0UL;
            if (carryExtra != 0UL)
            {
                sum += 1UL;
                if (sum == 0UL)
                {
                    carryOut++;
                }
            }

            destination[i] = sum;
            ulong nextCarry = high + carryOut;
            carryExtra = nextCarry < high ? 1UL : 0UL;
            carry = nextCarry;
        }

        return carry + carryExtra;
    }

    private static int GetBitLength(ReadOnlySpan<ulong> words, int wordCount)
    {
        for (int i = wordCount - 1; i >= 0; i--)
        {
            ulong word = words[i];
            if (word != 0UL)
            {
                int leading = XMath.LeadingZeroCount(word);
                return (i << 6) + (64 - leading);
            }
        }

        return 0;
    }

    private static ulong AddUInt64ToWords(ArrayView<ulong> words, int wordCount, ulong value)
    {
        ulong carry = value;
        for (int i = 0; i < wordCount; i++)
        {
            if (carry == 0UL)
            {
                return 0UL;
            }

            ulong sum = words[i] + carry;
            words[i] = sum;
            carry = sum < carry ? 1UL : 0UL;
        }

        return carry;
    }
    private static ulong MultiplyWordsByUInt64(ArrayView<ulong> source, ArrayView<ulong> destination, int wordCount, ulong factor)
    {
        ulong carry = 0UL;
        ulong carryExtra = 0UL;
        for (int i = 0; i < wordCount; i++)
        {
            ulong high;
            ulong low;
            MultiplyPartsGpu(source[i], factor, out high, out low);

            ulong sum = low + carry;
            ulong carryOut = sum < low ? 1UL : 0UL;
            if (carryExtra != 0UL)
            {
                sum += 1UL;
                if (sum == 0UL)
                {
                    carryOut++;
                }
            }

            destination[i] = sum;
            ulong nextCarry = high + carryOut;
            carryExtra = nextCarry < high ? 1UL : 0UL;
            carry = nextCarry;
        }

        return carry + carryExtra;
    }

    private static ulong ReverseBits64(ulong value)
    {
        value = ((value >> 1) & 0x5555555555555555UL) | ((value & 0x5555555555555555UL) << 1);
        value = ((value >> 2) & 0x3333333333333333UL) | ((value & 0x3333333333333333UL) << 2);
        value = ((value >> 4) & 0x0F0F0F0F0F0F0F0FUL) | ((value & 0x0F0F0F0F0F0F0F0FUL) << 4);
        value = ((value >> 8) & 0x00FF00FF00FF00FFUL) | ((value & 0x00FF00FF00FF00FFUL) << 8);
        value = ((value >> 16) & 0x0000FFFF0000FFFFUL) | ((value & 0x0000FFFF0000FFFFUL) << 16);
        value = (value >> 32) | (value << 32);
        return value;
    }

    private static void ReverseBits1024(ArrayView<ulong> prefix, ArrayView<ulong> reversed)
    {
        reversed[0] = ReverseBits64(prefix[15]);
        reversed[1] = ReverseBits64(prefix[14]);
        reversed[2] = ReverseBits64(prefix[13]);
        reversed[3] = ReverseBits64(prefix[12]);
        reversed[4] = ReverseBits64(prefix[11]);
        reversed[5] = ReverseBits64(prefix[10]);
        reversed[6] = ReverseBits64(prefix[9]);
        reversed[7] = ReverseBits64(prefix[8]);
        reversed[8] = ReverseBits64(prefix[7]);
        reversed[9] = ReverseBits64(prefix[6]);
        reversed[10] = ReverseBits64(prefix[5]);
        reversed[11] = ReverseBits64(prefix[4]);
        reversed[12] = ReverseBits64(prefix[3]);
        reversed[13] = ReverseBits64(prefix[2]);
        reversed[14] = ReverseBits64(prefix[1]);
        reversed[15] = ReverseBits64(prefix[0]);
    }

    private static void MultiplyPartsGpu(ulong left, ulong right, out ulong high, out ulong low)
    {
        ulong leftLow = (uint)left;
        ulong leftHigh = left >> 32;
        ulong rightLow = (uint)right;
        ulong rightHigh = right >> 32;

        ulong lowProduct = unchecked(leftLow * rightLow);
        ulong cross1 = unchecked(leftHigh * rightLow);
        ulong cross2 = unchecked(leftLow * rightHigh);
        ulong highProduct = unchecked(leftHigh * rightHigh);

        ulong carry = unchecked((lowProduct >> 32) + (uint)cross1 + (uint)cross2);
        low = unchecked((lowProduct & 0xFFFFFFFFUL) | (carry << 32));
        high = unchecked(highProduct + (cross1 >> 32) + (cross2 >> 32) + (carry >> 32));
    }

    private static int GetBitLength(ArrayView<ulong> words, int wordCount)
    {
        for (int i = wordCount - 1; i >= 0; i--)
        {
            ulong word = words[i];
            if (word != 0UL)
            {
                int leading = XMath.LeadingZeroCount(word);
                return (i << 6) + (64 - leading);
            }
        }

        return 0;
    }

    private static bool BuildQOneOffsetsWords(
        ArrayView<ulong> words,
        int wordCount,
        ArrayView<int> offsets,
        out int offsetCount)
    {
        offsetCount = 0;
        for (int wordIndex = 0; wordIndex < wordCount; wordIndex++)
        {
            ulong word = words[wordIndex];
            while (word != 0UL)
            {
                int bit = XMath.TrailingZeroCount(word);
                if (offsetCount >= MaxQOffsets)
                {
                    return false;
                }

                offsets[offsetCount] = (wordIndex << 6) + bit;
                offsetCount++;
                word &= word - 1UL;
            }
        }

        return true;
    }

    private static void FindBounds(ArrayView<int> values, int valueCount, int lowerTarget, int upperTarget, out int lower, out int upper)
    {
        int lo = 0;
        int hi = valueCount;
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
        hi = valueCount;
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
    private static bool TryRunHighBitAndBorrowPrefiltersCombinedGpu(
        ArrayView<int> qOffsets,
        int qOffsetCount,
        int pLong,
        int maxAllowedA,
        ArrayView<long> lo0,
        ArrayView<long> hi0,
        ArrayView<long> lo1,
        ArrayView<long> hi1)
    {
        int column = pLong - 1;
        int columnPlusOne = column + 1;

        int highBitColumns = HighBitCarryPrefilterColumns;
        highBitColumns = highBitColumns > columnPlusOne ? columnPlusOne : highBitColumns;

        int borrowColumns = 0;
        bool runBorrowPrefilter = qOffsetCount >= TopDownBorrowMinUnknown;
        if (runBorrowPrefilter)
        {
            borrowColumns = TopDownBorrowColumns;
            borrowColumns = borrowColumns > columnPlusOne ? columnPlusOne : borrowColumns;
        }

        int sequentialColumns = highBitColumns > borrowColumns ? highBitColumns : borrowColumns;

        var outLo = lo0;
        var outHi = hi0;
        var inLo = lo1;
        var inHi = hi1;

        int outCount = 1;
        outLo[0] = 0;
        outHi[0] = 0;

        int inCount = 0;
        bool flip = false;
        CarryRange borrow = CarryRange.Zero;

        int step = 0;
        int upperT;
        int lowerT = column - maxAllowedA;
        lowerT = lowerT < 0 ? 0 : lowerT;

        FindBounds(qOffsets, qOffsetCount, lowerT, column, out lowerT, out upperT);
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
                inCount = 0;
                for (int i = 0; i < outCount; i++)
                {
                    long rMin = outLo[i];
                    long rMax = outHi[i];

                    if (rMax < 0) continue;
                    rMin = rMin < 0 ? 0 : rMin;
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
            lowerT = lowerT < 0 ? 0 : lowerT;

            FindBounds(qOffsets, qOffsetCount, lowerT, column, out lowerT, out upperT);
            n = upperT - lowerT;
            if (n < 0)
            {
                return false;
            }
        }

        return true;
    }

    private static void AddInterval(ref int count, ArrayView<long> loArr, ArrayView<long> hiArr, long lo, long hi)
    {
        if (hi < lo)
        {
            return;
        }

        lo = lo < 0 ? 0 : lo;
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
            hiArr[count - 1] = hi > lastHi ? hi : lastHi;
            return;
        }

        if (count >= IntervalBufferLength)
        {
            count = 0;
            return;
        }

        loArr[count] = lo;
        hiArr[count] = hi;
        count++;
    }
    private static void CopyWords(ArrayView<ulong> source, ArrayView<ulong> destination, int wordCount)
    {
        for (int i = 0; i < wordCount; i++)
        {
            destination[i] = source[i];
        }
    }

    private static void ClearWords(ArrayView<ulong> words, int wordCount)
    {
        for (int i = 0; i < wordCount; i++)
        {
            words[i] = 0UL;
        }
    }

    private static void MulMod2k(ArrayView<ulong> left, ArrayView<ulong> right, ArrayView<ulong> result, int wordCount)
    {
        ClearWords(result, wordCount);

        for (int i = 0; i < wordCount; i++)
        {
            ulong carry = 0UL;
            ulong carryExtra = 0UL;
            ulong leftValue = left[i];
            int maxJ = wordCount - i;
            for (int j = 0; j < maxJ; j++)
            {
                int target = i + j;
                ulong high;
                ulong low;
                MultiplyPartsGpu(leftValue, right[j], out high, out low);

                ulong sum = result[target] + low;
                ulong carryOut = sum < low ? 1UL : 0UL;
                sum += carry;
                if (sum < carry)
                {
                    carryOut++;
                }
                if (carryExtra != 0UL)
                {
                    sum += 1UL;
                    if (sum == 0UL)
                    {
                        carryOut++;
                    }
                }

                result[target] = sum;
                ulong nextCarry = high + carryOut;
                carryExtra = nextCarry < high ? 1UL : 0UL;
                carry = nextCarry;
            }
        }
    }

    private static void ComputeInverseMod2k(
        ArrayView<ulong> modulus,
        ArrayView<ulong> inverse,
        ArrayView<ulong> tmp,
        ArrayView<ulong> tmp2,
        int wordCount)
    {
        CopyWords(modulus, inverse, wordCount);

        for (int i = 0; i < InverseIterations; i++)
        {
            MulMod2k(modulus, inverse, tmp, wordCount);
            SubtractFromTwo(tmp, tmp2, wordCount);
            MulMod2k(inverse, tmp2, tmp, wordCount);
            CopyWords(tmp, inverse, wordCount);
        }
    }

    private static void SubtractFromTwo(ArrayView<ulong> value, ArrayView<ulong> result, int wordCount)
    {
        ulong borrow = 0UL;
        for (int i = 0; i < wordCount; i++)
        {
            ulong minuend = i == 0 ? 2UL : 0UL;
            ulong sub = value[i];
            bool hasBorrow = borrow != 0UL;
            ulong res = hasBorrow ? (minuend - sub - 1UL) : (minuend - sub);
            borrow = hasBorrow ? (minuend <= sub ? 1UL : 0UL) : (minuend < sub ? 1UL : 0UL);
            result[i] = res;
        }
    }

    private static void NegateMod2k(ArrayView<ulong> value, ArrayView<ulong> result, int wordCount)
    {
        ulong carry = 1UL;
        for (int i = 0; i < wordCount; i++)
        {
            ulong inverted = ~value[i];
            ulong sum = inverted + carry;
            result[i] = sum;
            carry = sum < inverted ? 1UL : 0UL;
        }
    }
    private static void BuildStableUnknownDelta8(int unknown, ArrayView<int> delta)
    {
        for (int low = 0; low < DeltaLength; low++)
        {
            long value = low;
            for (int i = 0; i < DeltaColumnsAtOnce; i++)
            {
                int sMax = (((unknown ^ (int)value) & 1) != 0) ? unknown : (unknown - 1);
                value = (value + sMax - 1) >> 1;
            }

            delta[low] = (int)((value << DeltaColumnsAtOnce) - low);
        }
    }

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

    private static bool TryAdvanceStableUnknownGpu(ref CarryRange carry, int unknown, int columnCount, ArrayView<int> delta)
    {
        if (columnCount <= 0)
        {
            return true;
        }

        if (unknown <= 0)
        {
            return TryAdvanceZeroColumns(ref carry, columnCount);
        }

        long carryMin = columnCount >= 63 ? 0 : (carry.Min >> columnCount);
        long carryMax = carry.Max;
        int remaining = columnCount;

        const long CarryDecimalMask = (1L << DeltaColumnsAtOnce) - 1L;
        while (remaining >= DeltaColumnsAtOnce)
        {
            int carryMask = (int)(carryMax & CarryDecimalMask);
            carryMax = (carryMax + delta[carryMask]) >> DeltaColumnsAtOnce;
            remaining -= DeltaColumnsAtOnce;
        }

        while (remaining > 0)
        {
            int sMax = (((unknown ^ (int)carryMax) & 1) != 0) ? unknown : unknown - 1;
            carryMax = (carryMax + sMax - 1) >> 1;
            remaining--;
        }

        if (carryMax < carryMin)
        {
            return false;
        }

        carry = new CarryRange(carryMin, carryMax);
        return true;
    }

    private static bool TryPropagateCarry(ref CarryRange carry, int min, int max, int requiredParity)
    {
        long minSum = carry.Min + min;
        long maxSum = carry.Max + max;

        long alignedMin = AlignUpToParity(minSum, requiredParity);
        long alignedMax = AlignDownToParity(maxSum, requiredParity);
        if (alignedMin > alignedMax)
        {
            return false;
        }

        carry.Min = (alignedMin - requiredParity) >> 1;
        carry.Max = (alignedMax - requiredParity) >> 1;
        return true;
    }

    private static long AlignUpToParity(long value, int parity) => (value & 1L) == parity ? value : value + 1L;

    private static long AlignDownToParity(long value, int parity) => (value & 1L) == parity ? value : value - 1L;
    private static void GetQPrefixMask(int startColumn, int qBitLen, int qWordCount, out int words, out ulong lastMask)
    {
        if (startColumn >= qBitLen - 1)
        {
            words = qWordCount;
            lastMask = ulong.MaxValue;
            return;
        }

        if (startColumn < 0)
        {
            words = 0;
            lastMask = 0;
            return;
        }

        int wordIndex = startColumn >> 6;
        words = wordIndex + 1;

        int bit = startColumn & 63;
        lastMask = bit == 63 ? ulong.MaxValue : ((1UL << (bit + 1)) - 1UL);
    }

    private static int GetAKnownStateRowAware(
        int aIndex,
        int segmentBase,
        int windowSize,
        ArrayView<ulong> knownOne0,
        ArrayView<ulong> knownZero0,
        ArrayView<ulong> knownOne1,
        ArrayView<ulong> knownZero1)
    {
        if (aIndex < 0)
        {
            return 2;
        }

        int rel;
        int word;
        ulong mask;
        if (aIndex >= segmentBase && aIndex < segmentBase + windowSize)
        {
            rel = aIndex - segmentBase;
            word = rel >> 6;
            mask = 1UL << (rel & 63);
            if ((knownOne0[word] & mask) != 0UL) return 1;
            if ((knownZero0[word] & mask) != 0UL) return 2;
            return 0;
        }

        int segment1Start = segmentBase - windowSize;
        if (aIndex >= segment1Start && aIndex < segmentBase)
        {
            rel = aIndex - segment1Start;
            word = rel >> 6;
            mask = 1UL << (rel & 63);
            if ((knownOne1[word] & mask) != 0UL) return 1;
            if ((knownZero1[word] & mask) != 0UL) return 2;
            return 0;
        }

        return 0;
    }

    private static void SetAKnownRowAware(
        int aIndex,
        bool isOne,
        int segmentBase,
        int windowSize,
        ArrayView<ulong> knownOne0,
        ArrayView<ulong> knownZero0,
        ArrayView<ulong> knownOne1,
        ArrayView<ulong> knownZero1)
    {
        if (aIndex < 0)
        {
            return;
        }

        int rel;
        int word;
        ulong mask;
        if (aIndex >= segmentBase && aIndex < segmentBase + windowSize)
        {
            rel = aIndex - segmentBase;
            word = rel >> 6;
            mask = 1UL << (rel & 63);
            knownOne0[word] = isOne ? (knownOne0[word] | mask) : (knownOne0[word] & ~mask);
            knownZero0[word] = isOne ? (knownZero0[word] & ~mask) : (knownZero0[word] | mask);

            return;
        }

        int segment1Start = segmentBase - windowSize;
        if (aIndex >= segment1Start && aIndex < segmentBase)
        {
            rel = aIndex - segment1Start;
            word = rel >> 6;
            mask = 1UL << (rel & 63);
            knownOne1[word] = isOne ? (knownOne1[word] | mask) : (knownOne1[word] & ~mask);
            knownZero1[word] = isOne ? (knownZero1[word] & ~mask) : (knownZero1[word] | mask);
        }
    }

    private static void ShiftLeft1InPlace(ArrayView<ulong> wordsOne, ArrayView<ulong> wordsZero, int wordCount, ulong lastWordMask)
    {
        ulong carryOne = 0UL;
        ulong carryZero = 0UL;

        for (int i = 0; i < wordCount; i++)
        {
            ulong word = wordsOne[i];
            ulong newCarry = word >> 63;
            wordsOne[i] = (word << 1) | carryOne;
            carryOne = newCarry;

            word = wordsZero[i];
            newCarry = word >> 63;
            wordsZero[i] = (word << 1) | carryZero;
            carryZero = newCarry;
        }

        if (wordCount > 0)
        {
            int last = wordCount - 1;
            wordsOne[last] &= lastWordMask;
            wordsZero[last] &= lastWordMask;
        }
    }

    private static void ComputeForcedRemainingAndCleanPrefix(
        ArrayView<ulong> qMask,
        ArrayView<ulong> oneWin,
        ArrayView<ulong> zeroWin,
        int wordCount,
        ulong lastMask,
        out int forced,
        out int remaining)
    {
        forced = 0;
        remaining = 0;

        int lastIndex = wordCount - 1;
        for (int i = 0; i < wordCount; i++)
        {
            ulong q = qMask[i];
            if (i == lastIndex)
            {
                q &= lastMask;
            }

            ulong one = oneWin[i] & q;
            ulong known = (oneWin[i] | zeroWin[i]);
            ulong unknown = q & ~known;

            forced += XMath.PopCount(one);
            remaining += XMath.PopCount(unknown);
        }
    }

    private static bool TryFindSingleUnknownInQPrefix(
        ArrayView<ulong> qMask,
        ArrayView<ulong> oneWin,
        ArrayView<ulong> zeroWin,
        int wordCount,
        ulong lastMask,
        out int offset)
    {
        offset = -1;
        int found = 0;
        int lastIndex = wordCount - 1;
        for (int w = 0; w < wordCount; w++)
        {
            ulong q = qMask[w];
            if (w == lastIndex)
            {
                q &= lastMask;
            }

            ulong known = oneWin[w] | zeroWin[w];
            ulong unknown = q & ~known;
            if (unknown == 0UL)
            {
                continue;
            }

            found += XMath.PopCount(unknown);
            if (found > 1)
            {
                return false;
            }

            int bit = XMath.TrailingZeroCount(unknown);
            offset = (w << 6) + bit;
        }

        return found == 1;
    }
    private static bool TryProcessBottomUpBlockRowAwareGpu(
        ArrayView<int> qOffsets,
        int qOffsetCount,
        ArrayView<ulong> qMaskWords,
        int qWordCount,
        ulong lastWordMask,
        int maxAllowedA,
        int startColumn,
        int columnCount,
        ref CarryRange carry,
        ref int maxKnownA,
        ref int segmentBase,
        ArrayView<ulong> knownOne0,
        ArrayView<ulong> knownOne1,
        ArrayView<ulong> knownZero0,
        ArrayView<ulong> knownZero1,
        int windowSize,
        int wordCount,
        ArrayView<ulong> aOneWin,
        ArrayView<ulong> aZeroWin,
        ref int windowColumn,
        ArrayView<int> stableUnknownDelta,
        int stableUnknown,
        ref int firstForced,
        ref int firstRemaining,
        ref int firstQWordsEff,
        ref ulong firstQLastEff,
        ref long firstCarryMin,
        ref long firstCarryMax)
    {
        bool needRebuildWindow = false;
        int endColumn = startColumn + columnCount;
        int endColumnMinusOne = endColumn - 1;
        int maxOffset = qOffsets[qOffsetCount - 1];
        int maxOffsetPlusOne = maxOffset + 1;

        if (windowColumn != startColumn)
        {
            return false;
        }

        int qWordCountMinusOne = qWordCount - 1;

        while (startColumn < endColumn)
        {
            int bound = (startColumn / windowSize) * windowSize;
            if (bound != segmentBase)
            {
                if (bound == segmentBase + windowSize)
                {
                    var temp = knownOne0;
                    knownOne0 = knownOne1;
                    knownOne1 = temp;
                    temp = knownZero0;
                    knownZero0 = knownZero1;
                    knownZero1 = temp;
                    ClearWords(knownOne0, wordCount);
                    ClearWords(knownZero0, wordCount);
                }
                else
                {
                    ClearWords(knownOne0, wordCount);
                    ClearWords(knownOne1, wordCount);
                    ClearWords(knownZero0, wordCount);
                    ClearWords(knownZero1, wordCount);
                    needRebuildWindow = true;
                }

                segmentBase = bound;
                if (needRebuildWindow)
                {
                    ClearWords(aOneWin, qWordCount);
                    ClearWords(aZeroWin, qWordCount);

                    for (int step = 0; step <= maxOffset; step++)
                    {
                        int aIndex = startColumn - step;
                        int state = aIndex < 0 || aIndex > maxAllowedA
                            ? 2
                            : GetAKnownStateRowAware(aIndex, segmentBase, windowSize, knownOne0, knownZero0, knownOne1, knownZero1);

                        ulong stepMask = 1UL << (step & 63);
                        aOneWin[step >> 6] |= state == 1 ? stepMask : 0UL;
                        aZeroWin[step >> 6] |= state == 2 ? stepMask : 0UL;
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

            if (startColumn >= maxOffset && startColumn <= maxAllowedA)
            {
                int remaining = endColumnMinusOne;
                if (remaining > maxAllowedA)
                {
                    remaining = maxAllowedA;
                }

                remaining = remaining - startColumn + 1;
                if (remaining > 0)
                {
                    while (remaining >= TailCarryBatchColumns)
                    {
                        if (!TryAdvanceStableUnknownGpu(ref carry, stableUnknown, TailCarryBatchColumns, stableUnknownDelta))
                        {
                            return false;
                        }

                        startColumn += TailCarryBatchColumns;
                        windowColumn = startColumn;
                        remaining -= TailCarryBatchColumns;
                    }

                    if (remaining > 0)
                    {
                        if (!TryAdvanceStableUnknownGpu(ref carry, stableUnknown, remaining, stableUnknownDelta))
                        {
                            return false;
                        }

                        startColumn += remaining;
                        windowColumn = startColumn;
                    }

                    ClearWords(aOneWin, qWordCount);
                    ClearWords(aZeroWin, qWordCount);
                    continue;
                }
            }

            ComputeForcedRemainingAndCleanPrefix(qMaskWords, aOneWin, aZeroWin, qWordsEff, qLastEff, out int forced, out int remainingCols);

            if (remainingCols == 1 && ((carry.Min ^ carry.Max) & 1L) == 0)
            {
                int step;
                if (TryFindSingleUnknownInQPrefix(qMaskWords, aOneWin, aZeroWin, qWordsEff, qLastEff, out step))
                {
                    int aIndex = startColumn - step;
                    const int requiredBit = 1;
                    int parity = (int)(carry.Min & 1L);
                    int needOne = (requiredBit ^ parity ^ (forced & 1)) & 1;

                    int state = GetAKnownStateRowAware(aIndex, segmentBase, windowSize, knownOne0, knownZero0, knownOne1, knownZero1);
                    if (needOne != 0)
                    {
                        if (state == 2)
                        {
                            return false;
                        }

                        SetAKnownRowAware(aIndex, true, segmentBase, windowSize, knownOne0, knownZero0, knownOne1, knownZero1);
                        int w = step >> 6;
                        int b = step & 63;
                        aOneWin[w] |= 1UL << b;
                        forced++;
                    }
                    else
                    {
                        if (state == 1)
                        {
                            return false;
                        }

                        SetAKnownRowAware(aIndex, false, segmentBase, windowSize, knownOne0, knownZero0, knownOne1, knownZero1);
                        int w = step >> 6;
                        int b = step & 63;
                        aZeroWin[w] |= 1UL << b;
                    }

                    remainingCols = 0;
                    if (aIndex > maxKnownA)
                    {
                        maxKnownA = aIndex;
                    }
                }
            }

            int max = forced + remainingCols;
            if (!TryPropagateCarry(ref carry, forced, max, 1))
            if (startColumn == 0 && firstForced < 0)
            {
                firstForced = forced;
                firstRemaining = remainingCols;
                firstQWordsEff = qWordsEff;
                firstQLastEff = qLastEff;
                firstCarryMin = carry.Min;
                firstCarryMax = carry.Max;
            }
            {
                return false;
            }

            int nextA = startColumn + 1;
            ShiftLeft1InPlace(aOneWin, aZeroWin, qWordCount, lastWordMask);

            int nextState = nextA > maxAllowedA
                ? 2
                : GetAKnownStateRowAware(nextA, segmentBase, windowSize, knownOne0, knownZero0, knownOne1, knownZero1);

            aOneWin[0] |= nextState == 1 ? 1UL : 0UL;
            aZeroWin[0] |= nextState == 2 ? 1UL : 0UL;

            startColumn++;
            windowColumn = startColumn;
        }

        return true;
    }
}


