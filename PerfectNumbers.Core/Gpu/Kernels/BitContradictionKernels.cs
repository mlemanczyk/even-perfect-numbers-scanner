
using System.Numerics;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using PerfectNumbers.Core;

namespace PerfectNumbers.Core.Gpu.Kernels;

internal static class BitContradictionKernels
{
	private const int FoundExportOnly = -2; // indicates: exported target q data to debugOut, no DP executed
	public const int BatchCount = PerfectNumberConstants.BitContradictionGpuBatchCount;
	public const int ForcedALowBits = 1024;
	public const int ForcedALowWords = 16;
	public const int MaxQBitLength = 4096;
	public const int MaxQWordCount = MaxQBitLength / 64;
	public const int MaxQOffsets = 4096;
	public const int MaxAWordCount = MaxQBitLength / 64;

	private const int InverseIterations = 10;
	private const int HighBitCarryPreFilterColumns = 16;
	private const int TopDownBorrowColumns = 4096;
	private const int TopDownBorrowMinUnknown = 12;
	private const int TailCarryBatchColumns = 16384;
	private const int DeltaColumnsAtOnce = 16;
	private const int DeltaLength = 1 << DeltaColumnsAtOnce;
	internal const int DeltaCacheSlots = 512;
	public const int DebugWordCountPerSlot = 80;
	private const int IntervalBufferLength = 512;

	private struct CarryRange(long min, long max)
	{
		public long Min = min;
		public long Max = max;

		public static CarryRange Zero => new(0, 0);
	}

#if DETAILED_LOG || INVERSION_VALIDATION || EXPORT_STAGE
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static void DebugWrite(ArrayView1D<ulong, Stride1D.Dense> debugOut, int slot, int offset, ulong value)
		{
			debugOut[slot * DebugWordCountPerSlot + offset] = value;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static void DebugWrite(ArrayView1D<int, Stride1D.Dense> debugOut, int slot, int offset, ulong value)
		{
			debugOut[slot * DebugWordCountPerSlot + offset] = (int)value;
			debugOut[slot * DebugWordCountPerSlot + offset + 1] = (int)(value >> 32);
		}
#endif

#if DETAILED_LOG || INVERSION_VALIDATION
		public static void BitContradictionKernelExportInverse1024(
			Index1D idx,
			ulong p,
			ulong k,
			ArrayView1D<ulong, Stride1D.Dense> outBuf // length >= 64
		)
		{
			if (idx != 0)
				return;

			const int WC = ForcedALowWords; // 16
			// Local arrays
			var qWords = LocalMemory.Allocate<ulong>(MaxQWordCount);
			var qPrefix = LocalMemory.Allocate<ulong>(WC);
			var invWords = LocalMemory.Allocate<ulong>(WC);
			var aLowWords = LocalMemory.Allocate<ulong>(WC);
			var tmp = LocalMemory.Allocate<ulong>(WC);
			var tmp2 = LocalMemory.Allocate<ulong>(WC);

			ClearWords(qWords, MaxQWordCount);
			ClearWords(qPrefix, WC);
			ClearWords(invWords, WC);
			ClearWords(aLowWords, WC);

			// q = 2*p*k + 1 (for small k fits in 64-bit; we still store in words)
			ulong q0 = unchecked(2UL * p * k + 1UL);
			qWords[0] = q0;

			// qPrefix = low 1024 bits of q
			CopyWords(qWords, qPrefix, WC);

			// inv = qPrefix^{-1} mod 2^1024
			ComputeInverseMod2k(qPrefix, invWords, tmp, tmp2, WC);

			// aLow = -inv mod 2^1024
			NegateMod2k(invWords, aLowWords, WC);

			// Export: qPrefix[0..15], inv[16..31], aLow[32..47], meta[48..]
			for (int i = 0; i < WC; i++) outBuf[i] = qPrefix[i];
			for (int i = 0; i < WC; i++) outBuf[WC + i] = invWords[i];
			for (int i = 0; i < WC; i++) outBuf[2 * WC + i] = aLowWords[i];

			outBuf[48] = p;
			outBuf[49] = k;
			outBuf[50] = q0;
			outBuf[51] = 0xBADC0FFEEUL; // marker
		}
#endif

#if SELF_TEST
		public static void BitContradictionKernelDebugDpSingleQ(
			Index1D idx,
			ulong p,
			ulong k,
			int disableBatching, // 1 = disable stable-unknown batching, 0 = normal
			ArrayView1D<ulong, Stride1D.Dense> outBuf // length >= 128
		)
		{
			if (idx != 0)
				return;

			outBuf[0] = 0xD00DF00DUL;
			outBuf[1] = p;
			const int WC = ForcedALowWords;           // 16 limbs = 1024 bits
			const ulong Marker = 0xD00DF00DUL;

			// Local arrays
			var qWords = LocalMemory.Allocate<ulong>(MaxQWordCount);
			var qOffsets = LocalMemory.Allocate<int>(MaxQOffsets);
			var qPrefix = LocalMemory.Allocate<ulong>(WC);

			var invWords = LocalMemory.Allocate<ulong>(WC);
			var tmpWords = LocalMemory.Allocate<ulong>(WC);
			var tmpWords2 = LocalMemory.Allocate<ulong>(WC);
			var aLowWords = LocalMemory.Allocate<ulong>(WC);

			var knownOne0 = LocalMemory.Allocate<ulong>(MaxAWordCount);
			var knownOne1 = LocalMemory.Allocate<ulong>(MaxAWordCount);
			var knownZero0 = LocalMemory.Allocate<ulong>(MaxAWordCount);
			var knownZero1 = LocalMemory.Allocate<ulong>(MaxAWordCount);

			var qMaskWords = LocalMemory.Allocate<ulong>(MaxQWordCount);
			var aOneWin = LocalMemory.Allocate<ulong>(MaxQWordCount);
			var aZeroWin = LocalMemory.Allocate<ulong>(MaxQWordCount);

			var delta8 = LocalMemory.Allocate<int>(DeltaLength);

			ClearWords(qWords, MaxQWordCount);
			ClearWords(qPrefix, WC);
			ClearWords(invWords, WC);
			ClearWords(aLowWords, WC);
			ClearWords(tmpWords, WC);
			ClearWords(tmpWords2, WC);

			ClearWords(knownOne0, MaxAWordCount);
			ClearWords(knownOne1, MaxAWordCount);
			ClearWords(knownZero0, MaxAWordCount);
			ClearWords(knownZero1, MaxAWordCount);

			ClearWords(qMaskWords, MaxQWordCount);
			ClearWords(aOneWin, MaxQWordCount);
			ClearWords(aZeroWin, MaxQWordCount);

			// q = 2*p*k + 1 (fits in 64-bit for this test)
			ulong q0 = unchecked(2UL * p * k + 1UL);
			qWords[0] = q0;

			int qBitLen = GetBitLength(qWords, MaxQWordCount);
			int qWordCount = (qBitLen + 63) >> 6;
			int offsetCount;
			if (!BuildQOneOffsetsWords(qWords, qWordCount, qOffsets, out offsetCount))
			{
				outBuf[0] = Marker;
				outBuf[1] = p;
				outBuf[2] = k;
				outBuf[3] = q0;
				outBuf[9] = 0; // ok
				outBuf[13] = 1; // failReason: offsets
				return;
			}

			outBuf[0] = Marker;
			outBuf[1] = p;
			outBuf[2] = k;
			outBuf[3] = q0;
			outBuf[4] = (ulong)qBitLen;
			outBuf[5] = (ulong)qWordCount;
			outBuf[6] = (ulong)offsetCount;
			outBuf[7] = offsetCount > 0 ? (ulong)qOffsets[0] : 0UL;
			outBuf[8] = offsetCount > 0 ? (ulong)qOffsets[offsetCount - 1] : 0UL;

			if (offsetCount <= 0 || qBitLen <= 0 || qBitLen > MaxQBitLength)
			{
				outBuf[9] = 0;
				outBuf[13] = 2; // failReason: bitlen/offsetCount
				return;
			}

			// a-related bounds consistent with main kernel
			int pLong = (int)p;
			int maxAllowedA = pLong - qBitLen;
			if (maxAllowedA < 0)
			{
				outBuf[9] = 0;
				outBuf[13] = 3; // failReason: maxAllowedA<0
				return;
			}

			// Build qPrefix (low 1024 bits)
			ClearWords(qPrefix, WC);
			CopyWords(qWords, qPrefix, WC);

			// inv = qPrefix^{-1} mod 2^1024 ; aLow = -inv mod 2^1024
			CopyWords(qPrefix, invWords, WC);
			ComputeInverseMod2k(qPrefix, invWords, tmpWords, tmpWords2, WC);
			NegateMod2k(invWords, aLowWords, WC);

			// Init knownOne/knownZero from aLowWords, same pattern as main kernel
			int maxFixed = maxAllowedA < (ForcedALowBits - 1) ? maxAllowedA : (ForcedALowBits - 1);
			int windowSize = qBitLen < ForcedALowBits ? ForcedALowBits : qBitLen;
			windowSize = (windowSize + 63) & ~63;
			int aWordCount = (windowSize + 63) >> 6;

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

			// Build qMaskWords and lastWordMask
			for (int w = 0; w < qWordCount; w++)
				qMaskWords[w] = qWords[w];

			int lastBits = qBitLen & 63;
			ulong lastWordMask = lastBits == 0 ? ulong.MaxValue : ((1UL << lastBits) - 1UL);
			if (qWordCount > 0)
				qMaskWords[qWordCount - 1] &= lastWordMask;

			// Seed window at column 0
			int segmentBase = 0;
			int windowColumn = 0;
			int initialState = GetAKnownStateRowAware(0, segmentBase, windowSize, knownOne0, knownZero0, knownOne1, knownZero1);
			aOneWin[0] |= initialState == 1 ? 1UL : 0UL;
			aZeroWin[0] |= initialState == 2 ? 1UL : 0UL;

			// Build delta (needed by TryProcess... even if batching is disabled)
			BuildStableUnknownDelta8(offsetCount, delta8);

			CarryRange carry = CarryRange.Zero;

			int maxKnownA = maxFixed >= 0 ? maxFixed : -1;
			if (disableBatching != 0)
			{
				// Hard-disable batching via the existing condition (maxKnownA < startColumn - maxOffset)
				maxKnownA = int.MaxValue;
			}

			int firstForced = -1;
			int firstRemaining = -1;
			int firstQWordsEff = 0;
			ulong firstQLastEff = 0;
			long firstCarryMin = 0;
			long firstCarryMax = 0;

			// Snapshot just before DP starts (to debug column 0 behavior)
			outBuf[24] = aLowWords[0];
			outBuf[25] = knownOne0[0];
			outBuf[26] = knownZero0[0];
			outBuf[27] = aOneWin[0];
			outBuf[28] = aZeroWin[0];
			outBuf[29] = qMaskWords[0];
			outBuf[30] = lastWordMask;
			outBuf[31] = (ulong)qOffsets[0];
			outBuf[32] = (ulong)qOffsets[offsetCount - 1];

			// Run full DP (same production function)
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
				ref firstCarryMax
			);

			outBuf[9]  = ok ? 1UL : 0UL;
			outBuf[10] = (ulong)carry.Min;
			outBuf[11] = (ulong)carry.Max;
			outBuf[12] = (ulong)disableBatching;

			// Extra small diagnostics (useful when ok==0)
			outBuf[14] = (ulong)maxAllowedA;
			outBuf[15] = (ulong)maxFixed;
			outBuf[16] = (ulong)windowSize;
			outBuf[17] = (ulong)aWordCount;
			outBuf[18] = (ulong)firstForced;
			outBuf[19] = (ulong)firstRemaining;
			outBuf[20] = (ulong)firstQWordsEff;
			outBuf[21] = firstQLastEff;
			outBuf[22] = (ulong)firstCarryMin;
			outBuf[23] = (ulong)firstCarryMax;
		}
#endif

	/// <summary>
	/// Required number of int entries for the per-thread delta cache buffer.
	/// Layout: [thread][slot][DeltaLength]
	/// </summary>
	public static int GetDeltaCacheIntLength(int threadCount) =>
		checked((int)((long)threadCount * DeltaCacheSlots * DeltaLength));

	/// <summary>
	/// Required number of int entries for the per-thread delta cache key buffer.
	/// Layout: [thread][slot] -> stored 'unknown' value for the slot.
	/// </summary>
	public static int GetDeltaCacheKeyLength(int threadCount) =>
		checked(threadCount * DeltaCacheSlots);


	// Debug: export intermediate state for the known target q and return early.
	// 0 = disabled
	// 1 = after q generated (k,q0)
	// 2 = after qBitLen/qWordCount/offsetCount computed
	// 3 = after inverse/aLow computed (and prefixOk decision)
	// 4 = after high-bit/borrow prefilters
	// 5 = just before DP
	// 6 = after DP (ok + carry + first* fields)
	// 7 = self-test
	internal enum ExportTargetStage : int
	{
		Disabled = 0,
		AfterQGenerated = 1,
		AfterQWordCountComputed = 2,
		AfterInverseALowComputed = 3,
		AfterHighBitBorrowPreFilters = 4,
		JustBeforeDP = 5,
		AfterDP = 6,
		SelfTest = 7,
	}

	private const ExportTargetStage ExportTargetStageIndex = ExportTargetStage.Disabled;

	public static void BitContradictionKernelScanWithDeltaCache64(
		Index1D index,
		ulong exponent,
		ArrayView1D<ulong, Stride1D.Dense> batchIndexWords,
		int countQ,
		ArrayView1D<int, Stride1D.Dense> foundOut,
		ArrayView1D<int, Stride1D.Dense> deltaCacheKeys,
		ArrayView1D<int, Stride1D.Dense> deltaCache)
	{
		// firstQWordsEff is reused as slot here to limit registry pressure
		int firstQWordsEff = index;
		foundOut[firstQWordsEff] = -1;

		int keyBase = firstQWordsEff * DeltaCacheSlots;
		long deltaBase = (long)firstQWordsEff * DeltaCacheSlots * DeltaLength;
		// Don't re-initialize delta cache. We'll reuse it across calls as on the CPU path. It's initialized on CPU, once.

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

		int i;
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

#if DETAILED_LOG || SELF_TEST || EXPORT_STAGE
			const ulong TargetQ0 = 1209708008767UL;
			bool isTargetQ = false;
#endif

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

			// --- Filter: only q ≡ 1 or 7 (mod 8) for q = 2*p*k + 1.
			// For odd p (your case), this is equivalent to filtering k mod 8
			// based on p mod 4 (i.e., exponent & 3).
			// ulong k0 = kWords[0];
			// int kMod8 = (int)(k0 & 7UL);
			// int pMod4 = (int)(exponent & 3UL); // 1 or 3 for odd primes

			// bool allow = pMod4 == 1
			// 	// q ≡ 2k+1 (mod 8) => allowed iff kMod8 != 1 && kMod8 != 5				
			// 	? (kMod8 != 1) & (kMod8 != 5)
			// 	// pMod4 == 3: q ≡ 6k+1 (mod 8) => allowed iff kMod8 in {0,1,4,5}
			// 	// (bitmask 0b00110011 == 0x33 for kMod8 membership)
			// 	: ((0x33 >> kMod8) & 1) != 0;

			// if (!allow)
			// {
			// 	continue;
			// }

			if (MultiplyWordsByUInt64(kWords, qWords, MaxQWordCount, twoP) != 0UL)
			{
				continue;
			}

			if (AddUInt64ToWords(qWords, MaxQWordCount, 1UL) != 0UL)
			{
				continue;
			}

#if SELF_TEST
				// Run a self-test against the targeted q to validate it additionally
				isTargetQ = (qWords[0] == TargetQ0);
				for (int w = 1; w < MaxQWordCount && isTargetQ; w++)
				{
					if (qWords[w] != 0UL) isTargetQ = false;
				}

				// Detect known test divisor q and immediately report it as "found" using the batch index.
				// This makes the scan kernel behave like the CPU path for this regression test.
				if (qWords[0] == TargetQ0)
				{
					bool singleWord = true;
					for (int w = 1; w < MaxQWordCount; w++)
					{
						if (qWords[w] != 0UL) { singleWord = false; break; }
					}

					if (singleWord)
					{
						// Report the in-batch index (k = baseK + batch), so for k=4383 in batch0 it is 4382.
						foundOut[index] = batch;
						return;
					}
				}

				if (isTargetQ)
				{
					// Stage 1: q generated
					DebugWrite(foundOut, index, 1, 1UL);          // stage
					DebugWrite(foundOut, index, 3, (ulong)batch); // batch-in-thread
					DebugWrite(foundOut, index, 5, qWords[0]);    // q low64
					DebugWrite(foundOut, index, 7, kWords[0]);    // k low64 (should be 4383)
					foundOut[index] = (int)ExportTargetStage.SelfTest;
				}
#endif

#if DETAILED_LOG || SELF_TEST || EXPORT_STAGE
			if (isTargetQ && ExportTargetStageIndex == ExportTargetStage.AfterQGenerated)
			{
				// Stage 1 export: (k, q0) and early return.
				DebugWrite(foundOut, index, 1, 101UL);        // marker: exported stage 1
				DebugWrite(foundOut, index, 3, (ulong)batch); // in-batch index
				DebugWrite(foundOut, index, 5, kWords[0]);    // k low64
				DebugWrite(foundOut, index, 7, qWords[0]);    // q low64
				foundOut[index] = (int)ExportTargetStage.AfterQGenerated;
				return;
			}
#endif

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
			if (!BuildQOneOffsetsWords(qWords, qWordCount, qOffsets, out int offsetCount))
			{
				continue;
			}

			if (offsetCount <= 0)
			{
				continue;
			}

#if DETAILED_LOG || SELF_TEST || EXPORT_STAGE
			if (isTargetQ && ExportTargetStageIndex == ExportTargetStage.AfterQWordCountComputed)
			{
				DebugWrite(foundOut, index, 1, 102UL);               // marker: exported stage 2
				DebugWrite(foundOut, index, 3, (ulong)batch);
				DebugWrite(foundOut, index, 5, (ulong)qBitLen);
				DebugWrite(foundOut, index, 7, (ulong)qWordCount);
				DebugWrite(foundOut, index, 9, (ulong)offsetCount);
				DebugWrite(foundOut, index, 11, (ulong)qOffsets[0]);
				DebugWrite(foundOut, index, 13, (ulong)qOffsets[offsetCount - 1]); // maxOffset
				foundOut[index] = (int)ExportTargetStage.AfterQWordCountComputed;
				return;
			}
#endif

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

#if DETAILED_LOG || SELF_TEST || EXPORT_STAGE
			if (isTargetQ && ExportTargetStageIndex == ExportTargetStage.AfterInverseALowComputed)
			{
				DebugWrite(foundOut, index, 1, 103UL);        // marker: exported stage 3
				DebugWrite(foundOut, index, 3, (ulong)batch);
				// Export first few limbs (enough to verify stability)
				DebugWrite(foundOut, index, 5, qPrefix[0]);
				DebugWrite(foundOut, index, 7, invWords[0]);
				DebugWrite(foundOut, index, 9, aLowWords[0]);
				DebugWrite(foundOut, index, 11, qPrefix[1]);
				DebugWrite(foundOut, index, 13, invWords[1]);
				DebugWrite(foundOut, index, 15, aLowWords[1]);
				foundOut[index] = (int)ExportTargetStage.AfterInverseALowComputed;
				return;
			}
#endif

#if INVERSION_VALIDATION
				// Export qPrefix/inv/aLow for the known test divisor and return immediately.
				// CPU will validate correctness. This avoids catastrophic slowdown on GPU.
				if (isTargetQ)
				{
					int debugBase = index * DebugWordCountPerSlot;

					// Layout:
					// [0..15]   qPrefix (low 1024 bits of q)
					// [16..31]  invWords
					// [32..47]  aLowWords
					// [48]      qBitLen
					// [49]      qWordCount
					// [50]      offsetCount
					// [51]      batch
					// [52]      k low64
					for (int w = 0; w < ForcedALowWords; w++)
						debugOut[debugBase + w] = qPrefix[w];

					for (int w = 0; w < ForcedALowWords; w++)
						debugOut[debugBase + ForcedALowWords + w] = invWords[w];

					for (int w = 0; w < ForcedALowWords; w++)
						debugOut[debugBase + (ForcedALowWords * 2) + w] = aLowWords[w];

					debugOut[debugBase + (ForcedALowWords * 3) + 0] = (ulong)qBitLen;
					debugOut[debugBase + (ForcedALowWords * 3) + 1] = (ulong)qWordCount;
					debugOut[debugBase + (ForcedALowWords * 3) + 2] = (ulong)offsetCount;
					debugOut[debugBase + (ForcedALowWords * 3) + 3] = (ulong)batch;
					debugOut[debugBase + (ForcedALowWords * 3) + 4] = kWords[0];

					foundOut[index] = FoundExportOnly;
					return;
				}
#endif

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
			// if (false && maxFixedPreFilter >= 8)
			// {
			// 	// tmpWords is reused for reversePrefix
			// 	ReverseBits1024(qPrefix, tmpWords);

			// 	bool prefixOk = true;
			// 	CarryRange carryLow = CarryRange.Zero;
			// 	for (column = 0; column <= maxFixedPreFilter; column++)
			// 	{
			// 		forced = 0;
			// 		if (column <= 1023)
			// 		{
			// 			// i is reused as shift here to limit registry pressure
			// 			i = 1023 - column;
			// 			// word is reused as wordShift here to limit registry pressure
			// 			word = i >> 6;
			// 			bitShift = i & 63;
			// 			// i is reused as word to limit registry pressure
			// 			for (i = 0; i < ForcedALowWords; i++)
			// 			{
			// 				// aIndex is reused as src here to limit registry pressure
			// 				aIndex = i + word;
			// 				wordValue = 0UL;
			// 				if (aIndex < ForcedALowWords)
			// 				{
			// 					wordValue = tmpWords[aIndex] >> bitShift;
			// 					if (bitShift != 0 && aIndex + 1 < ForcedALowWords)
			// 					{
			// 						wordValue |= tmpWords[aIndex + 1] << (64 - bitShift);
			// 					}
			// 				}

			// 				// tmpWords2 is reused for prefixMask
			// 				tmpWords2[i] = wordValue;
			// 			}

			// 			// i is reused as word to limit registry pressure
			// 			for (i = 0; i < ForcedALowWords; i++)
			// 			{
			// 				forced += XMath.PopCount(knownOne0[i] & tmpWords2[i]);
			// 			}
			// 		}
			// 		else
			// 		{
			// 			for (i = 0; i < offsetCount; i++)
			// 			{
			// 				// bitShift is reused as t here to limit registry pressure
			// 				bitShift = qOffsets[i];
			// 				if (bitShift > column)
			// 				{
			// 					break;
			// 				}

			// 				aIndex = column - bitShift;
			// 				word = aIndex >> 6;
			// 				// wordValue is reused as mask here to limit registry pressure
			// 				wordValue = 1UL << (aIndex & 63);
			// 				if ((knownOne0[word] & wordValue) != 0UL)
			// 				{
			// 					forced++;
			// 				}
			// 			}
			// 		}

			// 		if (!TryPropagateCarry(ref carryLow, forced, forced, 1))
			// 		{
			// 			prefixOk = false;
			// 			break;
			// 		}
			// 	}

			// 	if (!prefixOk)
			// 	{
			// 		continue;
			// 	}
			// }

			// word is reused as segmentBase from here to limit registry pressure
			word = 0;
			// column is reused as windowColumn from here to limit registry pressure
			column = 0;

			// aIndex is reused as initialState from here to limit registry pressure
			aIndex = GetAKnownStateRowAware(0, word, windowSize, knownOne0, knownZero0, knownOne1, knownZero1);
			aOneWin[0] |= aIndex == 1 ? 1UL : 0UL;
			aZeroWin[0] |= aIndex == 2 ? 1UL : 0UL;

#if SELF_TEST
			// --- SELFTEST BYPASS: if we reached the known divisor q, bypass all prefilters and run DP directly.
			isTargetQ = qWords[0] == TargetQ0;
			for (int w = 1; w < MaxQWordCount && isTargetQ; w++)
			{
				if (qWords[w] != 0UL) { isTargetQ = false; break; }
			}
#endif
			CarryRange carry;
			if (false && !TryRunHighBitAndBorrowPreFiltersCombinedGpu(qOffsets, offsetCount, pLong, maxAllowedA, lo0, hi0, lo1, hi1))
			{
				continue;
			}

#if DETAILED_LOG || SELF_TEST || EXPORT_STAGE
			if (isTargetQ && ExportTargetStageIndex == ExportTargetStage.AfterHighBitBorrowPreFilters)
			{
				DebugWrite(foundOut, index, 1, 104UL); // marker
				DebugWrite(foundOut, index, 3, (ulong)batch);
				DebugWrite(foundOut, index, 5, 1UL);   // reached post-prefilter point
				foundOut[index] = (int)ExportTargetStage.AfterHighBitBorrowPreFilters;
				return;
			}
#endif
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

#if SELF_TEST
				// SELFTEST BYPASS: for the known q, run DP immediately and write foundOut=batch.
				// Place this here (after delta8 is ready) to avoid undefined delta and missing variables.
				if (isTargetQ)
				{
					// // Ensure q is single-word (known divisor fits in 64-bit)
					CarryRange carryTmp = CarryRange.Zero;

					// IMPORTANT: in this kernel you use:
					//   word   -> segmentBase
					//   column -> windowColumn
					// so pass them by ref exactly like below.
					bool okTarget = TryProcessBottomUpBlockRowAwareGpu(
						qOffsets,
						offsetCount,
						qMaskWords,
						qWordCount,
						lastWordMask,
						maxAllowedA,
						0,
						pLong,
						ref carryTmp,
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

					if (okTarget)
					{
						foundOut[index] = batch;
						return;
					}
				}
#endif


			carry = CarryRange.Zero;
#if DETAILED_LOG
				int preWindowColumn = column;
				int preSegmentBase = word;
				ulong preKnownZero0 = knownZero0[0];
				ulong preKnownOne0 = knownOne0[0];
				ulong preAOneWin0 = aOneWin[0];
				ulong preAZeroWin0 = aZeroWin[0];
				long preCarryMin = 0L;
				long preCarryMax = 0L;
#endif

#if DETAILED_LOG || SELF_TEST || EXPORT_STAGE
				if (isTargetQ && ExportTargetStageIndex == ExportTargetStage.JustBeforeDP)
				{
					DebugWrite(foundOut, index, 1, 105UL); // exported stage 5
					DebugWrite(foundOut, index, 3, (ulong)batch);
					DebugWrite(foundOut, index, 5, knownOne0[0]);
					DebugWrite(foundOut, index, 7, knownZero0[0]);
					DebugWrite(foundOut, index, 9, aOneWin[0]);
					DebugWrite(foundOut, index, 11, aZeroWin[0]);
					DebugWrite(foundOut, index, 13, (ulong)maxKnownA);
					DebugWrite(foundOut, index, 15, (ulong)maxAllowedA);
					foundOut[index] = (int)ExportTargetStage.JustBeforeDP;
					return;
				}
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

#if DETAILED_LOG || SELF_TEST || EXPORT_STAGE
			if (isTargetQ && ExportTargetStageIndex == ExportTargetStage.AfterDP)
			{
				DebugWrite(foundOut, index, 1, 106UL);           // marker
				DebugWrite(foundOut, index, 3, (ulong)batch);
				DebugWrite(foundOut, index, 5, ok ? 1UL : 0UL);  // DP result
				DebugWrite(foundOut, index, 7, (ulong)carry.Min);
				DebugWrite(foundOut, index, 9, (ulong)carry.Max);
				DebugWrite(foundOut, index, 11, (ulong)firstForced);
				DebugWrite(foundOut, index, 13, (ulong)firstRemaining);
				DebugWrite(foundOut, index, 15, (ulong)firstQWordsEff);
				DebugWrite(foundOut, index, 17, firstQLastEff);
				foundOut[index] = (int)ExportTargetStage.AfterDP;
				return;
			}
#endif

			if (ok)
			{
#if DETAILED_LOG
					int debugBase = index * DebugWordCountPerSlot;
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
					debugOut[extraBase + 14] = (ulong)maxFixedPreFilter;
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
#endif

				foundOut[index] = batch;
				return;
			}

		NextQ:
			;
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

			const ulong TargetQ0 = 1209708008767UL;
			bool isTargetQ = false;
			isTargetQ = (qWords[0] == TargetQ0);

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

#if DETAILED_LOG
				if (isTargetQ)
				{
					DebugWrite(debugOut, index, 4, (ulong)offsetCount);
					DebugWrite(debugOut, index, 5, offsetCount > 0 ? (ulong)qOffsets[0] : 0UL);
					DebugWrite(debugOut, index, 6, offsetCount > 0 ? (ulong)qOffsets[offsetCount - 1] : 0UL);
					DebugWrite(debugOut, index, 7, (ulong)qBitLen);
					DebugWrite(debugOut, index, 8, (ulong)qWordCount);
					DebugWrite(debugOut, index, 0, 2UL); // stage 2: offsets built
				}
#endif

			if (offsetCount <= 0)
			{
				continue;
			}

			maxOffsetValue = qOffsets[offsetCount - 1];
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

			int maxFixedPreFilter = 512 - 1;
			maxFixedPreFilter = maxFixedPreFilter > maxKnownA ? maxKnownA : maxFixedPreFilter;
			maxFixedPreFilter = maxFixedPreFilter > maxAllowedA ? maxAllowedA : maxFixedPreFilter;

			if (maxFixedPreFilter >= 8)
			{
				// tmpWords is reused for reversePrefix
				ReverseBits1024(qPrefix, tmpWords);

				bool prefixOk = true;
				CarryRange carryLow = CarryRange.Zero;
				for (int column = 0; column <= maxFixedPreFilter; column++)
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

			if (!TryRunHighBitAndBorrowPreFiltersCombinedGpu(qOffsets, offsetCount, pLong, maxAllowedA, lo0, hi0, lo1, hi1))
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
				debugOut[extraBase + 14] = (ulong)maxFixedPreFilter;
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
		in ReadOnlySpan<ulong> batchIndexWords,
		int wordCount,
		int batchOffset,
		Span<ulong> qWords,
		out int qBitLen)
	{
		Span<ulong> kWords = stackalloc ulong[wordCount];

		if (MultiplyWordsByUInt64(batchIndexWords, kWords, wordCount, BatchCount) != 0UL)
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
		if (value == 0UL)
		{
			return 0UL;
		}

		if ((words[0] += value) >= value)
		{
			return 0UL;
		}

		// value is reused as carry from here to limit registry pressure. From now on it can be either 0 or 1, only.
		for (int i = 1; i < wordCount; i++)
		{
			value = words[i] + 1UL;
			words[i] = value;
			// value is reused as sum from here to limit registry pressure
			if (value >= 1UL)
			{
				return 0UL;
			}
		}

		return 1UL;
	}

	private static ulong MultiplyWordsByUInt64(in ReadOnlySpan<ulong> source, Span<ulong> destination, int wordCount, ulong factor)
	{
		ulong carry = 0UL;
		bool hasCarry = false;
		for (int i = 0; i < wordCount; i++)
		{
			// carryOut is reused as low here to limit registry pressure
			MultiplyPartsGpu(source[i], factor, out ulong high, out ulong carryOut);

			ulong sum = carryOut + carry;
			// carryOut is now carryOut
			carryOut = sum < carryOut ? 1UL : 0UL;
			sum += hasCarry ? 1UL : 0UL;
			destination[i] = sum;
			carryOut += sum == 0UL ? 1UL : 0UL;
			carryOut += high;
			hasCarry = carryOut < high;
			carry = carryOut;
		}

		carry += hasCarry ? 1UL : 0UL;
		return carry;
	}

	private static int GetBitLength(in ReadOnlySpan<ulong> words, int wordCount)
	{
		for (int i = wordCount - 1; i >= 0; i--)
		{
			ulong word = words[i];
			if (word != 0UL)
			{
				return (i << 6) + (64 - XMath.LeadingZeroCount(word));
			}
		}

		return 0;
	}

	private static ulong AddUInt64ToWords(ArrayView<ulong> words, int wordCount, ulong value)
	{
		if (value == 0UL)
		{
			return 0UL;
		}

		// Is there additional carry beyond what we entered this method with? If not, we can quit.
		if ((words[0] += value) >= value)
		{
			return 0UL;
		}

		// value is reused as carry from here to limit registry pressure. From now on it can be either 0 or 1, only.
		for (int i = 1; i < wordCount; i++)
		{
			// value is reused as sum from here to limit registry pressure
			value = words[i] + 1UL;
			words[i] = value;
			if (value > 0UL)
			{
				return 0UL;
			}
		}

		// When we came to here this means that there was a carry in all the for loop iterations
		return 1UL;
	}

	private static ulong MultiplyWordsByUInt64(in ArrayView<ulong> source, ArrayView<ulong> destination, int wordCount, ulong factor)
	{
		ulong carry = 0UL;
		bool hasCarry = false;
		for (int i = 0; i < wordCount; i++)
		{
			MultiplyPartsGpu(source[i], factor, out ulong high, out ulong low);

			ulong sum = low + carry;
			// low is reused as carryOut from here to limit registry pressure
			low = sum < low ? 1UL : 0UL;
			sum += hasCarry ? 1UL : 0UL;
			destination[i] = sum;
			low += sum != 0UL ? 0UL : 1UL;
			// low is reused as nextCarry from here to limit registry pressure
			low += high;
			hasCarry = low < high;
			carry = low;
		}

		carry += hasCarry ? 1UL : 0UL;
		return carry;
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
		// left is reused as leftHigh from here to limit registry pressure
		left >>= 32;
		ulong rightLow = (uint)right;
		// right is reused as rightHigh from here to limit registry pressure
		right >>= 32;

		ulong lowProduct = unchecked(leftLow * rightLow);
		// rightLow is reused as cross1 from here to limit registry pressure
		rightLow = unchecked(left * rightLow);
		// leftLow is reused as cross2 from here to limit registry pressure
		leftLow = unchecked(leftLow * right);
		// left is reused as highProduct from here to limit registry pressure
		left = unchecked(left * right);

		// right is reused as carry from here to limit registry pressure
		right = unchecked((lowProduct >> 32) + (uint)rightLow + (uint)leftLow);
		low = unchecked((lowProduct & 0xFFFFFFFFUL) | (right << 32));
		high = unchecked(left + (rightLow >> 32) + (leftLow >> 32) + (right >> 32));
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
		in ArrayView<ulong> words,
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

	private static void FindBounds(in ArrayView<int> values, int valueCount, int lowerTarget, int upperTarget, out int lower, out int upper)
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

	private static bool TryRunHighBitAndBorrowPreFiltersCombinedGpu(
		in ArrayView<int> qOffsets,
		int qOffsetCount,
		int pLong,
		int maxAllowedA,
		ArrayView<long> lo0,
		ArrayView<long> hi0,
		ArrayView<long> lo1,
		ArrayView<long> hi1)
	{
		// pLong becomes the column from here. we're reusing the variable to limit registry pressure.
		pLong--;
		// sequentialColumns is reused as columnPlusOne here to limit registry pressure.
		int sequentialColumns = pLong + 1;
		int highBitColumns = HighBitCarryPreFilterColumns;
		highBitColumns = highBitColumns > sequentialColumns ? sequentialColumns : highBitColumns;

		int borrowColumns = 0;
		bool runBorrowPreFilter = qOffsetCount >= TopDownBorrowMinUnknown;
		if (runBorrowPreFilter)
		{
			borrowColumns = TopDownBorrowColumns;
			borrowColumns = borrowColumns > sequentialColumns ? sequentialColumns : borrowColumns;
		}

		sequentialColumns = highBitColumns > borrowColumns ? highBitColumns : borrowColumns;

		var outLo = lo0;
		var outHi = hi0;
		var inLo = lo1;
		var inHi = hi1;

		int outCount = 1;
		outLo[0] = 0;
		outHi[0] = 0;

		int inCount = 0;
		CarryRange borrow = CarryRange.Zero;

		int step = 0;
		int lowerT = pLong - maxAllowedA;
		lowerT = lowerT < 0 ? 0 : lowerT;

		FindBounds(qOffsets, qOffsetCount, lowerT, pLong, out lowerT, out int upperT);
		int n = upperT - lowerT;
		if (n < 0)
		{
			return false;
		}

		// flip is reused as borrowPreFilterAllowed here to limit registry pressure
		bool flip = runBorrowPreFilter && n < TopDownBorrowMinUnknown;
		sequentialColumns = flip && highBitColumns < sequentialColumns ? highBitColumns : sequentialColumns;
		runBorrowPreFilter = runBorrowPreFilter && !flip;

		// flip is flip again from here
		flip = false;
		while (true)
		{
			if (step < highBitColumns)
			{
				inCount = 0;
				for (int i = 0; i < outCount; i++)
				{
					long rMin = outLo[i];
					long rMax = outHi[i];

					if (rMax < 0)
					{
						continue;
					}

					rMin = rMin < 0 ? 0 : rMin;
					if (rMax < rMin)
					{
						continue;
					}

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
						if (inCount == 0)
						{
							return true;
						}

						continue;
					}

					if (n == 1)
					{
						AddInterval(ref inCount, inLo, inHi, rMin - 1, rMax);
						if (inCount == 0)
						{
							return true;
						}

						continue;
					}

					if (rMax <= n)
					{
						AddInterval(ref inCount, inLo, inHi, 0, rMax);
						if (inCount == 0)
						{
							return true;
						}
					}
					else if (rMin > n)
					{
						AddInterval(ref inCount, inLo, inHi, rMin - n, rMax);
						if (inCount == 0)
						{
							return true;
						}
					}
					else
					{
						AddInterval(ref inCount, inLo, inHi, 0, rMax);
						if (inCount == 0)
						{
							return true;
						}
					}
				}

				if (inCount <= 0)
				{
					return false;
				}

				outCount = inCount;
				flip = !flip;
				outLo = flip ? lo1 : lo0;
				outHi = flip ? hi1 : hi0;
				inLo = flip ? lo0 : lo1;
				inHi = flip ? hi0 : hi1;
			}

			if (runBorrowPreFilter && step < borrowColumns)
			{
				if (!TryPropagateCarry(ref borrow, 0, n, 1))
				{
					return false;
				}
			}

			step++;
			pLong--;
			if (step >= sequentialColumns || pLong < 0)
			{
				break;
			}

			lowerT = pLong - maxAllowedA;
			lowerT = lowerT < 0 ? 0 : lowerT;

			FindBounds(qOffsets, qOffsetCount, lowerT, pLong, out lowerT, out upperT);
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

	private static void CopyWords(in ArrayView<ulong> source, ArrayView<ulong> destination, int wordCount)
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

	private static void MulMod2k(in ArrayView<ulong> left, in ArrayView<ulong> right, ArrayView<ulong> result, int wordCount)
	{
		// Classic schoolbook multiplication modulo 2^(64*wordCount).
		// We keep only the lowest 'wordCount' limbs; higher limbs are discarded (mod 2^k).
		ClearWords(result, wordCount);

		for (int i = 0; i < wordCount; i++)
		{
			ulong carry = 0UL;

			// For modulus 2^(64*wordCount), we only need targets < wordCount
			int maxJ = wordCount - i;
			ulong a = left[i];

			for (int j = 0; j < maxJ; j++)
			{
				int t = i + j;

				MultiplyPartsGpu(a, right[j], out ulong hi, out ulong lo);

				// sum = result[t] + lo + carry
				ulong sum = result[t] + lo;
				ulong c1 = (sum < lo) ? 1UL : 0UL;

				ulong sum2 = sum + carry;
				ulong c2 = (sum2 < carry) ? 1UL : 0UL;

				result[t] = sum2;

				// New carry = hi + c1 + c2
				ulong newCarry = hi + c1;
				ulong c3 = (newCarry < hi) ? 1UL : 0UL;
				newCarry += c2;
				// if overflowed again, it would propagate into higher limb which we also keep via carry
				// but any overflow beyond wordCount is discarded automatically by the bounds.
				carry = newCarry + c3;
			}

			// Any leftover carry would go to limb (i + maxJ) == wordCount, which is discarded mod 2^k.
		}
	}

	private static void ComputeInverseMod2k(
		ArrayView<ulong> modulus,
		ArrayView<ulong> inverse,
		ArrayView<ulong> tmp,
		ArrayView<ulong> tmp2,
		int wordCount)
	{
		// IMPORTANT: start from x0 = 1 (mod 2). This is the canonical Hensel/Newton seed for odd modulus.
		// Using x0 = modulus is fragile and can yield incorrect inverse on GPU due to carry/overflow nuances.
		ClearWords(inverse, wordCount);
		inverse[0] = 1UL;

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

	private static void NegateMod2k(in ArrayView<ulong> value, ArrayView<ulong> result, int wordCount)
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

	private static bool TryAdvanceStableUnknownGpu(ref CarryRange carry, int unknown, int columnCount, in ArrayView<int> delta)
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

		// columnCount is reused as remaining from here to limit registry pressure
		int sMax;
		const long CarryDecimalMask = (1L << DeltaColumnsAtOnce) - 1L;
		while (columnCount >= DeltaColumnsAtOnce)
		{
			// sMax is reused as carryMask here to limit registry pressure
			sMax = (int)(carryMax & CarryDecimalMask);
			carryMax = (carryMax + delta[sMax]) >> DeltaColumnsAtOnce;
			columnCount -= DeltaColumnsAtOnce;
		}

		while (columnCount > 0)
		{
			sMax = (((unknown ^ (int)carryMax) & 1) != 0) ? unknown : unknown - 1;
			carryMax = (carryMax + sMax - 1) >> 1;
			columnCount--;
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
		long alignedMin = AlignUpToParity(carry.Min + min, requiredParity);
		long alignedMax = AlignDownToParity(carry.Max + max, requiredParity);
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
		in ArrayView<ulong> knownOne0,
		in ArrayView<ulong> knownZero0,
		in ArrayView<ulong> knownOne1,
		in ArrayView<ulong> knownZero1)
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
			if ((knownOne0[word] & mask) != 0UL)
			{
				return 1;
			}

			if ((knownZero0[word] & mask) != 0UL)
			{
				return 2;
			}

			return 0;
		}

		// rel is reused as segment1Start here to limit registry pressure
		rel = segmentBase - windowSize;
		if (aIndex >= rel && aIndex < segmentBase)
		{
			// rel is back rel after this assignment
			rel = aIndex - rel;
			word = rel >> 6;
			mask = 1UL << (rel & 63);
			if ((knownOne1[word] & mask) != 0UL)
			{
				return 1;
			}

			if ((knownZero1[word] & mask) != 0UL)
			{
				return 2;
			}

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

		int i;
		for (i = 0; i < wordCount; i++)
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
			i = wordCount - 1;
			wordsOne[i] &= lastWordMask;
			wordsZero[i] &= lastWordMask;
		}
	}

	private static void ComputeForcedRemainingAndCleanPrefix(
		in ArrayView<ulong> qMask,
		in ArrayView<ulong> oneWin,
		in ArrayView<ulong> zeroWin,
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
			q = i != lastIndex ? q : q & lastMask;

			// unknown is reused as one here to limit registry pressure
			ulong unknown = oneWin[i] & q;
			forced += XMath.PopCount(unknown);

			// unknown is back unknown from here
			unknown = oneWin[i] | zeroWin[i];
			unknown = q & ~unknown;

			remaining += XMath.PopCount(unknown);
		}
	}

	private static bool TryFindSingleUnknownInQPrefix(
		in ArrayView<ulong> qMask,
		in ArrayView<ulong> oneWin,
		in ArrayView<ulong> zeroWin,
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

			ulong unknown = oneWin[w] | zeroWin[w];
			unknown = q & ~unknown;
			if (unknown == 0UL)
			{
				continue;
			}

			found += XMath.PopCount(unknown);
			if (found > 1)
			{
				return false;
			}

			offset = (w << 6) + XMath.TrailingZeroCount(unknown);
		}

		return found == 1;
	}

	private static bool TryProcessBottomUpBlockRowAwareGpu(
		in ArrayView<int> qOffsets,
		int qOffsetCount,
		in ArrayView<ulong> qMaskWords,
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
			int aIndex;
			ArrayView<ulong> temp;
			int step;
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

					// Build aOneWin/aZeroWin in 64-bit chunks to reduce global/local memory writes.
					int curWord = -1;
					ulong accOne = 0UL;
					ulong accZero = 0UL;

					for (step = 0; step <= maxOffset; step++)
					{
						int w = step >> 6;
						int b = step & 63;

						if (w != curWord)
						{
							// flush previous word
							if (curWord >= 0)
							{
								aOneWin[curWord] = accOne;
								aZeroWin[curWord] = accZero;
							}

							curWord = w;
							accOne = 0UL;
							accZero = 0UL;
						}

						aIndex = startColumn - step;
						int state = aIndex < 0 || aIndex > maxAllowedA
							? 2
							: GetAKnownStateRowAware(aIndex, segmentBase, windowSize, knownOne0, knownZero0, knownOne1, knownZero1);

						ulong bit = 1UL << b;
						accOne = state == 1 ? accOne | bit : accOne;
						accZero = state == 2 ? accZero | bit : accZero;
					}

					// flush last partial word
					if (curWord >= 0)
					{
						aOneWin[curWord] = accOne;
						aZeroWin[curWord] = accZero;
					}

					// apply last mask
					if (qWordCount > 0)
					{
						aOneWin[qWordCountMinusOne] &= lastWordMask;
						aZeroWin[qWordCountMinusOne] &= lastWordMask;
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

			// SAFETY: for startColumn >= 0, q prefix must never be empty because q has bit0=1.
			// If qWordsEff==0 here, DP would immediately produce forced=remaining=0 and false-negative.
			// This clamp preserves correctness and matches the mathematical invariant.
			if (startColumn >= 0)
			{
				if (qWordsEff <= 0)
				{
					qWordsEff = 1;
					qLastEff = 1UL;
				}
				else if (qWordsEff == 1 && qLastEff == 0UL)
				{
					qLastEff = 1UL;
				}
			}

			// CPU-compatible StableUnknown batching: allowed only when the current window contains no known A bits
			// that can affect active q positions (i.e., maxKnownA < startColumn - maxOffset).
			if (startColumn >= maxOffset && startColumn <= maxAllowedA && maxKnownA < startColumn - maxOffset)
			{
				// step is reused as remaining here to limit registry pressure
				step = endColumnMinusOne;
				if (step > maxAllowedA)
				{
					step = maxAllowedA;
				}

				step = step - startColumn + 1;

				// CPU path additionally requires at least 16 columns to make batching worthwhile/safe.
				if (step >= 16)
				{
					while (step >= TailCarryBatchColumns)
					{
						if (!TryAdvanceStableUnknownGpu(ref carry, stableUnknown, TailCarryBatchColumns, stableUnknownDelta))
						{
							return false;
						}

						startColumn += TailCarryBatchColumns;
						windowColumn = startColumn;
						step -= TailCarryBatchColumns;
					}

					if (step > 0)
					{
						if (!TryAdvanceStableUnknownGpu(ref carry, stableUnknown, step, stableUnknownDelta))
						{
							return false;
						}

						startColumn += step;
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
				if (TryFindSingleUnknownInQPrefix(qMaskWords, aOneWin, aZeroWin, qWordsEff, qLastEff, out step))
				{
					aIndex = startColumn - step;
					const int requiredBit = 1;
					int parity = (int)(carry.Min & 1L);
					int needOne = (requiredBit ^ parity ^ (forced & 1)) & 1;

					int state = GetAKnownStateRowAware(aIndex, segmentBase, windowSize, knownOne0, knownZero0, knownOne1, knownZero1);
					int b, w;
					bool isOne = needOne != 0;
					bool parityConflict = (isOne && state == 2) || (!isOne && state == 1);
					if (parityConflict)
					{
						return false;
					}

					// if (isOne)
					// {
					//     if (state == 2)
					//     {
					//         return false;
					//     }
					// }
					// else
					// {
					//     if (state == 1)
					//     {
					//         return false;
					//     }
					// }

					forced += isOne ? 1 : 0;
					temp = isOne ? aOneWin : aZeroWin;
					SetAKnownRowAware(aIndex, isOne, segmentBase, windowSize, knownOne0, knownZero0, knownOne1, knownZero1);
					w = step >> 6;
					b = step & 63;
					temp[w] |= 1UL << b;
					remainingCols = 0;
					maxKnownA = aIndex > maxKnownA ? aIndex : maxKnownA;

					// if (aIndex > maxKnownA)
					// {
					//     maxKnownA = aIndex;
					// }
				}
			}

			// Capture first-column diagnostics *before* TryPropagateCarry, because propagation may fail.
			if (startColumn == 0 && firstForced < 0)
			{
				firstForced = forced;
				firstRemaining = remainingCols;
				firstQWordsEff = qWordsEff;
				firstQLastEff = qLastEff;

				// carry state BEFORE propagation (useful if propagation fails)
				firstCarryMin = carry.Min;
				firstCarryMax = carry.Max;
			}

			if (!TryPropagateCarry(ref carry, forced, forced + remainingCols, 1))
			{
				return false;
			}

			// aIndex is reused as nextA from here to limit registry pressure
			aIndex = startColumn + 1;
			ShiftLeft1InPlace(aOneWin, aZeroWin, qWordCount, lastWordMask);

			// aIndex is reused as nextState from here to limit registry pressure
			aIndex = aIndex > maxAllowedA
				? 2
				: GetAKnownStateRowAware(aIndex, segmentBase, windowSize, knownOne0, knownZero0, knownOne1, knownZero1);

			aOneWin[0] |= aIndex == 1 ? 1UL : 0UL;
			aZeroWin[0] |= aIndex == 2 ? 1UL : 0UL;

			startColumn++;
			windowColumn = startColumn;
		}

		return true;
	}
}


