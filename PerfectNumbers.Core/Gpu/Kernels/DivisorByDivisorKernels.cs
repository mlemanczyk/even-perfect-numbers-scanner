using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core;

namespace PerfectNumbers.Core.Gpu;

internal static class DivisorByDivisorKernels
{
    public static void CheckKernel(Index1D index, ArrayView<MontgomeryDivisorData> divisors, ArrayView<ulong> exponents, ArrayView<byte> hits)
    {
        MontgomeryDivisorData divisor = divisors[index];
        ulong modulus = divisor.Modulus;
        // The GPU by-divisor pipeline only materializes odd moduli greater than one (q = 2kp + 1),
        // so the defensive guard for invalid values stays disabled here to keep the kernel branch-free.
        ulong exponent = exponents[index];
        ulong montgomeryResult = ULongExtensions.Pow2MontgomeryModWindowedGpuConvertToStandard(divisor, exponent);
        hits[index] = montgomeryResult == 1UL ? (byte)1 : (byte)0;
    }

    public static void GenerateCandidatesAndGapsKernel(
        Index1D index,
        ArrayView<ulong> candidates,
        ArrayView<ulong> gaps,
        GpuUInt128 startValue,
        GpuUInt128 stride)
    {
        int globalIndex = index;
        int candidateLength = (int)candidates.Length;
        if (globalIndex >= candidateLength)
        {
            return;
        }

        ulong offset = (ulong)globalIndex;
        GpuUInt128 candidateValue;
        if (offset == 0UL)
        {
            candidateValue = startValue;
        }
        else
        {
            candidateValue = stride;
            candidateValue.Mul(offset);
            candidateValue.Add(startValue);
        }

        candidates[globalIndex] = candidateValue.Low;

        int gapLength = (int)gaps.Length;
        if (globalIndex >= gapLength)
        {
            return;
        }

        if (offset == 0UL)
        {
            gaps[globalIndex] = 0UL;
            return;
        }

        GpuUInt128 previousValue = startValue;
        ulong previousOffset = offset - 1UL;
        if (previousOffset != 0UL)
        {
            GpuUInt128 scaledStride = stride;
            scaledStride.Mul(previousOffset);
            previousValue.Add(scaledStride);
        }

        GpuUInt128 gapValue = candidateValue;
        gapValue.Sub(previousValue);
        gaps[globalIndex] = gapValue.Low;
    }

    public static void ComputeRemainderDeltasKernel(Index1D index, ArrayView<ulong> gaps, ArrayView<byte> deltas, byte modulus)
    {
        deltas[index] = (byte)(gaps[index] % modulus);
    }

    public static void AccumulateRemaindersKernel(
        Index1D index,
        ArrayView<byte> deltas,
        ArrayView<byte> remainders,
        byte baseRemainder,
        byte modulus)
    {
        int globalIndex = index;
        int length = (int)remainders.Length;
        if (globalIndex >= length)
        {
            return;
        }

        int localIndex = Group.IdxX;
        int groupSize = Group.Dimension.X;
        var shared = SharedMemory.GetDynamic<int>();

        shared[localIndex] = deltas[globalIndex];
        Group.Barrier();

        int offset = 1;
        int mod = modulus;
        while (offset < groupSize)
        {
            bool apply = localIndex >= offset;
            int addend = apply ? shared[localIndex - offset] : 0;

            Group.Barrier();

            int current = shared[localIndex];
            int sum = apply ? current + addend : current;
            int wrap = sum >= mod ? 1 : 0;
            sum -= wrap * mod;
            shared[localIndex] = apply ? sum : current;

            Group.Barrier();

            offset <<= 1;
        }

        int remainder = baseRemainder + shared[localIndex];
        int remainderWrap = remainder >= mod ? 1 : 0;
        remainder -= remainderWrap * mod;
        remainders[globalIndex] = (byte)remainder;
    }

    public static void EvaluateCandidateMaskKernel(
        Index1D index,
        ArrayView<byte> remainder10,
        ArrayView<byte> remainder8,
        ArrayView<byte> remainder5,
        ArrayView<byte> remainder3,
        byte lastIsSevenFlag,
        ArrayView<byte> mask)
    {
        int globalIndex = index;
        int length = (int)mask.Length;
        if (globalIndex >= length)
        {
            return;
        }

        byte value10 = remainder10[globalIndex];
        bool lastIsSeven = lastIsSevenFlag != 0;
        bool acceptSeven = value10 == 3 || value10 == 7 || value10 == 9;
        bool acceptNonSeven = value10 == 1 || value10 == 3 || value10 == 7 || value10 == 9;
        bool accept10 = lastIsSeven ? acceptSeven : acceptNonSeven;
        byte value8 = remainder8[globalIndex];
        bool accept8 = value8 == 1 || value8 == 7;
        bool accept5 = remainder5[globalIndex] != 0;
        bool accept3 = remainder3[globalIndex] != 0;
        byte accepted = (byte)((accept10 && accept8 && accept3 && accept5) ? 1 : 0);
        mask[globalIndex] = accepted;
    }

    public static void ComputeMontgomeryExponentKernel(Index1D index, MontgomeryDivisorData divisor, ArrayView<ulong> exponents, ArrayView<ulong> results)
    {
        ulong modulus = divisor.Modulus;
        // Each divisor flowing through this kernel follows q = 2kp + 1 with k >= 1, so modulus is always odd and exceeds one.
        ulong exponent = exponents[index];
        ulong montgomeryResult = ULongExtensions.Pow2MontgomeryModWindowedGpuKeepMontgomery(divisor, exponent);
        results[index] = montgomeryResult;
    }
}
