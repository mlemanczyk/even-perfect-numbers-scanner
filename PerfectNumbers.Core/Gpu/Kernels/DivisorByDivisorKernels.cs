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
        if (modulus <= 1UL || (modulus & 1UL) == 0UL)
        {
            hits[index] = 0;
            return;
        }

        ulong exponent = exponents[index];
        hits[index] = ULongExtensions.Pow2MontgomeryModWindowedGpu(divisor, exponent, keepMontgomery: false) == 1UL ? (byte)1 : (byte)0;
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
            int addend = 0;
            if (localIndex >= offset)
            {
                addend = shared[localIndex - offset];
            }

            Group.Barrier();

            if (localIndex >= offset)
            {
                int sum = shared[localIndex] + addend;
                if (sum >= mod)
                {
                    sum %= mod;
                }

                shared[localIndex] = sum;
            }

            Group.Barrier();

            offset <<= 1;
        }

        if (globalIndex == 0)
        {
            remainders[0] = baseRemainder;
            return;
        }

        int remainder = baseRemainder + shared[localIndex];
        if (remainder >= mod)
        {
            remainder %= mod;
        }

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
        bool accept10;
        if (lastIsSevenFlag != 0)
        {
            accept10 = value10 == 3 || value10 == 7 || value10 == 9;
        }
        else
        {
            accept10 = value10 == 1 || value10 == 3 || value10 == 7 || value10 == 9;
        }

        if (!accept10)
        {
            mask[globalIndex] = 0;
            return;
        }

        byte value8 = remainder8[globalIndex];
        if (value8 != 1 && value8 != 7)
        {
            mask[globalIndex] = 0;
            return;
        }

        if (remainder3[globalIndex] == 0 || remainder5[globalIndex] == 0)
        {
            mask[globalIndex] = 0;
            return;
        }

        mask[globalIndex] = 1;
    }

    public static void ComputeMontgomeryExponentKernel(Index1D index, MontgomeryDivisorData divisor, ArrayView<ulong> exponents, ArrayView<ulong> results)
    {
        ulong modulus = divisor.Modulus;
        if (modulus <= 1UL || (modulus & 1UL) == 0UL)
        {
            results[index] = 0UL;
            return;
        }

        ulong exponent = exponents[index];

        results[index] = ULongExtensions.Pow2MontgomeryModWindowedGpu(divisor, exponent, keepMontgomery: true);
    }
}
