using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core.Gpu;

internal static class DivisorByDivisorKernels
{
    private const int WindowSizeMax = 8;
    public static void CheckKernel(
        Index1D index,
        ArrayView<ulong> divisors,
        ArrayView<ulong> exponents,
        ArrayView<byte> hits,
        ArrayView<int> firstHit)
    {
        int globalIndex = index;
        ulong modulus = divisors[globalIndex];
        ulong exponent = exponents[globalIndex];
        ulong result = Pow2ModWindowedKernel(exponent, modulus);
        byte hit = result == 1UL ? (byte)1 : (byte)0;
        hits[globalIndex] = hit;

        if (hit != 0)
        {
            Atomic.Min(ref firstHit[0], globalIndex);
        }
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Pow2ModWindowedKernel(ulong exponent, ulong modulus)
    {
        int bitLength = GetBitLength(exponent);
        int windowSize = GetWindowSize(bitLength);
        ulong result = 1UL;
        int index = bitLength - 1;

        while (index >= 0)
        {
            ulong currentBit = (exponent >> index) & 1UL;
            ulong squared = MultiplyMod(result, result, modulus);
            bool processWindow = currentBit != 0UL;

            result = processWindow ? result : squared;
            index -= (int)(currentBit ^ 1UL);

            int windowStartCandidate = index - windowSize + 1;
            int negativeMask = windowStartCandidate >> 31;
            windowStartCandidate &= ~negativeMask;

            int windowStart = processWindow ? GetNextSetBitIndex(exponent, windowStartCandidate) : windowStartCandidate;
            int windowLength = processWindow ? index - windowStart + 1 : 0;

            for (int square = 0; square < windowLength; square++)
            {
                result = MultiplyMod(result, result, modulus);
            }

            if (!processWindow)
            {
                continue;
            }

            ulong mask = (1UL << windowLength) - 1UL;
            ulong windowValue = (exponent >> windowStart) & mask;
            ulong multiplier = ComputeWindowedOddPower(windowValue, modulus);
            result = MultiplyMod(result, multiplier, modulus);
            index = windowStart - 1;
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong ComputeWindowedOddPower(ulong windowValue, ulong modulus)
    {
        ulong baseValue = 2UL % modulus;
        if (windowValue == 1UL)
        {
            return baseValue;
        }

        ulong result = baseValue;
        ulong remaining = (windowValue - 1UL) >> 1;
        ulong squareBase = MultiplyMod(baseValue, baseValue, modulus);

        while (remaining != 0UL)
        {
            if ((remaining & 1UL) != 0UL)
            {
                result = MultiplyMod(result, squareBase, modulus);
            }

            remaining >>= 1;
            if (remaining == 0UL)
            {
                break;
            }

            squareBase = MultiplyMod(squareBase, squareBase, modulus);
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int GetBitLength(ulong value)
    {
        return 64 - XMath.LeadingZeroCount(value);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int GetWindowSize(int bitLength)
    {
        int window = WindowSizeMax;
        window = bitLength <= 671 ? 7 : window;
        window = bitLength <= 239 ? 6 : window;
        window = bitLength <= 79 ? 5 : window;
        window = bitLength <= 23 ? 4 : window;
        int clamped = bitLength >= 1 ? bitLength : 1;
        window = bitLength <= 6 ? clamped : window;
        return window;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int GetNextSetBitIndex(ulong exponent, int startIndex)
    {
        ulong guard = (ulong)(((long)startIndex - 64) >> 63);
        int shift = startIndex & 63;
        ulong mask = (~0UL << shift) & guard;
        ulong masked = exponent & mask;
        return XMath.TrailingZeroCount(masked);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MultiplyMod(ulong left, ulong right, ulong modulus)
    {
        if (left == 0UL || right == 0UL)
        {
            return 0UL;
        }

        ulong threshold = ulong.MaxValue / left;
        if (right <= threshold)
        {
            ulong product = left * right;
            return product % modulus;
        }

        return MultiplyModWide(left, right, modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MultiplyModWide(ulong left, ulong right, ulong modulus)
    {
        ulong result = 0UL;
        ulong addend = left;
        ulong remaining = right;

        while (remaining != 0UL)
        {
            if ((remaining & 1UL) != 0UL)
            {
                result += addend;
                if (result >= modulus)
                {
                    result -= modulus;
                }
            }

            addend <<= 1;
            if (addend >= modulus)
            {
                addend -= modulus;
            }

            remaining >>= 1;
        }

        return result;
    }

    public static void EvaluateDivisorWithStepperKernel(
        Index1D _,
        MontgomeryDivisorData divisor,
        ulong divisorCycle,
        ulong firstCycleRemainder,
        ArrayView<ulong> exponents,
        ArrayView<byte> hits)
    {
        int length = (int)exponents.Length;
        // EvenPerfectBitScanner always supplies at least one exponent per GPU divisor check,
        // so the guard stays commented out to keep the kernel branch-free.
        // if (length == 0)
        // {
        //     return;
        // }

        var exponentStepper = new ExponentRemainderStepperGpu(divisor);
        ulong firstPrime = exponents[0];
        bool firstUnity = exponentStepper.InitializeGpuIsUnity(firstPrime);
        // EvenPerfectBitScanner only schedules divisors with non-zero cycle lengths (equal to the tested prime),
        // so the no-cycle path remains commented out to keep the kernel branch-free.
        // if (divisorCycle == 0UL)
        // {
        //     hits[0] = firstUnity ? (byte)1 : (byte)0;
        //     for (int i = 1; i < length; i++)
        //     {
        //         ulong primeNoCycle = exponents[i];
        //         bool unityNoCycle = exponentStepper.ComputeNextGpuIsUnity(primeNoCycle);
        //         hits[i] = unityNoCycle ? (byte)1 : (byte)0;
        //     }
        //     return;
        // }

        ulong cycleLength = divisorCycle;
        ulong previousPrime = firstPrime;
        // The host precomputes the initial cycle remainder to keep the kernel free of modulo operations.
        ulong cycleRemainder = firstCycleRemainder;
        hits[0] = cycleRemainder == 0UL && firstUnity ? (byte)1 : (byte)0;

        for (int i = 1; i < length; i++)
        {
            ulong prime = exponents[i];
            ulong delta = prime - previousPrime;
            previousPrime = prime;

            var remainderValue = new GpuUInt128(cycleRemainder);
            remainderValue.AddMod(delta, cycleLength);
            cycleRemainder = remainderValue.Low;

            hits[i] = cycleRemainder == 0UL
                ? (exponentStepper.ComputeNextGpuIsUnity(prime) ? (byte)1 : (byte)0)
                : (byte)0;
        }
    }

    public static void ConvertMontgomeryResultsToHitsKernel(
        Index1D index,
        ArrayView<ulong> results,
        ArrayView<byte> hits)
    {
        int globalIndex = index;
        // EvenPerfectBitScanner launches this kernel with an extent matching the result and hit spans,
        // so the defensive bounds guard stays commented out to keep the conversion branch-free.
        // int length = (int)results.Length;
        // if (globalIndex >= length)
        // {
        //     return;
        // }

        ulong montgomeryResult = results[globalIndex];
        hits[globalIndex] = montgomeryResult == 1UL ? (byte)1 : (byte)0;
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
