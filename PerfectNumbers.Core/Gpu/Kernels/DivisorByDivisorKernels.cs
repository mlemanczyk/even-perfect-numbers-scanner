using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core.Gpu;

internal static class DivisorByDivisorKernels
{
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
        ulong result = ExponentRemainderStepperGpu.Pow2ModWindowed(exponent, modulus);
        byte hit = result == 1UL ? (byte)1 : (byte)0;
        hits[globalIndex] = hit;

        if (hit != 0)
        {
            Atomic.Min(ref firstHit[0], globalIndex);
        }
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
        bool firstUnity = exponentStepper.InitializeIsUnityGpu(firstPrime);
        // EvenPerfectBitScanner only schedules divisors with non-zero cycle lengths (equal to the tested prime),
        // so the no-cycle path remains commented out to keep the kernel branch-free.
        // if (divisorCycle == 0UL)
        // {
        //     hits[0] = firstUnity ? (byte)1 : (byte)0;
        //     for (int i = 1; i < length; i++)
        //     {
        //         ulong primeNoCycle = exponents[i];
        //         bool unityNoCycle = exponentStepper.ComputeNextIsUnityGpu(primeNoCycle);
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
                ? (exponentStepper.ComputeNextIsUnityGpu(prime) ? (byte)1 : (byte)0)
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
