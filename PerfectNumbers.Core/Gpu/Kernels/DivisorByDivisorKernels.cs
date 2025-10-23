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
        ArrayView<MontgomeryDivisorData> divisors,
        ArrayView<ulong> exponents,
        ArrayView<byte> hits,
        ArrayView<int> firstHit)
    {
        int globalIndex = index;
        MontgomeryDivisorData divisor = divisors[globalIndex];
        ulong modulus = divisor.Modulus;
        // The GPU by-divisor pipeline only materializes odd moduli greater than one (q = 2kp + 1),
        // so the defensive guard for invalid values stays disabled here to keep the kernel branch-free.
        ulong exponent = exponents[globalIndex];
        ulong montgomeryResult = ULongExtensions.Pow2MontgomeryModWindowedGpuConvertToStandard(divisor, exponent);
        byte hit = montgomeryResult == 1UL ? (byte)1 : (byte)0;
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

        var exponentStepper = new ExponentRemainderStepper(divisor);
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

    public static void GenerateFilteredCandidatesKernel(
        Index1D index,
        ArrayView<ulong> candidates,
        ArrayView<int> filteredCount,
        ArrayView<GpuUInt128> nextStarts,
        GpuUInt128 startValue,
        GpuUInt128 stride,
        byte lastIsSevenFlag)
    {
        int globalIndex = index;
        int localIndex = Group.IdxX;
        int groupSize = Group.Dimension.X;

        ulong candidateOffset = (ulong)globalIndex;
        GpuUInt128 candidateValue = stride;
        candidateValue.Mul(candidateOffset);
        candidateValue.Add(startValue);
        ulong candidate = candidateValue.Low;
        candidates[globalIndex] = candidate;

        if (globalIndex == 0)
        {
            ulong candidateCount = (ulong)groupSize;
            GpuUInt128 nextValue = stride;
            nextValue.Mul(candidateCount);
            nextValue.Add(startValue);
            nextStarts[0] = nextValue;
        }

        bool lastIsSeven = lastIsSevenFlag != 0;
        byte remainder10 = (byte)(candidate % 10UL);
        bool acceptSeven = remainder10 == 3 || remainder10 == 7 || remainder10 == 9;
        bool acceptNonSeven = remainder10 == 1 || remainder10 == 3 || remainder10 == 7 || remainder10 == 9;
        bool accept10 = lastIsSeven ? acceptSeven : acceptNonSeven;
        byte remainder8 = (byte)(candidate & 7UL);
        bool accept8 = remainder8 == 1 || remainder8 == 7;
        byte remainder5 = (byte)(candidate % 5UL);
        bool accept5 = remainder5 != 0;
        byte remainder3 = (byte)(candidate % 3UL);
        bool accept3 = remainder3 != 0;
        int accepted = (accept10 && accept8 && accept3 && accept5) ? 1 : 0;

        var shared = SharedMemory.GetDynamic<int>();
        int length = groupSize;
        int prefixBase = 0;
        int mappingBase = length;
        int fallbackBase = length * 2;

        shared[prefixBase + localIndex] = accepted;
        shared[fallbackBase + localIndex] = globalIndex;

        Group.Barrier();

        int scanOffset = 1;
        while (scanOffset < groupSize)
        {
            int addendIndex = prefixBase + localIndex - scanOffset;
            int addend = localIndex >= scanOffset ? shared[addendIndex] : 0;

            Group.Barrier();

            int currentIndex = prefixBase + localIndex;
            int current = shared[currentIndex];
            int sum = current + addend;
            int shouldUpdate = localIndex >= scanOffset ? 1 : 0;
            shared[currentIndex] = shouldUpdate != 0 ? sum : current;

            Group.Barrier();

            scanOffset <<= 1;
        }

        int inclusivePrefix = shared[prefixBase + localIndex];
        int exclusiveIndex = inclusivePrefix - accepted;
        int targetBase = accepted != 0 ? mappingBase : fallbackBase;
        int targetOffset = accepted != 0 ? exclusiveIndex : localIndex;
        shared[targetBase + targetOffset] = globalIndex;

        Group.Barrier();

        int totalAccepted = shared[prefixBase + length - 1];
        int writerMask = localIndex < totalAccepted ? 1 : 0;
        int lookupBase = writerMask != 0 ? mappingBase : fallbackBase;
        int sourceIndex = shared[lookupBase + localIndex];

        ulong candidateValueToStore = candidates[sourceIndex];
        if (writerMask != 0)
        {
            candidates[localIndex] = candidateValueToStore;
        }

        int lastLaneMask = globalIndex == length - 1 ? 1 : 0;
        int countValue = shared[prefixBase + localIndex];
        if (lastLaneMask != 0)
        {
            filteredCount[0] = countValue;
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
