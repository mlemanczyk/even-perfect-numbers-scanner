using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using PerfectNumbers.Core;

namespace PerfectNumbers.Core.Gpu;

internal static class DivisorByDivisorKernels
{
    public static void CheckKernel(
        Index1D index,
        ArrayView<GpuDivisorPartialData> divisors,
        ArrayView<int> exponentOffsets,
        ArrayView<int> exponentCounts,
        ArrayView<ulong> exponents,
        ArrayView<ulong> divisorCycles,
        ArrayView<byte> hits,
        ArrayView<int> firstHit)
    {
        int globalIndex = index;
        int count = exponentCounts[globalIndex];
        if (count <= 0)
        {
            return;
        }

        int offset = exponentOffsets[globalIndex];
        GpuDivisorPartialData divisor = divisors[globalIndex];
        ulong cycleLength = divisorCycles[globalIndex];

        var stepper = new ExponentRemainderStepperGpu(divisor);

        ulong firstExponent = exponents[offset];
        bool firstUnity = stepper.InitializeIsUnityGpu(firstExponent);

        byte firstHitValue;
        ulong previousExponent = firstExponent;
        ulong cycleRemainder = 0UL;
        bool hasCycle = cycleLength != 0UL;
        if (hasCycle)
        {
            cycleRemainder = firstExponent % cycleLength;
            firstHitValue = cycleRemainder == 0UL && firstUnity ? (byte)1 : (byte)0;
        }
        else
        {
            firstHitValue = firstUnity ? (byte)1 : (byte)0;
        }

        hits[offset] = firstHitValue;
        byte anyHit = firstHitValue;

        int endIndex = offset + count;
        if (hasCycle)
        {
            GpuUInt128 remainderValue = new(cycleRemainder);
            for (int exponentIndex = offset + 1; exponentIndex < endIndex; exponentIndex++)
            {
                ulong exponent = exponents[exponentIndex];
                ulong delta = exponent - previousExponent;
                previousExponent = exponent;

                remainderValue.AddMod(delta, cycleLength);
                cycleRemainder = remainderValue.Low;

                byte hit = 0;
                if (cycleRemainder == 0UL)
                {
                    hit = stepper.ComputeNextIsUnityGpu(exponent) ? (byte)1 : (byte)0;
                }

                hits[exponentIndex] = hit;
                anyHit |= hit;
            }
        }
        else
        {
            for (int exponentIndex = offset + 1; exponentIndex < endIndex; exponentIndex++)
            {
                ulong exponent = exponents[exponentIndex];
                byte hit = stepper.ComputeNextIsUnityGpu(exponent) ? (byte)1 : (byte)0;
                hits[exponentIndex] = hit;
                anyHit |= hit;
            }
        }

        if (anyHit != 0 && firstHit.Length > 0)
        {
            Atomic.Min(ref firstHit[0], globalIndex);
        }
    }
}
