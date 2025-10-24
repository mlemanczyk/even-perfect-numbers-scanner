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
        ArrayView<ulong> firstCycleRemainders,
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
        ulong firstCycleRemainder = firstCycleRemainders[globalIndex];

        var stepper = new ExponentRemainderStepperGpu(divisor);

        ulong firstExponent = exponents[offset];
        bool firstUnity = stepper.InitializeIsUnityGpu(firstExponent);

        byte firstHitValue;
        ulong previousExponent = firstExponent;
        ulong cycleRemainder = 0UL;
        if (cycleLength == 0UL)
        {
            firstHitValue = firstUnity ? (byte)1 : (byte)0;
        }
        else
        {
            cycleRemainder = firstCycleRemainder;
            firstHitValue = cycleRemainder == 0UL && firstUnity ? (byte)1 : (byte)0;
        }

        hits[offset] = firstHitValue;
        byte anyHit = firstHitValue;

        for (int i = 1; i < count; i++)
        {
            ulong exponent = exponents[offset + i];
            ulong delta = exponent - previousExponent;
            previousExponent = exponent;

            if (cycleLength != 0UL)
            {
                var remainderValue = new GpuUInt128(cycleRemainder);
                remainderValue.AddMod(delta, cycleLength);
                cycleRemainder = remainderValue.Low;

                byte hit = 0;
                if (cycleRemainder == 0UL)
                {
                    hit = stepper.ComputeNextIsUnityGpu(exponent) ? (byte)1 : (byte)0;
                }

                hits[offset + i] = hit;
                anyHit |= hit;
            }
            else
            {
                byte hit = stepper.ComputeNextIsUnityGpu(exponent) ? (byte)1 : (byte)0;
                hits[offset + i] = hit;
                anyHit |= hit;
            }
        }

        if (anyHit != 0 && firstHit.Length > 0)
        {
            Atomic.Min(ref firstHit[0], globalIndex);
        }
    }
}
