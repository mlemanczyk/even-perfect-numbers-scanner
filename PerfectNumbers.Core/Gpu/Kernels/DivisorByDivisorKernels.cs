using System.Runtime.CompilerServices;
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

        int remaining = count - 1;
        if (remaining > 0)
        {
            int exponentIndex = offset + 1;
            int endIndex = offset + count;

            if (hasCycle)
            {
                int blockEnd = exponentIndex + (remaining & ~7);
                for (; exponentIndex < blockEnd; exponentIndex += 8)
                {
                    ulong exponent0 = exponents[exponentIndex];
                    ulong delta0 = exponent0 - previousExponent;
                    previousExponent = exponent0;
                    cycleRemainder = AdvanceCycleRemainder(cycleRemainder, delta0, cycleLength);
                    byte hit0 = cycleRemainder == 0UL ? ComputeUnityHit(ref stepper, exponent0) : (byte)0;
                    hits[exponentIndex] = hit0;

                    ulong exponent1 = exponents[exponentIndex + 1];
                    ulong delta1 = exponent1 - previousExponent;
                    previousExponent = exponent1;
                    cycleRemainder = AdvanceCycleRemainder(cycleRemainder, delta1, cycleLength);
                    byte hit1 = cycleRemainder == 0UL ? ComputeUnityHit(ref stepper, exponent1) : (byte)0;
                    hits[exponentIndex + 1] = hit1;

                    ulong exponent2 = exponents[exponentIndex + 2];
                    ulong delta2 = exponent2 - previousExponent;
                    previousExponent = exponent2;
                    cycleRemainder = AdvanceCycleRemainder(cycleRemainder, delta2, cycleLength);
                    byte hit2 = cycleRemainder == 0UL ? ComputeUnityHit(ref stepper, exponent2) : (byte)0;
                    hits[exponentIndex + 2] = hit2;

                    ulong exponent3 = exponents[exponentIndex + 3];
                    ulong delta3 = exponent3 - previousExponent;
                    previousExponent = exponent3;
                    cycleRemainder = AdvanceCycleRemainder(cycleRemainder, delta3, cycleLength);
                    byte hit3 = cycleRemainder == 0UL ? ComputeUnityHit(ref stepper, exponent3) : (byte)0;
                    hits[exponentIndex + 3] = hit3;

                    ulong exponent4 = exponents[exponentIndex + 4];
                    ulong delta4 = exponent4 - previousExponent;
                    previousExponent = exponent4;
                    cycleRemainder = AdvanceCycleRemainder(cycleRemainder, delta4, cycleLength);
                    byte hit4 = cycleRemainder == 0UL ? ComputeUnityHit(ref stepper, exponent4) : (byte)0;
                    hits[exponentIndex + 4] = hit4;

                    ulong exponent5 = exponents[exponentIndex + 5];
                    ulong delta5 = exponent5 - previousExponent;
                    previousExponent = exponent5;
                    cycleRemainder = AdvanceCycleRemainder(cycleRemainder, delta5, cycleLength);
                    byte hit5 = cycleRemainder == 0UL ? ComputeUnityHit(ref stepper, exponent5) : (byte)0;
                    hits[exponentIndex + 5] = hit5;

                    ulong exponent6 = exponents[exponentIndex + 6];
                    ulong delta6 = exponent6 - previousExponent;
                    previousExponent = exponent6;
                    cycleRemainder = AdvanceCycleRemainder(cycleRemainder, delta6, cycleLength);
                    byte hit6 = cycleRemainder == 0UL ? ComputeUnityHit(ref stepper, exponent6) : (byte)0;
                    hits[exponentIndex + 6] = hit6;

                    ulong exponent7 = exponents[exponentIndex + 7];
                    ulong delta7 = exponent7 - previousExponent;
                    previousExponent = exponent7;
                    cycleRemainder = AdvanceCycleRemainder(cycleRemainder, delta7, cycleLength);
                    byte hit7 = cycleRemainder == 0UL ? ComputeUnityHit(ref stepper, exponent7) : (byte)0;
                    hits[exponentIndex + 7] = hit7;

                    byte blockHits = (byte)(hit0 | hit1 | hit2 | hit3 | hit4 | hit5 | hit6 | hit7);
                    anyHit |= blockHits;
                }

                for (; exponentIndex < endIndex; exponentIndex++)
                {
                    ulong exponent = exponents[exponentIndex];
                    ulong delta = exponent - previousExponent;
                    previousExponent = exponent;
                    cycleRemainder = AdvanceCycleRemainder(cycleRemainder, delta, cycleLength);
                    byte hit = cycleRemainder == 0UL ? ComputeUnityHit(ref stepper, exponent) : (byte)0;
                    hits[exponentIndex] = hit;
                    anyHit |= hit;
                }
            }
            else
            {
                int blockEnd = exponentIndex + (remaining & ~7);
                for (; exponentIndex < blockEnd; exponentIndex += 8)
                {
                    ulong exponent0 = exponents[exponentIndex];
                    byte hit0 = ComputeUnityHit(ref stepper, exponent0);
                    hits[exponentIndex] = hit0;

                    ulong exponent1 = exponents[exponentIndex + 1];
                    byte hit1 = ComputeUnityHit(ref stepper, exponent1);
                    hits[exponentIndex + 1] = hit1;

                    ulong exponent2 = exponents[exponentIndex + 2];
                    byte hit2 = ComputeUnityHit(ref stepper, exponent2);
                    hits[exponentIndex + 2] = hit2;

                    ulong exponent3 = exponents[exponentIndex + 3];
                    byte hit3 = ComputeUnityHit(ref stepper, exponent3);
                    hits[exponentIndex + 3] = hit3;

                    ulong exponent4 = exponents[exponentIndex + 4];
                    byte hit4 = ComputeUnityHit(ref stepper, exponent4);
                    hits[exponentIndex + 4] = hit4;

                    ulong exponent5 = exponents[exponentIndex + 5];
                    byte hit5 = ComputeUnityHit(ref stepper, exponent5);
                    hits[exponentIndex + 5] = hit5;

                    ulong exponent6 = exponents[exponentIndex + 6];
                    byte hit6 = ComputeUnityHit(ref stepper, exponent6);
                    hits[exponentIndex + 6] = hit6;

                    ulong exponent7 = exponents[exponentIndex + 7];
                    byte hit7 = ComputeUnityHit(ref stepper, exponent7);
                    hits[exponentIndex + 7] = hit7;

                    byte blockHits = (byte)(hit0 | hit1 | hit2 | hit3 | hit4 | hit5 | hit6 | hit7);
                    anyHit |= blockHits;
                }

                for (; exponentIndex < endIndex; exponentIndex++)
                {
                    ulong exponent = exponents[exponentIndex];
                    byte hit = ComputeUnityHit(ref stepper, exponent);
                    hits[exponentIndex] = hit;
                    anyHit |= hit;
                }
            }
        }

        if (anyHit != 0 && firstHit.Length > 0)
        {
            Atomic.Min(ref firstHit[0], globalIndex);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong AdvanceCycleRemainder(ulong currentRemainder, ulong delta, ulong cycleLength)
    {
        if (delta >= cycleLength)
        {
            delta -= cycleLength;
            if (delta >= cycleLength)
            {
                delta %= cycleLength;
            }
        }

        ulong nextRemainder = currentRemainder + delta;
        if (nextRemainder >= cycleLength)
        {
            nextRemainder -= cycleLength;
        }

        return nextRemainder;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte ComputeUnityHit(ref ExponentRemainderStepperGpu stepper, ulong exponent)
    {
        return stepper.ComputeNextIsUnityGpu(exponent) ? (byte)1 : (byte)0;
    }
}
