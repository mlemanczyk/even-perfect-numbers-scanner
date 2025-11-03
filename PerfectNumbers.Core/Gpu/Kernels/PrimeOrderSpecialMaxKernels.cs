using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using PerfectNumbers.Core;

namespace PerfectNumbers.Core.Gpu;

internal static partial class PrimeOrderGpuHeuristics
{
    internal static void EvaluateSpecialMaxCandidatesFilterKernel(
        Index1D index,
        ulong phi,
        ArrayView1D<ulong, Stride1D.Dense> factors,
        int factorCount,
        ArrayView1D<ulong, Stride1D.Dense> candidates,
        ArrayView1D<int, Stride1D.Dense> candidateCount)
    {
        int threadIndex = index;
        long strideValue = GridExtensions.GridStrideLoopStride.Size;
        int stride = strideValue <= 0L ? 1 : (int)strideValue;

        for (int i = threadIndex; i < factorCount; i += stride)
        {
            ulong factor = factors[i];
            if (factor <= 1UL)
            {
                continue;
            }

            ulong reduced = phi / factor;
            if (reduced == 0UL)
            {
                continue;
            }

            int slot = Atomic.Add(ref candidateCount[0], 1);
            if (slot < candidates.Length)
            {
                candidates[slot] = reduced;
            }
        }
    }

    internal static void EvaluateSpecialMaxCandidatesFinalizeKernel(
        Index1D index,
        MontgomeryDivisorData divisor,
        ArrayView1D<ulong, Stride1D.Dense> candidates,
        ArrayView1D<byte, Stride1D.Dense> resultOut,
        ArrayView1D<int, Stride1D.Dense> candidateCount)
    {
        if (index != 0)
        {
            return;
        }

        int actual = candidateCount[0];
        if (actual <= 0)
        {
            resultOut[0] = 1;
            return;
        }

        long candidateCapacity = candidates.Length;
        if (candidateCapacity <= 0L)
        {
            resultOut[0] = 1;
            return;
        }

        if (actual > candidateCapacity)
        {
            actual = (int)candidateCapacity;
        }

        for (int i = 1; i < actual; i++)
        {
            ulong current = candidates[i];
            int insert = i - 1;
            while (insert >= 0 && candidates[insert] > current)
            {
                candidates[insert + 1] = candidates[insert];
                insert--;
            }

            candidates[insert + 1] = current;
        }

        var divisorPartial = new GpuDivisorPartialData(divisor.Modulus);
        var stepper = new ExponentRemainderStepperGpu(divisorPartial);
        if (stepper.InitializeIsUnityGpu(candidates[0]))
        {
            resultOut[0] = 0;
            return;
        }

        for (int i = 1; i < actual; i++)
        {
            if (stepper.ComputeNextIsUnityGpu(candidates[i]))
            {
                resultOut[0] = 0;
                return;
            }
        }

        resultOut[0] = 1;
    }
}
