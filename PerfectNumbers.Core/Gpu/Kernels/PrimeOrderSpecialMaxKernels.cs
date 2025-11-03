using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core;

namespace PerfectNumbers.Core.Gpu;

internal static partial class PrimeOrderGpuHeuristics
{
    internal static void EvaluateSpecialMaxCandidatesKernel(
        Index1D index,
        ulong phi,
        ArrayView1D<ulong, Stride1D.Dense> factors,
        int factorCount,
        MontgomeryDivisorData divisor,
        ArrayView1D<ulong, Stride1D.Dense> candidates,
        ArrayView1D<byte, Stride1D.Dense> resultOut)
    {
        if (index != 0)
        {
            return;
        }

        int actual = 0;
        for (int i = 0; i < factorCount; i++)
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

            candidates[actual] = reduced;
            actual++;
        }

        if (actual == 0)
        {
            resultOut[0] = 1;
            return;
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
