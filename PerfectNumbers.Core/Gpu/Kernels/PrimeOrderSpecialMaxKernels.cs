using ILGPU;

namespace PerfectNumbers.Core.Gpu;

internal static partial class PrimeOrderGpuHeuristics
{
    internal static void EvaluateSpecialMaxCandidatesKernel(
        Index1D index,
        ulong phi,
        ArrayView<ulong> factors,
        int factorCount,
        ulong divisorModulus,
        ArrayView<ulong> resultOut)
    {
        for (int i = 0; i < factorCount; i++)
        {
            factors[i] = phi / factors[i];
        }

        for (int i = 1; i < factorCount; i++)
        {
            ulong current = factors[i];
            int insert = i - 1;
            while (insert >= 0 && factors[insert] > current)
            {
                factors[insert + 1] = factors[insert];
                insert--;
            }

            factors[insert + 1] = current;
        }

        var divisorPartial = new GpuDivisorPartialData(divisorModulus);
        var stepper = new ExponentRemainderStepperGpu(divisorPartial);
        if (stepper.InitializeIsUnityGpu(factors[0]))
        {
			resultOut[0] = 0UL;
			// Atomic.Exchange(ref resultOut[0], 0);
            return;
        }

        for (int i = 1; i < factorCount; i++)
        {
            if (stepper.ComputeNextIsUnityGpu(factors[i]))
            {
				resultOut[0] = 0UL;
				// Atomic.Exchange(ref resultOut[0], 0);
                return;
            }
        }

        resultOut[0] = 1UL;
        // Atomic.Exchange(ref resultOut[0], 1);
    }
}
