using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

internal static partial class PrimeOrderCalculator
{
        private static bool IsGpuPow2Allowed => s_pow2ModeInitialized && s_allowGpuPow2;

        private const int GpuSmallPrimeFactorSlots = 64;

        private static bool TryPopulateSmallPrimeFactorsGpu(ulong value, uint limit, Dictionary<ulong, int> counts, out ulong remaining)
        {
                ulong[] primeBufferArray = ThreadStaticPools.UlongPool.Rent(GpuSmallPrimeFactorSlots);
                int[] exponentBufferArray = ThreadStaticPools.IntPool.Rent(GpuSmallPrimeFactorSlots);
                Span<ulong> primeBuffer = primeBufferArray.AsSpan(0, GpuSmallPrimeFactorSlots);
                Span<int> exponentBuffer = exponentBufferArray.AsSpan(0, GpuSmallPrimeFactorSlots);
                primeBuffer.Clear();
                exponentBuffer.Clear();

                if (!PrimeOrderGpuHeuristics.TryPartialFactor(value, limit, primeBuffer, exponentBuffer, out int factorCount, out ulong gpuRemaining, out _))
                {
                        remaining = value;
                        ThreadStaticPools.UlongPool.Return(primeBufferArray);
                        ThreadStaticPools.IntPool.Return(exponentBufferArray);
                        return false;
                }

                remaining = gpuRemaining;
                for (int i = 0; i < factorCount; i++)
                {
                        ulong primeValue = primeBuffer[i];
                        int exponent = exponentBuffer[i];
                        if (primeValue == 0UL || exponent == 0)
                        {
                                continue;
                        }

                        counts[primeValue] = exponent;
                }

                ThreadStaticPools.UlongPool.Return(primeBufferArray);
                ThreadStaticPools.IntPool.Return(exponentBufferArray);
                return true;
        }
}
