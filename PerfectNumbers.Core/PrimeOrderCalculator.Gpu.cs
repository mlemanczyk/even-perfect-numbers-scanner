using System;
using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

internal static partial class PrimeOrderCalculator
{
        private static bool IsGpuPow2Allowed => s_pow2ModeInitialized && s_allowGpuPow2;

        private const int GpuSmallPrimeFactorSlots = 64;

        private static bool TryPopulateSmallPrimeFactorsGpu(ulong value, uint limit, Dictionary<ulong, int> counts, out int factorCount, out ulong remaining)
        {
                var primeBufferArray = ThreadStaticPools.UlongPool.Rent(GpuSmallPrimeFactorSlots);
                var exponentBufferArray = ThreadStaticPools.IntPool.Rent(GpuSmallPrimeFactorSlots);
                Span<ulong> primeBuffer = primeBufferArray.AsSpan(0, GpuSmallPrimeFactorSlots);
                Span<int> exponentBuffer = exponentBufferArray.AsSpan(0, GpuSmallPrimeFactorSlots);
                remaining = value;

                var lease = GpuKernelPool.GetKernel(useGpuOrder: true);
                Accelerator accelerator = lease.Accelerator;
                AcceleratorStream stream = lease.Stream;

                SmallPrimeFactorTables tables = GpuKernelPool.EnsureSmallPrimeFactorTables(accelerator);
                SmallPrimeFactorScratch scratch = GpuKernelPool.EnsureSmallPrimeFactorScratch(accelerator, GpuSmallPrimeFactorSlots);

                var kernel = lease.SmallPrimeFactorKernel;
                kernel(
                        stream,
                        1,
                        value,
                        limit,
                        tables.PrimesView,
                        tables.SquaresView,
                        tables.Count,
                        scratch.PrimeSlots.View,
                        scratch.ExponentSlots.View,
                        scratch.CountSlot.View,
                        scratch.RemainingSlot.View);

                stream.Synchronize();

                factorCount = 0;
                scratch.CountSlot.View.CopyToCPU(ref factorCount, 1);
                factorCount = Math.Min(factorCount, GpuSmallPrimeFactorSlots);

                if (factorCount > 0)
                {
                        scratch.PrimeSlots.View.CopyToCPU(ref MemoryMarshal.GetReference(primeBuffer), factorCount);
                        scratch.ExponentSlots.View.CopyToCPU(ref MemoryMarshal.GetReference(exponentBuffer), factorCount);
                }

                scratch.RemainingSlot.View.CopyToCPU(ref remaining, 1);

                for (int i = 0; i < factorCount; i++)
                {
                        ulong primeValue = primeBuffer[i];
                        int exponent = exponentBuffer[i];
                        counts.Add(primeValue, exponent);
                }

                lease.Dispose();
                ThreadStaticPools.UlongPool.Return(primeBufferArray);
                ThreadStaticPools.IntPool.Return(exponentBufferArray);
                return true;
        }

        private static bool EvaluateSpecialMaxCandidatesGpu(
                Span<ulong> buffer,
                ReadOnlySpan<FactorEntry> factors,
                ulong phi,
                ulong prime,
                in MontgomeryDivisorData divisorData)
        {
                _ = prime;

                if (factors.Length == 0)
                {
                        return true;
                }

                int factorCount = factors.Length;
                Span<ulong> factorSpan = buffer[..factorCount];
                for (int i = 0; i < factorCount; i++)
                {
                        factorSpan[i] = factors[i].Value;
                }

                var lease = GpuKernelPool.GetKernel(useGpuOrder: true);
                Accelerator accelerator = lease.Accelerator;
                AcceleratorStream stream = lease.Stream;

                SpecialMaxScratch scratch = GpuKernelPool.EnsureSpecialMaxScratch(accelerator, factorCount);

                scratch.FactorsView.SubView(0, factorCount).CopyFromCPU(ref MemoryMarshal.GetReference(factorSpan), factorCount);

                scratch.ResetCount(stream);

                var (gridExtent, groupExtent) = GridExtensions.ComputeGridStrideLoopExtent(accelerator, factorCount);
                long totalThreads = gridExtent.Size * (long)groupExtent.Size;
                if (totalThreads <= 0L)
                {
                        totalThreads = 1L;
                }
                else if (totalThreads > int.MaxValue)
                {
                        totalThreads = int.MaxValue;
                }

                var filterKernel = lease.SpecialMaxFilterKernel;
                filterKernel(
                        stream,
                        new Index1D((int)totalThreads),
                        phi,
                        scratch.FactorsView,
                        factorCount,
                        scratch.CandidatesView,
                        scratch.CountView);

                var finalizeKernel = lease.SpecialMaxFinalizeKernel;
                finalizeKernel(
                        stream,
                        1,
                        divisorData,
                        scratch.CandidatesView,
                        scratch.ResultView,
                        scratch.CountView);

                stream.Synchronize();

                byte result = 0;
                scratch.ResultView.CopyToCPU(ref result, 1);

                lease.Dispose();

                return result != 0;
        }
}

