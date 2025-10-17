using System;
using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
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
                remaining = value;

                try
                {
                        var lease = GpuKernelPool.GetKernel(useGpuOrder: true);
                        try
                        {
                                var execution = lease.EnterExecutionScope();
                                try
                                {
                                        Accelerator accelerator = lease.Accelerator;
                                        AcceleratorStream stream = lease.Stream;

                                        SmallPrimeFactorTables tables = GpuKernelPool.EnsureSmallPrimeFactorTables(accelerator);
                                        SmallPrimeFactorScratch scratch = GpuKernelPool.EnsureSmallPrimeFactorScratch(accelerator, GpuSmallPrimeFactorSlots);
                                        scratch.Clear();

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

                                        int factorCount = 0;
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
                                                if (primeValue == 0UL || exponent == 0)
                                                {
                                                        continue;
                                                }

                                                counts[primeValue] = exponent;
                                        }

                                        return true;
                                }
                                finally
                                {
                                        execution.Dispose();
                                }
                        }
                        finally
                        {
                                lease.Dispose();
                        }
                }
                catch (CLException)
                {
                        remaining = value;
                        return false;
                }
                catch (Exception ex) when (ex is AcceleratorException or InternalCompilerException or NotSupportedException or InvalidOperationException or AggregateException)
                {
                        remaining = value;
                        return false;
                }
                finally
                {
                        ThreadStaticPools.UlongPool.Return(primeBufferArray);
                        ThreadStaticPools.IntPool.Return(exponentBufferArray);
                }
        }
}
