using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

public static partial class ULongExtensions
{
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong MulHighGpuCompatible(this ulong x, ulong y)
        {
                GpuUInt128 product = new(x);
                product.Mul64(new GpuUInt128(y));
                return product.High;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong MulMod64GpuCompatible(this ulong a, ulong b, ulong modulus)
        {
                // TODO: Remove this GPU-compatible shim from production once callers migrate to MulMod64,
                // which the benchmarks show is roughly 6-7× faster on dense 64-bit inputs.
                GpuUInt128 state = new(a % modulus);
                return state.MulMod(b, modulus);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong MulMod64GpuCompatibleDeferred(this ulong a, ulong b, ulong modulus)
        {
                // TODO: Move this deferred helper to the benchmark suite; the baseline MulMod64 avoids the
                // 5-40× slowdown seen across real-world operand distributions.
                GpuUInt128 state = new(a);
                return state.MulModWithNativeModulo(b, modulus);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Pow2MontgomeryModWindowedGpu(this ulong exponent, in MontgomeryDivisorData divisor, bool keepMontgomery)
        {
                // We shouldn't hit it in production code. If we do, we're doing something wrong.
                // ulong modulus = divisor.Modulus;
                // if (exponent == 0UL)
                // {
                //      return keepMontgomery ? divisor.MontgomeryOne : 1UL % modulus;
                // }

                // We barely ever hit it in production code. It's e.g. 772 calls out of billions
                // if (exponent <= Pow2WindowFallbackThreshold)
                // {
                //      return Pow2MontgomeryModSingleBit(exponent, divisor, keepMontgomery);
                // }

                if (Pow2MontgomeryGpuExecutor.TryExecute(exponent, divisor, keepMontgomery, out ulong gpuResult))
                {
                        return gpuResult;
                }

                return Pow2MontgomeryModWindowedCpu(exponent, divisor, keepMontgomery);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Pow2MontgomeryModWithCycleGpu(this ulong exponent, ulong cycleLength, in MontgomeryDivisorData divisor)
        {
                ulong rotationCount = exponent % cycleLength;
                return Pow2MontgomeryModWindowedGpu(rotationCount, divisor, keepMontgomery: false);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Pow2MontgomeryModFromCycleRemainderGpu(this ulong reducedExponent, in MontgomeryDivisorData divisor)
        {
                return Pow2MontgomeryModWindowedGpu(reducedExponent, divisor, keepMontgomery: false);
        }

        private static class Pow2MontgomeryGpuExecutor
        {
                private static readonly ConcurrentDictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, byte, ArrayView1D<ulong, Stride1D.Dense>>> KernelCache = new();

                public static bool TryExecute(ulong exponent, in MontgomeryDivisorData divisor, bool keepMontgomery, out ulong result)
                {
                        result = 0UL;

                        GpuKernelLease lease = GpuKernelPool.GetKernel(useGpuOrder: true);
                        var execution = lease.EnterExecutionScope();

                        Accelerator accelerator = lease.Accelerator;
                        // Keep this commented out. It should never happen in production code.
                        // if (accelerator.AcceleratorType == AcceleratorType.CPU)
                        // {
                        //      return false;
                        // }

                        var kernel = KernelCache.GetOrAdd(accelerator, static accel =>
                        {
                                var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, byte, ArrayView1D<ulong, Stride1D.Dense>>(Pow2MontgomeryKernels.Pow2MontgomeryKernel);
                                var launcher = KernelUtil.GetKernel(loaded);
                                return launcher.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, byte, ArrayView1D<ulong, Stride1D.Dense>>>();
                        });

                        var exponentBuffer = accelerator.Allocate1D<ulong>(1);
                        var resultBuffer = accelerator.Allocate1D<ulong>(1);

                        exponentBuffer.View.CopyFromCPU(ref exponent, 1);
                        // We don't need to worry about any left-overs here.
                        // resultBuffer.MemSetToZero();

                        // TODO: Have 2 kernels for both cases to have branch-less solution.
                        byte keepFlag = keepMontgomery ? (byte)1 : (byte)0;
                        AcceleratorStream stream = lease.Stream;
                        kernel(stream, 1, exponentBuffer.View, divisor, keepFlag, resultBuffer.View);
                        stream.Synchronize();

                        resultBuffer.View.CopyToCPU(ref result, 1);

                        resultBuffer.Dispose();
                        exponentBuffer.Dispose();
                        execution.Dispose();
                        lease.Dispose();
                        // Keep this commented. We don't want to catch any exceptions. All should crash the scanner.
                        // catch (AcceleratorException)
                        // {
                        //         return false;
                        // }
                        // catch (NotSupportedException)
                        // {
                        //         return false;
                        // }
                        // Intentionally avoid exception handling here; any accelerator failure should crash the scanner.
                        return true;
                }
        }
}
