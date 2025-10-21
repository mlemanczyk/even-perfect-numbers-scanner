using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

public static partial class ULongExtensions
{
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong MulHighGpuCompatible(this ulong x, ulong y)
        {
                GpuUInt128 product = new(x);
                GpuUInt128.Mul64(ref product, 0UL, y);
                return product.High;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong MulMod64Gpu(this ulong a, ulong b, ulong modulus)
        {
                // TODO: Remove this GPU-compatible shim from production once callers migrate to MulMod64,
                // which the benchmarks show is roughly 6-7× faster on dense 64-bit inputs.
                GpuUInt128 state = new(a % modulus);
                return state.MulMod(b, modulus);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong MulMod64GpuDeferred(this ulong a, ulong b, ulong modulus)
        {
                // TODO: Move this deferred helper to the benchmark suite; the baseline MulMod64 avoids the
                // 5-40× slowdown seen across real-world operand distributions.
                GpuUInt128 state = new(a);
                return state.MulModWithNativeModulo(b, modulus);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Pow2ModWindowedGpu(this ulong exponent, ulong modulus)
        {
                // EvenPerfectBitScanner never routes moduli ≤ 1 through this path; keep the guard commented out so the hot loop stays branch-free.
                // if (modulus <= 1UL)
                // {
                //         return 0UL;
                // }

                // EvenPerfectBitScanner always works with positive exponents here; leave the defensive check commented for future reference.
                // if (exponent == 0UL)
                // {
                //         return 1UL % modulus;
                // }

                int bitLength = GetPortableBitLengthGpu(exponent);
                int windowSize = GetWindowSizeGpu(bitLength);
                int oddPowerCount = 1 << (windowSize - 1);

                // oddPowerCount is always ≥ 1 for the production workloads handled by EvenPerfectBitScanner.
                // if (oddPowerCount <= 0)
                // {
                //         return 1UL;
                // }

                ulong[] oddPowersArray = ThreadStaticPools.UlongPool.Rent(oddPowerCount);
                Span<ulong> oddPowers = oddPowersArray.AsSpan(0, oddPowerCount);

                InitializeStandardOddPowers(modulus, oddPowers);

                ulong result = 1UL;
                int index = bitLength - 1;

                while (index >= 0)
                {
                        ulong currentBit = (exponent >> index) & 1UL;
                        ulong squared = result.MulMod64Gpu(result, modulus);
                        result = currentBit == 0UL ? squared : result;
                        index = currentBit == 0UL ? index - 1 : index;
                        if (currentBit == 0UL)
                        {
                                continue;
                        }

                        int windowStart = index - windowSize + 1;
                        // EvenPerfectBitScanner guarantees windowStart stays non-negative for production exponents, so keep the guard commented out to avoid extra branching.
                        // if (windowStart < 0)
                        // {
                        //         windowStart = 0;
                        // }

                        windowStart = GetNextSetBitIndexGpu(exponent, windowStart);

                        int windowLength = index - windowStart + 1;
                        for (int square = 0; square < windowLength; square++)
                        {
                                result = result.MulMod64Gpu(result, modulus);
                        }

                        ulong mask = (1UL << windowLength) - 1UL;
                        ulong windowValue = (exponent >> windowStart) & mask;
                        int tableIndex = (int)((windowValue - 1UL) >> 1);
                        ulong multiplier = oddPowers[tableIndex];
                        result = result.MulMod64Gpu(multiplier, modulus);

                        index = windowStart - 1;
                }

                ThreadStaticPools.UlongPool.Return(oddPowersArray);

                return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static ulong Pow2MontgomeryModWindowedGpu(in MontgomeryDivisorData divisor, ulong exponent, bool keepMontgomery)
        {
                ulong modulus = divisor.Modulus;
                // Reuse the optimized MontgomeryMultiply extension highlighted in MontgomeryMultiplyBenchmarks so the GPU path matches the fastest CPU reduction.
                // We shouldn't hit it in production code. If we do, we're doing something wrong.
                // if (exponent == 0UL)
                // {
                //         return keepMontgomery ? divisor.MontgomeryOne : 1UL % modulus;
                // }

                // We barely ever hit it in production code. It's e.g. 772 calls out of billions
                // if (exponent <= Pow2WindowFallbackThreshold)
                // {
                //         return Pow2MontgomeryModSingleBit(exponent, divisor, keepMontgomery);
                // }

                int bitLength = GetPortableBitLengthGpu(exponent);
                int windowSize = GetWindowSizeGpu(bitLength);
                ulong result = divisor.MontgomeryOne;
                ulong nPrime = divisor.NPrime;

                int index = bitLength - 1;
                while (index >= 0)
                {
                        ulong currentBit = (exponent >> index) & 1UL;
                        ulong squared = MontgomeryMultiply(result, result, modulus, nPrime);
                        bool processWindow = currentBit != 0UL;

                        result = processWindow ? result : squared;
                        index -= (int)(currentBit ^ 1UL);

                        int windowStartCandidate = index - windowSize + 1;
                        // if (windowStartCandidate < 0)
                        // {
                        //         windowStartCandidate = 0;
                        // }
                        // Clamp the negative offset without branching so the GPU loop stays divergence-free.
                        // This still handles windows that would otherwise extend past the most significant bit.
                        int negativeMask = windowStartCandidate >> 31;
                        windowStartCandidate &= ~negativeMask;

                        int windowStart = processWindow ? GetNextSetBitIndexGpu(exponent, windowStartCandidate) : windowStartCandidate;
                        int windowLength = processWindow ? index - windowStart + 1 : 0;
                        for (int square = 0; square < windowLength; square++)
                        {
                                result = MontgomeryMultiply(result, result, modulus, nPrime);
                        }

                        result = processWindow
                                ? MontgomeryMultiply(
                                        result,
                                        ComputeMontgomeryOddPowerGpu(
                                                (exponent >> windowStart) & ((1UL << windowLength) - 1UL),
                                                divisor,
                                                modulus,
                                                nPrime),
                                        modulus,
                                        nPrime)
                                : result;

                        index = processWindow ? windowStart - 1 : index;
                }

                result = keepMontgomery
                        ? result
                        : MontgomeryMultiply(result, 1UL, modulus, nPrime);

                return result;
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

                return Pow2MontgomeryGpuExecutor.Execute(exponent, divisor, keepMontgomery);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void InitializeStandardOddPowers(ulong modulus, Span<ulong> oddPowers)
        {
                // oddPowers spans at least one entry for production workloads; keep the guard commented out to avoid redundant checks in the hot loop.
                // if (oddPowers.IsEmpty)
                // {
                //         return;
                // }

                // EvenPerfectBitScanner feeds odd prime moduli (≥ 3) here, so the base value stays within range without a modulo reduction.
                ulong baseValue = 2UL;
                oddPowers[0] = baseValue;
                if (oddPowers.Length == 1)
                {
                        return;
                }

                ulong square = baseValue.MulMod64Gpu(baseValue, modulus);
                for (int i = 1; i < oddPowers.Length; i++)
                {
                        ulong previous = oddPowers[i - 1];
                        oddPowers[i] = previous.MulMod64Gpu(square, modulus);
                }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong ComputeMontgomeryOddPowerGpu(ulong exponent, in MontgomeryDivisorData divisor, ulong modulus, ulong nPrime)
        {
                ulong baseValue = divisor.MontgomeryTwo;
                ulong power = divisor.MontgomeryOne;
                ulong remaining = exponent;

                while (remaining != 0UL)
                {
                        ulong bit = remaining & 1UL;
                        ulong multiplied = MontgomeryMultiply(power, baseValue, modulus, nPrime);
                        ulong mask = (ulong)-(long)bit;
                        power = (power & ~mask) | (multiplied & mask);

                        remaining >>= 1;
                        if (remaining == 0UL)
                        {
                                break;
                        }

                        baseValue = MontgomeryMultiply(baseValue, baseValue, modulus, nPrime);
                }

                return power;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int GetPortableBitLengthGpu(ulong value)
        {
                // Keep this commented out. It will never happen in production code.
                // if (value == 0UL)
                // {
                //         return 0;
                // }

                return 64 - XMath.LeadingZeroCount(value);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int GetWindowSizeGpu(int n)
        {
                int w = Pow2WindowSize;

                w = (n <= 671) ? 7 : w;
                w = (n <= 239) ? 6 : w;
                w = (n <= 79) ? 5 : w;
                w = (n <= 23) ? 4 : w;

                int m = (n >= 1) ? n : 1;
                w = (n <= 6) ? m : w;

                return w;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int GetNextSetBitIndexGpu(ulong exponent, int startIndex)
        {
                ulong guard = (ulong)(((long)startIndex - 64) >> 63);
                int shift = startIndex & 63;
                ulong mask = (~0UL << shift) & guard;
                ulong masked = exponent & mask;

                return XMath.TrailingZeroCount(masked);
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

                public static ulong Execute(ulong exponent, in MontgomeryDivisorData divisor, bool keepMontgomery)
                {
                        ulong result = 0UL;

                        GpuKernelLease lease = GpuKernelPool.GetKernel(useGpuOrder: true);
                        var execution = lease.EnterExecutionScope();

                        Accelerator accelerator = lease.Accelerator;
                        // Keep this commented out. It should never happen in production code.
                        // if (accelerator.AcceleratorType == AcceleratorType.CPU)
                        // {
                        //         return 0UL;
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
                        //         return 0UL;
                        // }
                        // catch (NotSupportedException)
                        // {
                        //         return 0UL;
                        // }
                        // Intentionally avoid exception handling here; any accelerator failure should crash the scanner.
                        return result;
                }
        }
}
