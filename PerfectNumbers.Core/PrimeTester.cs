using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

public sealed class PrimeTester
{
    public PrimeTester()
    {
    }

    [ThreadStatic]
    private static PrimeTester? _tester;

    public static PrimeTester Exclusive => _tester ??= new();

    public bool IsPrime(ulong n, CancellationToken ct)
    {
        if (n <= 1UL)
        {
            return false;
        }

        if (n == 2UL)
        {
            throw new InvalidOperationException("PrimeTester.IsPrime encountered the sentinel input 2.");
        }

        bool isOdd = (n & 1UL) != 0UL;
        bool result = isOdd;

        bool requiresTrialDivision = result && n >= 7UL;

        if (requiresTrialDivision)
        {
            // EvenPerfectBitScanner streams exponents starting at 136,279,841, so the Mod10/GCD guard never fires on the
            // production path. Leave the logic commented out as instrumentation for diagnostic builds.
            // bool sharesMaxExponentFactor = n.Mod10() == 1UL && SharesFactorWithMaxExponent(n);
            // result &= !sharesMaxExponentFactor;

            if (result)
            {
                var smallPrimeDivisorsLength = PrimesGenerator.SmallPrimes.Length;
                uint[] smallPrimeDivisors = PrimesGenerator.SmallPrimes;
                ulong[] smallPrimeDivisorsMul = PrimesGenerator.SmallPrimesPow2;
                for (int i = 0; i < smallPrimeDivisorsLength; i++)
                {
                    if (smallPrimeDivisorsMul[i] > n)
                    {
                        break;
                    }

                    if (n % smallPrimeDivisors[i] == 0)
                    {
                        result = false;
                        break;
                    }
                }
            }
        }

        return result;
    }

    public static bool IsPrimeGpu(ulong n)
    {
        return Exclusive.IsPrimeGpu(n, CancellationToken.None);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool IsPrimeGpu(ulong n, CancellationToken ct)
    {
        bool forceCpu = GpuContextPool.ForceCpu;
        Span<ulong> one = stackalloc ulong[1];
        Span<byte> outFlags = stackalloc byte[1];
        one[0] = n;
        outFlags[0] = 0;

        if (!forceCpu)
        {
            IsPrimeBatchGpu(one, outFlags);
        }

        bool belowGpuRange = n < 31UL;
        bool gpuReportedPrime = !forceCpu && !belowGpuRange && outFlags[0] != 0;
        bool requiresCpuFallback = forceCpu || belowGpuRange || !gpuReportedPrime;

        return requiresCpuFallback ? IsPrimeCpu(n, ct) : true;
    }

    public static bool IsPrimeCpu(ulong n, CancellationToken ct)
    {
        // Preserve the legacy entry point for callers that bypass the thread-local caches.
        return Exclusive.IsPrime(n, ct);
    }


    public static int GpuBatchSize { get; set; } = 262_144;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void IsPrimeBatchGpu(ReadOnlySpan<ulong> values, Span<byte> results)
    {
        var limiter = GpuPrimeWorkLimiter.Acquire();
        var gpu = PrimeTesterGpuContextPool.Rent();
        var accelerator = gpu.Accelerator;
        var state = GpuKernelState.GetOrCreate(accelerator);
        int batchSize = Math.Max(1, GpuBatchSize);

        lock (gpu.ExecutionLock)
        {
            var scratch = state.RentScratch(batchSize, accelerator);
            var input = scratch.Input;
            var output = scratch.Output;
            ArrayPool<ulong> pool = ThreadStaticPools.UlongPool;
            ulong[] temp = pool.Rent(batchSize);

            try
            {
                IsPrimeBatchGpu(values, results, accelerator, input, output, temp);
            }
            finally
            {
                pool.Return(temp, clearArray: false);
                state.ReturnScratch(scratch);
            }
        }

        gpu.Dispose();
        limiter.Dispose();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void IsPrimeBatchGpu(
        ReadOnlySpan<ulong> values,
        Span<byte> results,
        Accelerator accelerator,
        MemoryBuffer1D<ulong, Stride1D.Dense> deviceInput,
        MemoryBuffer1D<byte, Stride1D.Dense> deviceOutput,
        Span<ulong> stagingBuffer)
    {
        if (values.IsEmpty)
        {
            return;
        }

        int capacity = (int)Math.Min(Math.Min(deviceInput.Length, deviceOutput.Length), stagingBuffer.Length);

        var state = GpuKernelState.GetOrCreate(accelerator);
        var inputView = deviceInput.View;
        var outputView = deviceOutput.View;
        var primesView = state.DevicePrimes.View;
        var stagingSpan = stagingBuffer;
        ref ulong stagingRef = ref MemoryMarshal.GetReference(stagingSpan);

        int totalLength = values.Length;
        int pos = 0;
        while (pos < totalLength)
        {
            int remaining = totalLength - pos;
            int count = remaining > capacity ? capacity : remaining;

            var stagingSlice = stagingSpan.Slice(0, count);
            values.Slice(pos, count).CopyTo(stagingSlice);

            var inputSlice = inputView.SubView(0, count);
            inputSlice.CopyFromCPU(ref stagingRef, count);

            var outputSlice = outputView.SubView(0, count);
            state.Kernel(count, inputSlice, primesView, outputSlice);
            accelerator.Synchronize();
            outputSlice.CopyToCPU(ref results[pos], count);

            pos += count;
        }
    }

    internal static class PrimeTesterGpuContextPool
    {
        internal sealed class PooledContext : IDisposable
        {
            private bool _disposed;

            public Context Context { get; }

            public Accelerator Accelerator { get; }

            public object ExecutionLock { get; } = new();

            public PooledContext()
            {
                Context = Context.CreateDefault();
                Accelerator = Context.GetPreferredDevice(false).CreateAccelerator(Context);
            }

            public void Dispose()
            {
                if (_disposed)
                {
                    return;
                }

                PrimeTester.ClearGpuCaches(Accelerator);
                Accelerator.Dispose();
                Context.Dispose();
                _disposed = true;
            }
        }

        private static readonly ConcurrentQueue<PooledContext> Pool = new();
        private static readonly object CreationLock = new();

        internal static PrimeTesterGpuContextLease Rent()
        {
            if (Pool.TryDequeue(out var ctx))
            {
                return new PrimeTesterGpuContextLease(ctx);
            }

            lock (CreationLock)
            {
                return new PrimeTesterGpuContextLease(new PooledContext());
            }
        }

        internal static void DisposeAll()
        {
            while (Pool.TryDequeue(out var ctx))
            {
                ctx.Dispose();
            }
        }

        private static void Return(PooledContext ctx)
        {
            ctx.Accelerator.Synchronize();
            Pool.Enqueue(ctx);
        }

        internal struct PrimeTesterGpuContextLease : IDisposable
        {
            private PooledContext? _ctx;

            internal PrimeTesterGpuContextLease(PooledContext ctx)
            {
                _ctx = ctx;
            }

            public Accelerator Accelerator => _ctx!.Accelerator;

            public object ExecutionLock => _ctx!.ExecutionLock;

            public void Dispose()
            {
                if (_ctx is { } ctx)
                {
                    Return(ctx);
                    _ctx = null;
                }
            }
        }
    }

    // Per-accelerator GPU state for prime sieve (kernel + uploaded primes).
    internal sealed class KernelState
    {
        public Action<Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<byte>> Kernel { get; }
        public Action<Index1D, ArrayView<ulong>, ulong, ArrayView<byte>> HeuristicTrialDivisionKernel { get; }
        public MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimes { get; }
        private readonly Accelerator _accel;
        private readonly System.Collections.Concurrent.ConcurrentBag<ScratchBuffers> _scratchPool = [];
        // TODO: Replace this ConcurrentBag with the lock-free ring buffer variant validated in
        // GpuModularArithmeticBenchmarks so renting scratch buffers stops contending on the bag's internal locks when
        // thousands of GPU batches execute per second.

        public KernelState(Accelerator accelerator)
        {
            _accel = accelerator;
            // Compile once per accelerator and upload primes once.
            Kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<byte>>(PrimeTesterKernels.SmallPrimeSieveKernel);
            HeuristicTrialDivisionKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ulong, ArrayView<byte>>(PrimeTesterKernels.HeuristicTrialDivisionKernel);

            var primes = PrimesGenerator.SmallPrimes;
            DevicePrimes = accelerator.Allocate1D<uint>(primes.Length);
            DevicePrimes.View.CopyFromCPU(primes);
        }

        internal sealed class ScratchBuffers : IDisposable
        {
            public MemoryBuffer1D<ulong, Stride1D.Dense> Input { get; private set; }
            public MemoryBuffer1D<byte, Stride1D.Dense> Output { get; private set; }
            public int Capacity { get; private set; }

            public ScratchBuffers(Accelerator accel, int capacity)
            {
                Capacity = Math.Max(1, capacity);
                Input = accel.Allocate1D<ulong>(Capacity);
                Output = accel.Allocate1D<byte>(Capacity);
            }

            public void Dispose()
            {
                Output.Dispose();
                Input.Dispose();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ScratchBuffers RentScratch(int minCapacity, Accelerator accel)
        {
            minCapacity = Math.Max(1, minCapacity);
            while (_scratchPool.TryTake(out var sb))
            {
                if (sb.Capacity >= minCapacity)
                {
                    return sb;
                }

                sb.Dispose();
            }

            return new ScratchBuffers(accel, minCapacity);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void ReturnScratch(ScratchBuffers scratch)
        {
            _scratchPool.Add(scratch);
        }

        public void Clear()
        {
            while (_scratchPool.TryTake(out var sb))
            {
                sb.Dispose();
            }

            DevicePrimes.Dispose();
        }
    }

    internal static class GpuKernelState
    {
        // Map accelerator to cached state; use Lazy to serialize kernel creation
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<Accelerator, Lazy<KernelState>> States = new();
        // TODO: Prewarm this per-accelerator cache during startup (and reuse a simple array keyed by accelerator index)
        // once the kernel pool exposes deterministic ordering; the Lazy wrappers showed measurable overhead in the
        // GpuModularArithmeticBenchmarks hot path.

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static KernelState GetOrCreate(Accelerator accelerator)
        {
            var lazy = States.GetOrAdd(
                accelerator,
                acc => new Lazy<KernelState>(() => new KernelState(acc), System.Threading.LazyThreadSafetyMode.ExecutionAndPublication));
            return lazy.Value;
        }

        public static void Clear(Accelerator accelerator)
        {
            if (States.TryRemove(accelerator, out var lazy))
            {
                if (lazy.IsValueCreated)
                {
                    var state = lazy.Value;
                    state.Clear();
                }
            }
        }
    }

    // Expose cache clearing for accelerator disposal coordination
    public static void ClearGpuCaches(Accelerator accelerator)
    {
        GpuKernelState.Clear(accelerator);
    }

    internal static void DisposeGpuContexts()
    {
        PrimeTesterGpuContextPool.DisposeAll();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool SharesFactorWithMaxExponent(ulong n)
    {
        // TODO: Replace this on-the-fly GCD probe with the cached factor table derived from
        // ResidueComputationBenchmarks so divisor-cycle metadata can short-circuit the test
        // instead of recomputing binary GCD for every candidate.
        ulong m = (ulong)BitOperations.Log2(n);
        return BinaryGcd(n, m) != 1UL;
    }

    internal static void SharesFactorWithMaxExponentBatch(ReadOnlySpan<ulong> values, Span<byte> results)
    {
        // TODO: Route this batch helper through the shared GPU kernel pool from
        // GpuUInt128BinaryGcdBenchmarks so we reuse cached kernels, pinned host buffers,
        // and divisor-cycle staging instead of allocating new device buffers per call.
        var gpu = PrimeTesterGpuContextPool.Rent();
        var accelerator = gpu.Accelerator;
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<byte>>(PrimeTesterKernels.SharesFactorKernel);

        int length = values.Length;
        var inputBuffer = accelerator.Allocate1D<ulong>(length);
        var resultBuffer = accelerator.Allocate1D<byte>(length);

        ArrayPool<ulong> pool = ThreadStaticPools.UlongPool;
        ulong[] temp = pool.Rent(length);
        values.CopyTo(temp);
        inputBuffer.View.CopyFromCPU(ref temp[0], length);
        kernel(length, inputBuffer.View, resultBuffer.View);
        accelerator.Synchronize();
        resultBuffer.View.CopyToCPU(ref results[0], length);
        pool.Return(temp, clearArray: false);
        resultBuffer.Dispose();
        inputBuffer.Dispose();
        gpu.Dispose();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong BinaryGcd(ulong u, ulong v)
    {
        // TODO: Swap this handwritten binary GCD for the optimized helper measured in
        // GpuUInt128BinaryGcdBenchmarks so CPU callers share the faster subtract-less
        // ladder once the common implementation is promoted into PerfectNumbers.Core.
        if (u == 0UL)
        {
            return v;
        }

        if (v == 0UL)
        {
            return u;
        }

        int shift = BitOperations.TrailingZeroCount(u | v);
        u >>= BitOperations.TrailingZeroCount(u);

        do
        {
            v >>= BitOperations.TrailingZeroCount(v);
            if (u > v)
            {
                (u, v) = (v, u);
            }

            v -= u;
        }
        while (v != 0UL);

        return u << shift;
    }
    }
