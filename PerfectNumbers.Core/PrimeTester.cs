using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

public sealed class PrimeTester
{
    public PrimeTester(bool useInternal = false)
    {
    }

    [ThreadStatic]
    private static PrimeTester? _tester;

    public static PrimeTester Exclusive => _tester ??= new();

    public bool IsPrime(ulong n, CancellationToken ct)
    {
        return IsPrimeInternal(n, ct);
    }

    public static bool IsPrimeGpu(ulong n)
    {
        return Exclusive.IsPrimeGpu(n, CancellationToken.None);
    }

    public static bool IsPrimeGpu(ulong n, ulong limit, byte nMod10)
    {
        return Exclusive.IsPrimeGpu(n, CancellationToken.None);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool IsPrimeGpu(ulong n, CancellationToken ct)
    {
        return IsPrimeGpuFallback(n, ct);
    }

    public static bool IsPrimeGpuFallback(ulong n, CancellationToken ct)
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

        return requiresCpuFallback ? IsPrimeInternal(n, ct) : true;
    }

    public static bool IsPrimeInternal(ulong n, CancellationToken ct)
    {
        bool isTwo = n == 2UL;
        bool isOdd = (n & 1UL) != 0UL;
        ulong mod5 = n % 5UL;
        bool divisibleByFive = n > 5UL && mod5 == 0UL;

        bool result = n >= 2UL && (isTwo || isOdd) && !divisibleByFive;
        bool requiresTrialDivision = result && n >= 7UL && !isTwo;

        if (requiresTrialDivision)
        {
            bool sharesMaxExponentFactor = n.Mod10() == 1UL && SharesFactorWithMaxExponent(n);
            result &= !sharesMaxExponentFactor;

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


    internal readonly struct HeuristicDivisorCandidate
    {
        private readonly HeuristicPrimeTester.HeuristicDivisorCandidate _inner;

        internal HeuristicDivisorCandidate(HeuristicPrimeTester.HeuristicDivisorCandidate inner)
        {
            _inner = inner;
        }

        public ulong Value => _inner.Value;

        public HeuristicPrimeTester.HeuristicDivisorGroup Group => _inner.Group;

        public byte Ending => _inner.Ending;

        public byte PriorityIndex => _inner.PriorityIndex;

        public ushort WheelResidue => _inner.WheelResidue;

        internal HeuristicPrimeTester.HeuristicDivisorCandidate AsHeuristic() => _inner;
    }

    internal readonly struct HeuristicDivisorPreparation
    {
        private readonly HeuristicPrimeTester.HeuristicDivisorPreparation _inner;

        internal HeuristicDivisorPreparation(HeuristicPrimeTester.HeuristicDivisorPreparation inner)
        {
            _inner = inner;
        }

        public HeuristicDivisorCandidate Candidate => new(_inner.Candidate);

        public MontgomeryDivisorData DivisorData => _inner.DivisorData;

        public ulong CycleLengthHint => _inner.CycleLengthHint;

        public bool HasCycleLengthHint => _inner.HasCycleLengthHint;

        public bool RequiresCycleComputation => _inner.RequiresCycleComputation;

        internal HeuristicPrimeTester.HeuristicDivisorPreparation AsHeuristic() => _inner;
    }

    internal ref struct MersenneHeuristicDivisorEnumerator
    {
        private HeuristicPrimeTester.MersenneHeuristicDivisorEnumerator _inner;

        public MersenneHeuristicDivisorEnumerator(ulong exponent, ulong maxDivisor)
        {
            _inner = HeuristicPrimeTester.CreateMersenneDivisorEnumerator(exponent, maxDivisor);
        }

        public ulong ProcessedCount => _inner.ProcessedCount;

        public ulong LastDivisor => _inner.LastDivisor;

        public bool Exhausted => _inner.Exhausted;

        public bool TryGetNext(out HeuristicDivisorCandidate candidate)
        {
            if (_inner.TryGetNext(out var innerCandidate))
            {
                candidate = new HeuristicDivisorCandidate(innerCandidate);
                return true;
            }

            candidate = default;
            return false;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static MersenneHeuristicDivisorEnumerator CreateMersenneDivisorEnumerator(ulong exponent, ulong maxDivisor)
    {
        return new MersenneHeuristicDivisorEnumerator(exponent, maxDivisor);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static HeuristicDivisorPreparation PrepareHeuristicDivisor(in HeuristicDivisorCandidate candidate)
    {
        var innerCandidate = candidate.AsHeuristic();
        var preparation = HeuristicPrimeTester.PrepareHeuristicDivisor(in innerCandidate);
        return new HeuristicDivisorPreparation(preparation);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ulong ResolveHeuristicCycleLength(
        ulong exponent,
        in HeuristicDivisorPreparation preparation,
        out bool cycleFromHint,
        out bool cycleComputed,
        out bool primeOrderFailed)
    {
        var innerPreparation = preparation.AsHeuristic();
        return HeuristicPrimeTester.ResolveHeuristicCycleLength(
            exponent,
            in innerPreparation,
            out cycleFromHint,
            out cycleComputed,
            out primeOrderFailed);
    }
    public static int GpuBatchSize { get; set; } = 262_144;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void IsPrimeBatchGpu(ReadOnlySpan<ulong> values, Span<byte> results)
    {
        var limiter = GpuPrimeWorkLimiter.Acquire();
        var gpu = PrimeTesterGpuContextPool.Rent();

        try
        {
            var accelerator = gpu.Accelerator;
            var state = GpuKernelState.GetOrCreate(accelerator);
            int totalLength = values.Length;
            int batchSize = Math.Max(1, GpuBatchSize);

            lock (gpu.ExecutionLock)
            {
                var scratch = state.RentScratch(batchSize, accelerator);
                var input = scratch.Input;
                var output = scratch.Output;
                ulong[] temp = ArrayPool<ulong>.Shared.Rent(batchSize);

                try
                {
                    int pos = 0;
                    while (pos < totalLength)
                    {
                        int remaining = totalLength - pos;
                        int count = remaining > batchSize ? batchSize : remaining;

                        values.Slice(pos, count).CopyTo(temp);
                        input.View.CopyFromCPU(ref temp[0], count);

                        state.Kernel(count, input.View, state.DevicePrimes.View, output.View);
                        accelerator.Synchronize();
                        output.View.CopyToCPU(ref results[pos], count);

                        pos += count;
                    }
                }
                finally
                {
                    ArrayPool<ulong>.Shared.Return(temp);
                    state.ReturnScratch(scratch);
                }
            }
        }
        finally
        {
            gpu.Dispose();
            limiter.Dispose();
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

        ulong[] temp = ArrayPool<ulong>.Shared.Rent(length);
        values.CopyTo(temp);
        inputBuffer.View.CopyFromCPU(ref temp[0], length);
        kernel(length, inputBuffer.View, resultBuffer.View);
        accelerator.Synchronize();
        resultBuffer.View.CopyToCPU(ref results[0], length);
        ArrayPool<ulong>.Shared.Return(temp);
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
