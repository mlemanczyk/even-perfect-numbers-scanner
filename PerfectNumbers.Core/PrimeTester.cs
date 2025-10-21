using System.Buffers;
using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

public sealed class PrimeTester(bool useInternal = false)
{
    public bool IsPrime(ulong n, CancellationToken ct) =>
        // TODO: Eliminate this wrapper once callers can reach IsPrimeInternal directly; the extra
        // indirection shows up when we sieve millions of candidates per second.
        IsPrimeInternal(n, ct);

    public static bool IsPrimeGpu(ulong n) => new PrimeTester().IsPrimeGpu(n, CancellationToken.None);

    // Optional GPU-assisted primality: batched small-prime sieve on device.
    public bool IsPrimeGpu(ulong n, CancellationToken ct)
    {
        // EvenPerfectBitScanner never enqueues exponents <= 3 on the GPU path; keep the legacy guard commented out
        // so ad-hoc callers that expect that behavior must re-enable it intentionally.
        // if (n <= 3UL)
        // {
        //     return n >= 2UL;
        // }

        // Candidate generation (Add/AddPrimes transforms plus ModResidueTracker) already strips even values and multiples of five.
        // The downstream small-factor sweep and Pollard routines factor divisors q = 2kp + 1 with k >= 1 without feeding new
        // exponents back into this path, so the GPU sees the same odd sequence that already passed the CPU residues.

        // EvenPerfectBitScanner seeds p at 136,279,841 and advances monotonically, so production scans never hit the GPU with
        // exponents below 31. Leave this CPU redirect commented out to keep the wrapper branchless while documenting the guard.
        // The ternary fallback below still routes sub-31 values through IsPrimeInternal for correctness when ad-hoc callers bypass
        // the scanner pipeline.
        // if (n < 31UL)
        // {
        //     return IsPrimeInternal(n, ct);
        // }

        bool forceCpu = GpuContextPool.ForceCpu;
        Span<ulong> one = stackalloc ulong[1];
        Span<byte> outFlags = stackalloc byte[1];
        // TODO: Inline the single-value GPU sieve fast path from GpuModularArithmeticBenchmarks so this wrapper
        // can skip stackalloc buffers and reuse the pinned upload span the benchmark identified as fastest.
        one[0] = n;
        outFlags[0] = 0;

        if (!forceCpu)
        {
            IsPrimeBatchGpu(one, outFlags);
        }

        bool belowGpuRange = n < 31UL;
        bool gpuReportedPrime = !forceCpu && !belowGpuRange && outFlags[0] != 0;
        bool requiresCpuFallback = forceCpu || belowGpuRange || !gpuReportedPrime;

        // Defensive fallback: GPU sieve produced a composite verdict or execution was skipped.
        // Confirm with CPU logic to prevent false negatives observed on some accelerators.
        return requiresCpuFallback ? IsPrimeInternal(n, ct) : true;
    }


    // Note: GPU-backed primality path is implemented via IsPrimeGpu/IsPrimeBatchGpu and is routed
    // from EvenPerfectBitScanner based on --primes-device.

    internal static bool IsPrimeInternal(ulong n, CancellationToken ct)
    {
        // EvenPerfectBitScanner streams monotonically increasing odd exponents that start at 136,279,841 and already exclude
        // multiples of five. Factorization helpers (PrimeOrderCalculator, Pollard routines, and divisor-cycle warmups) reuse
        // this path for arbitrary residues and cofactors, so keep the boolean guards even though the scanner never trips them.
        bool isTwo = n == 2UL;
        bool isOdd = (n & 1UL) != 0UL;
        // TODO: Replace this modulo check with ULongExtensions.Mod5 so the CPU hot path reuses the benchmarked helper instead of `%`.
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

                    // TODO: Route this small-prime filtering through the shared divisor-cycle cache once
                    // PrimeTester can consult it directly; the cached cycles avoid the repeated `%` work
                    // that slows these hot loops when sieving hundreds of millions of candidates.
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

    public static int GpuBatchSize { get; set; } = 262_144;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void IsPrimeBatchGpu(ReadOnlySpan<ulong> values, Span<byte> results)
    {
        // Limit concurrency and declare variables outside loops for performance and reuse
        var limiter = GpuPrimeWorkLimiter.Acquire();
        var gpu = PrimeTesterGpuContextPool.Rent();

        try
        {
            var accelerator = gpu.Accelerator;

            // Get per-accelerator cached kernel and device primes buffer.
            var state = GpuKernelState.GetOrCreate(accelerator);
            int totalLength = values.Length;
            int batchSize = Math.Max(1, GpuBatchSize);

            lock (gpu.ExecutionLock)
            {
                var scratch = state.RentScratch(batchSize, accelerator);
                var input = scratch.Input;
                var output = scratch.Output;
                ulong[] temp = ArrayPool<ulong>.Shared.Rent(batchSize);
                // TODO: Replace this ad-hoc ArrayPool buffer with the pinned span cache from
                // PrimeSieveGpuBenchmarks so batch uploads reuse preallocated GPU-friendly
                // memory and avoid the extra copy before every kernel launch.

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

    private static class PrimeTesterGpuContextPool
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
    private sealed class KernelState
    {
        public Action<Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<byte>> Kernel { get; }
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

    private static class GpuKernelState
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
