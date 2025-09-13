using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

public sealed class PrimeTester(bool useInternal = false)
{
    public bool IsPrime(ulong n, CancellationToken ct) => IsPrimeInternal(n, ct);

    // Optional GPU-assisted primality: batched small-prime sieve on device.
    // TODO: Consider extending with deterministic Miller–Rabin bases for 64-bit range when stable on ILGPU.
    public bool IsPrimeGpu(ulong n, CancellationToken ct)
    {
        if (GpuContextPool.ForceCpu)
        {
            return IsPrimeInternal(n, ct);
        }

        Span<ulong> one = stackalloc ulong[1];
        Span<byte> outFlags = stackalloc byte[1];
        one[0] = n;
        IsPrimeBatchGpu(one, outFlags);
        return outFlags[0] != 0;
    }

    // Note: GPU-backed primality path is implemented via IsPrimeGpu/IsPrimeBatchGpu and is routed
    // from EvenPerfectBitScanner based on --primes-device. Follow-up: add deterministic MR rounds.

    internal static bool IsPrimeInternal(ulong n, CancellationToken ct)
    {
        bool result = true;
        if (n <= 3UL)
        {
            result = n >= 2UL;
        }
        else if ((n & 1UL) == 0)
        {
            result = false;
        }
        else if (n > 5UL && (n % 5UL) == 0UL)
        {
            result = false;
        }
        else
        {
            if ((n % 10UL) == 1UL && SharesFactorWithMaxExponent(n))
            {
                result = false;
            }
            else
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

    public static int GpuBatchSize { get; set; } = 262_144;

    public static void IsPrimeBatchGpu(ReadOnlySpan<ulong> values, Span<byte> results)
    {
        // Limit concurrency and declare variables outside loops for performance and reuse
        var limiter = GpuPrimeWorkLimiter.Acquire();
        var gpu = GpuContextPool.Rent();
        var accelerator = gpu.Accelerator;

        // Get per-accelerator cached kernel and device primes buffer.
        var state = GpuKernelState.GetOrCreate(accelerator);

        MemoryBuffer1D<ulong, Stride1D.Dense> input;
        MemoryBuffer1D<byte, Stride1D.Dense> output;
        int totalLength, batchSize, pos, remaining, count;
        ulong[] temp;

        totalLength = values.Length;
        batchSize = Math.Max(1, GpuBatchSize);

        // Rent per-call scratch buffers sized to current batch (thread-safe)
        var scratch = state.RentScratch(batchSize, accelerator);
        input = scratch.Input;
        output = scratch.Output;

        // Reusable host buffer per call
        temp = ArrayPool<ulong>.Shared.Rent(batchSize);

        pos = 0;
        while (pos < totalLength)
        {
            remaining = totalLength - pos;
            count = remaining > batchSize ? batchSize : remaining;

            // Copy chunk to host temp then to device
            values.Slice(pos, count).CopyTo(temp);
            input.View.CopyFromCPU(ref temp[0], count);

            state.Kernel(count, input.View, state.DevicePrimes.View, output.View);
            accelerator.Synchronize();
            output.View.CopyToCPU(ref results[pos], count);

            pos += count;
        }

        ArrayPool<ulong>.Shared.Return(temp);
        state.ReturnScratch(scratch);
        gpu.Dispose();
        limiter.Dispose();
    }

    // Per-accelerator GPU state for prime sieve (kernel + uploaded primes).
    private sealed class KernelState
    {
        public Action<Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<byte>> Kernel { get; }
        public MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimes { get; }
        private readonly Accelerator _accel;
        private readonly System.Collections.Concurrent.ConcurrentBag<ScratchBuffers> _scratchPool = new();

        public KernelState(Accelerator accelerator)
        {
            _accel = accelerator;
            // Compile once per accelerator and upload primes once.
            Kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<byte>>(SmallPrimeSieveKernel);

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

    // GPU kernel: small-prime sieve only. Returns 1 if passes sieve (probable prime), 0 otherwise.
    // TODO: Follow-up: consider integration with MR rounds for bases {2,3,5,7,11,13} to make it deterministic for 64-bit.
    private static void SmallPrimeSieveKernel(Index1D index, ArrayView<ulong> numbers, ArrayView<uint> smallPrimes, ArrayView<byte> results)
    {
        ulong n = numbers[index];
        if (n <= 3UL)
        {
            results[index] = (byte)(n >= 2UL ? 1 : 0);
            return;
        }

        if ((n & 1UL) == 0UL || (n % 5UL) == 0UL)
        {
            results[index] = 0;
            return;
        }

        // Early reject special gcd heuristic with floor(log2 n)
        ulong m = 63UL - (ulong)ILGPU.Algorithms.XMath.LeadingZeroCount(n);
        if (BinaryGcdGpu(n, m) != 1UL)
        {
            results[index] = 0;
            return;
        }

        long len = smallPrimes.Length;
        for (int i = 0; i < len; i++)
        {
            ulong p = smallPrimes[i];
			if (p * p > n)
			{
				break;
			}
			
			if ((n % p) == 0UL)
			{
				results[index] = 0;
				return;
			}
        }

        results[index] = 1;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool SharesFactorWithMaxExponent(ulong n)
    {
        ulong m = (ulong)BitOperations.Log2(n);
        return BinaryGcd(n, m) != 1UL;
    }

    internal static void SharesFactorWithMaxExponentBatch(ReadOnlySpan<ulong> values, Span<byte> results)
    {
        var gpu = GpuContextPool.Rent();
        var accelerator = gpu.Accelerator;
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<byte>>(SharesFactorKernel);

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

    private static void SharesFactorKernel(Index1D index, ArrayView<ulong> numbers, ArrayView<byte> results)
    {
        ulong n = numbers[index];
        ulong m = 63UL - (ulong)XMath.LeadingZeroCount(n);
        ulong gcd = BinaryGcdGpu(n, m);
        results[index] = gcd == 1UL ? (byte)0 : (byte)1;
    }

    private static ulong BinaryGcdGpu(ulong u, ulong v)
    {
        if (u == 0UL)
        {
            return v;
        }

        if (v == 0UL)
        {
            return u;
        }

        int shift = XMath.TrailingZeroCount(u | v);
        u >>= XMath.TrailingZeroCount(u);

        do
        {
            v >>= XMath.TrailingZeroCount(v);
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
