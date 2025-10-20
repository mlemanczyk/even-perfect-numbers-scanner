using System.Buffers;
using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;
using UInt128 = System.UInt128;

namespace PerfectNumbers.Core.Gpu;

public class MersenneNumberLucasLehmerGpuTester
{
    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ulong, GpuUInt128, ArrayView<GpuUInt128>>> KernelCache = new();
    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ulong, ArrayView<GpuUInt128>>> AddSmallKernelCache = new();
    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ulong, ArrayView<GpuUInt128>>> SubSmallKernelCache = new();
    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ulong, ArrayView<GpuUInt128>>> ReduceKernelCache = new();
    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<GpuUInt128>, ArrayView<byte>>> IsZeroKernelCache = new();
    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<ulong>, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>>> BatchKernelCache = new();

    private Action<Index1D, ulong, GpuUInt128, ArrayView<GpuUInt128>> GetKernel(Accelerator accelerator)
    {
        return KernelCache.GetOrAdd(accelerator, accel => accel.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, ArrayView<GpuUInt128>>(LucasLehmerKernels.Kernel));
    }

    private Action<Index1D, ulong, ArrayView<GpuUInt128>> GetAddSmallKernel(Accelerator accelerator) =>
        AddSmallKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ulong, ArrayView<GpuUInt128>>(LucasLehmerKernels.AddSmallKernel));

    private Action<Index1D, ulong, ArrayView<GpuUInt128>> GetSubSmallKernel(Accelerator accelerator) =>
        SubSmallKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ulong, ArrayView<GpuUInt128>>(LucasLehmerKernels.SubtractSmallKernel));

    private Action<Index1D, ulong, ArrayView<GpuUInt128>> GetReduceKernel(Accelerator accelerator) =>
        ReduceKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ulong, ArrayView<GpuUInt128>>(LucasLehmerKernels.ReduceModMersenneKernel));

    private Action<Index1D, ArrayView<GpuUInt128>, ArrayView<byte>> GetIsZeroKernel(Accelerator accelerator) =>
        IsZeroKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, ArrayView<byte>>(LucasLehmerKernels.IsZeroKernel));

    private Action<Index1D, ArrayView<ulong>, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>> GetBatchKernel(Accelerator accelerator) =>
        BatchKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>>(LucasLehmerKernels.KernelBatch));

    // Configurable LL slice size to keep kernels short. Default 32.
    public int SliceSize = 32;

    public bool IsMersennePrime(ulong exponent)
    {
        // Default to global kernel preference for backward compatibility.
        bool runOnGpu = !GpuContextPool.ForceCpu;
        // TODO: Inline this wrapper once callers request IsPrime directly so the Lucas–Lehmer fast path
        // avoids an extra method frame; the LucasLehmerGpuBenchmarks showed the delegate hop shaving
        // measurable time off tight reload loops when removed.
        return IsPrime(exponent, runOnGpu);
    }

    public bool IsPrime(ulong exponent, bool runOnGpu)
    {
        // Early rejections aligned with incremental/order sieves, but safe for small p:
        // - If 3 | p and p != 3, then 7 | M_p -> composite.
        // - If p ≡ 1 (mod 4) and p shares a factor with (p-1), reject fast.
        // TODO: Replace this `% 3` guard with ULongExtensions.Mod3 once GPU LL filtering reuses the benchmarked bitmask helper.
        if ((exponent % 3UL) == 0UL && exponent != 3UL)
        {
            return false;
        }

        if ((exponent & 3UL) == 1UL && exponent.SharesFactorWithExponentMinusOne())
        {
            return false;
        }

        var limiter = GpuPrimeWorkLimiter.Acquire();
        bool result;
        if (exponent >= 128UL)
        {
            if (!TryGetNttParameters(exponent, out var nttMod, out var primitiveRoot))
            {
                limiter.Dispose();
                throw new NotSupportedException("NTT parameters for the given exponent are not available.");
            }

            result = IsMersennePrimeNtt(exponent, nttMod, primitiveRoot, runOnGpu);
        }
        else
        {
            var gpu = GpuContextPool.RentPreferred(preferCpu: !runOnGpu);
            var accelerator = gpu.Accelerator;
            var kernel = GetKernel(accelerator);
            var modulus = new GpuUInt128(((UInt128)1 << (int)exponent) - 1UL); // TODO: Cache these Mersenne moduli per exponent so LL GPU runs skip rebuilding them every launch.
            var buffer = accelerator.Allocate1D<GpuUInt128>(1);
            kernel(1, exponent, modulus, buffer.View);
            accelerator.Synchronize();
            result = buffer.GetAsArray1D()[0].IsZero;
            buffer.Dispose();
            gpu.Dispose();
        }

        limiter.Dispose();
        return result;
    }

    public void ComputeResidues(ReadOnlySpan<ulong> exponents, Span<GpuUInt128> residues)
    {
        int count = exponents.Length;
        if (residues.Length < count)
        {
            throw new ArgumentException("Result span too small", nameof(residues));
        }

        for (int i = 0; i < count; i++)
        {
            if (exponents[i] >= 128UL)
            {
                throw new NotSupportedException("Batch residue calculation supports exponents < 128 only.");
            }
        }

        var gpu = GpuContextPool.RentPreferred(preferCpu: false);
        var accelerator = gpu.Accelerator;
        var kernel = GetBatchKernel(accelerator);

        var expBuffer = accelerator.Allocate1D<ulong>(count);
        ulong[] expArray = ArrayPool<ulong>.Shared.Rent(count);
        exponents.CopyTo(expArray);
        expBuffer.View.CopyFromCPU(ref expArray[0], count);

        GpuUInt128[] modulusArray = ArrayPool<GpuUInt128>.Shared.Rent(count);
        for (int i = 0; i < count; i++)
        {
            // TODO: Replace this per-exponent shift with a small shared table (one entry per supported exponent < 128)
            // so Lucas–Lehmer batch runs reuse cached GpuUInt128 moduli instead of rebuilding them for every request.
            modulusArray[i] = new GpuUInt128(((UInt128)1 << (int)exponents[i]) - 1UL);
        }

        var modBuffer = accelerator.Allocate1D<GpuUInt128>(count);
        modBuffer.View.CopyFromCPU(ref modulusArray[0], count);

        var stateBuffer = accelerator.Allocate1D<GpuUInt128>(count);
        kernel(count, expBuffer.View, modBuffer.View, stateBuffer.View);
        accelerator.Synchronize();
        stateBuffer.View.CopyToCPU(ref residues[0], count);

        stateBuffer.Dispose();
        modBuffer.Dispose();
        expBuffer.Dispose();
        ArrayPool<GpuUInt128>.Shared.Return(modulusArray);
        ArrayPool<ulong>.Shared.Return(expArray);
        gpu.Dispose();
    }

    public void IsMersennePrimeBatch(ReadOnlySpan<ulong> exponents, Span<bool> results)
    {
        int count = exponents.Length;
        if (results.Length < count)
        {
            throw new ArgumentException("Result span too small", nameof(results));
        }

        GpuUInt128[] buffer = ArrayPool<GpuUInt128>.Shared.Rent(count);
        try
        {
            ComputeResidues(exponents, buffer.AsSpan(0, count));
            for (int i = 0; i < count; i++)
            {
                results[i] = buffer[i].IsZero;
            }
        }
        finally
        {
            ArrayPool<GpuUInt128>.Shared.Return(buffer);
        }
    }

    private bool IsMersennePrimeNtt(ulong exponent, GpuUInt128 nttMod, GpuUInt128 primitiveRoot, bool runOnGpu)
    {
        var gpu = GpuContextPool.RentPreferred(preferCpu: !runOnGpu);
        var accelerator = gpu.Accelerator;
        var addKernel = GetAddSmallKernel(accelerator);
        var subKernel = GetSubSmallKernel(accelerator);
        var reduceKernel = GetReduceKernel(accelerator);
        var zeroKernel = GetIsZeroKernel(accelerator);

        int limbCount = (int)((exponent + 127UL) / 128UL);
        int target = limbCount * 2;
        int transformLength = 1;
        while (transformLength < target)
        {
            transformLength <<= 1;
        }
        var stateBuffer = accelerator.Allocate1D<GpuUInt128>(transformLength);
        stateBuffer.MemSetToZero();
        addKernel(1, 4UL, stateBuffer.View);

        // TODO(LL-SLICE): Slice Lucas–Lehmer iterations into short batches to
        // avoid long-running kernels and TDR. Example: process 8–64 iterations
        // per slice, with synchronization between slices. Combine this with
        // stage-wise NTT to ensure each kernel stays < ~0.5–1.0s.
        ulong i = 0UL;
        ulong limit = exponent - 2UL;
        int slice = Math.Max(1, SliceSize); // process LL iterations in small slices
        while (i < limit)
        {
            int iter = (int)Math.Min((ulong)slice, limit - i);
            for (int s = 0; s < iter; s++)
            {
                // Square in NTT domain with staged kernels (short runtime per kernel).
                NttGpuMath.SquareDevice(accelerator, stateBuffer.View, nttMod, primitiveRoot);
                subKernel(1, 2UL, stateBuffer.View);
                reduceKernel(1, exponent, stateBuffer.View);
            }

            // Synchronize between slices to yield to scheduler and avoid TDR.
            accelerator.Synchronize();
            i += (ulong)iter;
        }

        reduceKernel(1, exponent, stateBuffer.View);

        var resultBuffer = accelerator.Allocate1D<byte>(1);
        zeroKernel(1, stateBuffer.View, resultBuffer.View);
        accelerator.Synchronize();
        byte[] result = new byte[1];
        resultBuffer.View.CopyToCPU(result);
        bool isPrime = result[0] != 0;
        resultBuffer.Dispose();
        stateBuffer.Dispose();
        gpu.Dispose();
        return isPrime;
    }

    private static readonly ConcurrentDictionary<int, (GpuUInt128 Modulus, GpuUInt128 PrimitiveRoot)> NttParameterCache = new()
    {
        // 2^12 transform length using a 64-bit NTT-friendly prime
        [4096] = (new GpuUInt128(0UL, 18446744069414584321UL), new GpuUInt128(0UL, 7UL)),
        // 2^22 transform length precomputed for p ~= 138,000,000
        [4194304] = (new GpuUInt128(0UL, 104857601UL), new GpuUInt128(0UL, 39193363UL))
    };

    private readonly object ParameterFileLock = new();
    private readonly string ParameterFilePath = Path.Combine(AppContext.BaseDirectory, "ntt-params.txt");

    public void WarmUpNttParameters(ulong exponent)
    {
        _ = TryGetNttParameters(exponent, out _, out _);
    }

    private bool TryGetNttParameters(ulong exponent, out GpuUInt128 modulus, out GpuUInt128 primitiveRoot)
    {
        int limbCount = (int)((exponent + 127UL) / 128UL);
        int length = 1;
        int target = limbCount * 2;
        while (length < target)
        {
            length <<= 1;
        }

        if (NttParameterCache.TryGetValue(length, out var cached))
        {
            modulus = cached.Modulus;
            primitiveRoot = cached.PrimitiveRoot;
            return true;
        }

        if (TryLoadPersistedParameters(length, out modulus, out primitiveRoot))
        {
            NttParameterCache[length] = (modulus, primitiveRoot);
            return true;
        }

        if (GenerateNttParameters(length, out modulus, out primitiveRoot))
        {
            NttParameterCache[length] = (modulus, primitiveRoot);
            PersistParameters(length, modulus, primitiveRoot);
            return true;
        }

        modulus = new GpuUInt128(0UL, 18446744069414584321UL); // 2^64 - 2^32 + 1
        primitiveRoot = new GpuUInt128(0UL, 7UL);
        return false;
    }

    private bool TryLoadPersistedParameters(int length, out GpuUInt128 modulus, out GpuUInt128 primitiveRoot)
    {
        if (!File.Exists(ParameterFilePath))
        {
            modulus = new GpuUInt128(0UL, 0UL);
            primitiveRoot = new GpuUInt128(0UL, 0UL);
            return false;
        }

        foreach (var line in File.ReadLines(ParameterFilePath))
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length != 5)
            {
                continue;
            }

            // TODO: Replace int.TryParse with the span-based Utf8Parser helper when loading cached parameters so we avoid
            // culture-aware parsing in this hot startup loop.
            if (!int.TryParse(parts[0], out var storedLength) || storedLength != length)
            {
                continue;
            }

            // TODO: Switch these ulong.Parse calls to the Utf8Parser-based fast-path once we expose a zero-allocation reader for
            // persisted kernel parameters.
            ulong modHigh = ulong.Parse(parts[1]);
            ulong modLow = ulong.Parse(parts[2]);
            ulong rootHigh = ulong.Parse(parts[3]);
            ulong rootLow = ulong.Parse(parts[4]);
            modulus = new GpuUInt128(modHigh, modLow);
            primitiveRoot = new GpuUInt128(rootHigh, rootLow);
            return true;
        }

        modulus = new GpuUInt128(0UL, 0UL);
        primitiveRoot = new GpuUInt128(0UL, 0UL);
        return false;
    }

    private void PersistParameters(int length, GpuUInt128 modulus, GpuUInt128 primitiveRoot)
    {
        lock (ParameterFileLock)
        {
            // TODO: Replace StreamWriter with the pooled TextFileWriter pipeline so persisting NTT
            // parameters reuses the zero-allocation buffered writes highlighted in the scanner I/O
            // benchmarks instead of allocating a new encoder per append.
            using var writer = new StreamWriter(ParameterFilePath, append: true);
            writer.WriteLine($"{length} {modulus.High} {modulus.Low} {primitiveRoot.High} {primitiveRoot.Low}");
        }
    }

    private static bool GenerateNttParameters(int length, out GpuUInt128 modulus, out GpuUInt128 primitiveRoot)
    {
        ulong foundCandidate = 0UL;
        ulong foundRoot = 0UL;
        bool success = false;
        var processorCount = Environment.ProcessorCount;
        using var cts = new System.Threading.CancellationTokenSource();
        object sync = new();

        // TODO: Move this Parallel.For to the shared low-overhead work scheduler once the NTT parameter
        // generator integrates with the GPU-first pipeline so parameter scans reuse the same batching
        // strategy measured fastest in the divisor-cycle benchmarks.
        System.Threading.Tasks.Parallel.For(0, processorCount, (worker, state) =>
        {
            ulong k = (ulong)worker + 1UL;
            while (!cts.IsCancellationRequested)
            {
                ulong candidate;
                try
                {
                    candidate = checked((ulong)length * k + 1UL);
                }
                catch (OverflowException)
                {
                    break;
                }

                if (IsPrime(candidate))
                {
                    ulong root = FindPrimitiveRoot(candidate, (ulong)length);
                    lock (sync)
                    {
                        foundCandidate = candidate;
                        foundRoot = root;
                        success = true;
                        cts.Cancel();
                        state.Stop();
                    }
                    break;
                }

                k += (ulong)processorCount;
            }
        });

        if (success)
        {
            modulus = new GpuUInt128(0UL, foundCandidate);
            primitiveRoot = new GpuUInt128(0UL, foundRoot);
            return true;
        }

        modulus = new GpuUInt128(0UL, 18446744069414584321UL);
        primitiveRoot = new GpuUInt128(0UL, 7UL);
        return false;
    }

    private static ulong FindPrimitiveRoot(ulong modulus, ulong order)
    {
        ulong phi = modulus - 1UL;
        // TODO: Reuse the divisor-cycle cache to factor phi via the precomputed small-prime windows once
        // the lookup tables land so primitive root searches stop iterating over slow trial divisions.
        var factors = Factorize(phi);
        for (ulong g = 2UL; g < modulus; g++)
        {
            bool ok = true;
            foreach (ulong f in factors)
            {
                if (ModPow(g, phi / f, modulus) == 1UL)
                {
                    ok = false;
                    break;
                }
            }

            if (ok)
            {
                return ModPow(g, phi / order, modulus);
            }
        }

        throw new InvalidOperationException("Primitive root not found.");
    }

    private static List<ulong> Factorize(ulong n)
    {
        var factors = new List<ulong>();
        for (ulong p = 2UL; p * p <= n; p += p == 2UL ? 1UL : 2UL)
        {
            // TODO: Replace these `%` factor checks with the shared Mod helpers (Mod3/Mod5/etc.) once the GPU
            // pre-filter adopts the benchmarked bitmask operations to avoid slow modulo instructions.
            if (n % p == 0UL)
            {
                factors.Add(p);
                while (n % p == 0UL)
                {
                    n /= p;
                }
            }
        }

        if (n > 1UL)
        {
            factors.Add(n);
        }

        return factors;
    }

    private static bool IsPrime(ulong n)
    {
        if (n < 2UL)
        {
            return false;
        }

        if ((n & 1UL) == 0UL)
        {
            return n == 2UL;
        }

        for (ulong divisor = 3UL; divisor * divisor <= n; divisor += 2UL)
        {
            // TODO: Swap this `%` for the divisor-cycle aware Mod helper once the residue pre-checks expose it
            // so primality filtering avoids slow modulo instructions in tight loops.
            if (n % divisor == 0UL)
            {
                return false;
            }
        }

        return true;
    }

    private static ulong MulMod(ulong a, ulong b, ulong modulus)
    {
        // TODO: Swap this UInt128 `%` reduction for the GPU-compatible MulMod helper once it adopts the faster
        // inline UInt128 path benchmarked in MulMod64Benchmarks so host/GPU parity avoids BigInteger-style fallbacks.
        return (ulong)(((UInt128)a * b) % modulus);
    }

    private static ulong ModPow(ulong value, ulong exponent, ulong modulus)
    {
        ulong result = 1UL;
        while (exponent > 0UL)
        {
            if ((exponent & 1UL) != 0UL)
            {
                result = MulMod(result, value, modulus);
            }

            value = MulMod(value, value, modulus);
            exponent >>= 1;
        }

        return result;
    }


}
