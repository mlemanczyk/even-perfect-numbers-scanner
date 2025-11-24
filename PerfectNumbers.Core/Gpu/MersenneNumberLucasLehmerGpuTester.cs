using System.Buffers;
using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu.Accelerators;
using UInt128 = System.UInt128;

namespace PerfectNumbers.Core.Gpu;

public class MersenneNumberLucasLehmerGpuTester
{
	[ThreadStatic]
	private static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ulong, GpuUInt128, ArrayView<GpuUInt128>>>? _lucasLehmerKernel;

	[ThreadStatic]
	private static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ulong, ArrayView<GpuUInt128>>>? _addSmallKernel;

	[ThreadStatic]
	private static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ulong, ArrayView<GpuUInt128>>>? _subSmallKernel;
	
	[ThreadStatic]
	private static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ulong, ArrayView<GpuUInt128>>>? _reduceKernel;

	[ThreadStatic]
    private static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, ArrayView<byte>>>? _isZeroKernel;

	[ThreadStatic]
    private static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView<ulong>, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>>>? _lucasLehmerBatchKernel;

    private static Action<AcceleratorStream, Index1D, ulong, GpuUInt128, ArrayView<GpuUInt128>> GetLucasLehmerKernel(Accelerator accelerator)
	{
		var pool = _lucasLehmerKernel ??= [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}

		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, ArrayView<GpuUInt128>>(LucasLehmerKernels.Kernel);

		var kernel = KernelUtil.GetKernel(loaded);

		cached = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, GpuUInt128, ArrayView<GpuUInt128>>>();
		pool[accelerator] = cached;
		return cached;
	}

	private static Action<AcceleratorStream, Index1D, ulong, ArrayView<GpuUInt128>> GetAddSmallKernel(Accelerator accelerator)
	{
		var pool = _addSmallKernel ??= [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}

		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ulong, ArrayView<GpuUInt128>>(LucasLehmerKernels.AddSmallKernel);

		var kernel = KernelUtil.GetKernel(loaded);

		cached = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, ArrayView<GpuUInt128>>>();
		pool[accelerator] = cached;
		return cached;
	}

	private static Action<AcceleratorStream, Index1D, ulong, ArrayView<GpuUInt128>> GetSubSmallKernel(Accelerator accelerator)
	{
		var pool = _subSmallKernel ??= [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}

		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ulong, ArrayView<GpuUInt128>>(LucasLehmerKernels.SubtractSmallKernel);

		var kernel = KernelUtil.GetKernel(loaded);

		cached = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, ArrayView<GpuUInt128>>>();
		pool[accelerator] = cached;
		return cached;
	}

	private static Action<AcceleratorStream, Index1D, ulong, ArrayView<GpuUInt128>> GetReduceKernel(Accelerator accelerator)
	{
		var pool = _reduceKernel ??= [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}

		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ulong, ArrayView<GpuUInt128>>(LucasLehmerKernels.ReduceModMersenneKernel);

		var kernel = KernelUtil.GetKernel(loaded);

		cached = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, ArrayView<GpuUInt128>>>();
		pool[accelerator] = cached;
		return cached;
	}

	private static Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, ArrayView<byte>> GetIsZeroKernel(Accelerator accelerator)
	{
		var pool = _isZeroKernel ??= [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}

		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, ArrayView<byte>>(LucasLehmerKernels.IsZeroKernel);

		var kernel = KernelUtil.GetKernel(loaded);

		cached = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, ArrayView<byte>>>();
		pool[accelerator] = cached;
		return cached;
	}

	private static Action<AcceleratorStream, Index1D, ArrayView<ulong>, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>> GetLucasLehmerBatchKernel(Accelerator accelerator)
	{
		var pool = _lucasLehmerBatchKernel ??= [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}
		
		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>>(LucasLehmerKernels.KernelBatch);

		var kernel = KernelUtil.GetKernel(loaded);

		cached = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<ulong>, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>>>();
		pool[accelerator] = cached;
		return cached;
	}

    // Configurable LL slice size to keep kernels short. Default 32.
    public int SliceSize = 32;

    public bool IsPrime(PrimeOrderCalculatorAccelerator gpu, ulong exponent, bool runOnGpu)
    {
        // Early rejections aligned with incremental/order sieves, but safe for small p:
        // - If 3 | p and p != 3, then 7 | M_p -> composite.
        // - If p ≡ 1 (mod 4) and p shares a factor with (p-1), reject fast.
        // TODO: Replace this `% 3` guard with ULongExtensions.Mod3 once GPU LL filtering reuses the benchmarked bitmask helper.
        if ((exponent % 3UL) == 0UL && exponent != 3UL)
        {
            return false;
        }

        if ((exponent & 3UL) == 1UL && exponent.SharesFactorWithExponentMinusOne(gpu))
        {
            return false;
        }

        bool result;
        if (exponent >= 128UL)
        {
            if (!TryGetNttParameters(exponent, out var nttMod, out var primitiveRoot))
            {
                throw new NotSupportedException("NTT parameters for the given exponent are not available.");
            }

            result = IsMersennePrimeNtt(exponent, nttMod, primitiveRoot, runOnGpu);
        }
        else
        {
        	// GpuPrimeWorkLimiter.Acquire();
            var acceleratorIndex = AcceleratorPool.Shared.Rent();
			var accelerator = _accelerators[acceleratorIndex];
            var stream = accelerator.CreateStream();
            var modulus = new GpuUInt128(((UInt128)1 << (int)exponent) - 1UL); // TODO: Cache these Mersenne moduli per exponent so LL GPU runs skip rebuilding them every launch.
			var buffer = accelerator.Allocate1D<GpuUInt128>(1);
			var lucasLehmerKernel = GetLucasLehmerKernel(accelerator);
            lucasLehmerKernel(stream, 1, exponent, modulus, buffer.View);
			stream.Synchronize();
			// TODO: Replace this with .CopyToCPU with pre-allocated buffer using stream. Synchronize on the stream after copy.
            result = buffer.GetAsArray1D()[0].IsZero;
            buffer.Dispose();
            stream.Dispose();
        }

		// GpuPrimeWorkLimiter.Release();
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

        var acceleratorIndex = AcceleratorPool.Shared.Rent();
		var accelerator = _accelerators[acceleratorIndex];
        var stream = accelerator.CreateStream();

        var expBuffer = accelerator.Allocate1D<ulong>(count);
		ArrayPool<ulong> ulongPool = ThreadStaticPools.UlongPool;
		ulong[] expArray = ulongPool.Rent(count);
        exponents.CopyTo(expArray);
        expBuffer.View.CopyFromCPU(stream, ref expArray[0], count);

		ArrayPool<GpuUInt128> gpuUInt128Pool = ThreadStaticPools.GpuUInt128Pool;
		GpuUInt128[] modulusArray = gpuUInt128Pool.Rent(count);
        for (int i = 0; i < count; i++)
        {
            // TODO: Replace this per-exponent shift with a small shared table (one entry per supported exponent < 128)
            // so Lucas–Lehmer batch runs reuse cached GpuUInt128 moduli instead of rebuilding them for every request.
            modulusArray[i] = new GpuUInt128(((UInt128)1 << (int)exponents[i]) - 1UL);
        }

        var modBuffer = accelerator.Allocate1D<GpuUInt128>(count);
        modBuffer.View.CopyFromCPU(stream, ref modulusArray[0], count);

		var stateBuffer = accelerator.Allocate1D<GpuUInt128>(count);
		var lucasLehmerBatchKernel = GetLucasLehmerBatchKernel(accelerator);
        lucasLehmerBatchKernel(stream, count, expBuffer.View, modBuffer.View, stateBuffer.View);
        stateBuffer.View.CopyToCPU(stream, ref residues[0], count);
        stream.Synchronize();

        stateBuffer.Dispose();
        modBuffer.Dispose();
        expBuffer.Dispose();
        gpuUInt128Pool.Return(modulusArray);
        ulongPool.Return(expArray);
        stream.Dispose();
    }

    public void IsMersennePrimeBatch(ReadOnlySpan<ulong> exponents, Span<bool> results)
    {
        int count = exponents.Length;
        if (results.Length < count)
        {
            throw new ArgumentException("Result span too small", nameof(results));
        }

		ArrayPool<GpuUInt128> gpuUInt128Pool = ThreadStaticPools.GpuUInt128Pool;
		GpuUInt128[] buffer = gpuUInt128Pool.Rent(count);
        ComputeResidues(exponents, buffer.AsSpan(0, count));
        for (int i = 0; i < count; i++)
        {
            results[i] = buffer[i].IsZero;
        }
        gpuUInt128Pool.Return(buffer);
    }

	private static readonly Accelerator[] _accelerators = AcceleratorPool.Shared.Accelerators;

    private bool IsMersennePrimeNtt(ulong exponent, GpuUInt128 nttMod, GpuUInt128 primitiveRoot, bool runOnGpu)
    {
        var acceleratorIndex = AcceleratorPool.Shared.Rent();
		var accelerator = _accelerators[acceleratorIndex];
        var stream = accelerator.CreateStream();

        var subKernel = GetSubSmallKernel(accelerator);
		var reduceKernel = GetReduceKernel(accelerator);
		var isZeroKernel = GetIsZeroKernel(accelerator);

        int limbCount = (int)((exponent + 127UL) / 128UL);
        int target = limbCount * 2;
        int transformLength = 1;
        while (transformLength < target)
        {
            transformLength <<= 1;
        }
        var stateBuffer = accelerator.Allocate1D<GpuUInt128>(transformLength);
		stateBuffer.MemSetToZero(stream);
		var addSmallKernel = GetAddSmallKernel(accelerator);
		addSmallKernel(stream, 1, 4UL, stateBuffer.View);

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
                NttGpuMath.SquareDevice(accelerator, stream, stateBuffer.View, nttMod, primitiveRoot);
                subKernel(stream, 1, 2UL, stateBuffer.View);
                reduceKernel(stream, 1, exponent, stateBuffer.View);
            }

            // Synchronize between slices to yield to scheduler and avoid TDR.
            i += (ulong)iter;
        }

        reduceKernel(stream, 1, exponent, stateBuffer.View);
		var resultBuffer = accelerator.Allocate1D<byte>(1);
        isZeroKernel(stream, 1, stateBuffer.View, resultBuffer.View);
        Span<byte> result = stackalloc byte[1];
        resultBuffer.View.CopyToCPU(stream, result);
		stream.Synchronize();
		
        bool isPrime = result[0] != 0;
        resultBuffer.Dispose();
        stateBuffer.Dispose();
        stream.Dispose();
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
