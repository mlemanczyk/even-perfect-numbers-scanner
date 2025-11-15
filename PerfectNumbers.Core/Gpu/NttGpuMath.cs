using System.Buffers;
using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;
using System.Numerics;
using System.Runtime.CompilerServices;


namespace PerfectNumbers.Core.Gpu;

public enum NttBackend
{
    Reference,
    Staged,
}

public enum ModReductionMode
{
    Auto,
    GpuUInt128,
    Mont64,
    Barrett128,
}

public static class NttGpuMath
{
    // Global backend setting for GPU transforms. Used by ForwardGpu/InverseGpu.
    // TODO(NTT-OPT): Consider passing backend explicitly to avoid global state.
    public static NttBackend GpuTransformBackend { get; set; } = NttBackend.Reference;
	// Controls modular reduction strategy inside staged NTT paths.
	public static ModReductionMode ReductionMode { get; set; } = ModReductionMode.Auto;

	[ThreadStatic]
	private static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>, GpuUInt128>>? _mulKernel; // TODO: Replace this concurrent cache with the prewarmed accelerator-indexed tables from GpuModularArithmeticBenchmarks so kernel launches avoid dictionary lookups once the kernels are baked during startup.
	
	[ThreadStatic]
	private static Dictionary<Accelerator, Action<AcceleratorStream,Index1D, ArrayView<GpuUInt128>, int, int, int, ArrayView<GpuUInt128>, GpuUInt128>>? _stageKernel; // TODO: Replace this concurrent cache with the prewarmed accelerator-indexed tables from GpuModularArithmeticBenchmarks so kernel launches avoid dictionary lookups once the kernels are baked during startup.

	[ThreadStatic]
	private static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, GpuUInt128, GpuUInt128>>? _scaleKernel; // TODO: Replace this concurrent cache with the prewarmed accelerator-indexed tables from GpuModularArithmeticBenchmarks so kernel launches avoid dictionary lookups once the kernels are baked during startup.
	
	[ThreadStatic]
	private static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, int, int, int, ArrayView<GpuUInt128>, ulong, ulong>>? _stageMontKernel; // TODO: Replace this concurrent cache with the prewarmed accelerator-indexed tables from GpuModularArithmeticBenchmarks so kernel launches avoid dictionary lookups once the kernels are baked during startup.

	[ThreadStatic]
	private static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, int, int, int, ArrayView<GpuUInt128>, ulong, ulong, ulong, ulong>>? _stageBarrett128Kernel; // TODO: Replace this concurrent cache with the prewarmed accelerator-indexed tables from GpuModularArithmeticBenchmarks so kernel launches avoid dictionary lookups once the kernels are baked during startup.
	
	[ThreadStatic]
	private static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, ulong, ulong, ulong>>? _toMont64Kernel; // TODO: Replace this concurrent cache with the prewarmed accelerator-indexed tables from GpuModularArithmeticBenchmarks so kernel launches avoid dictionary lookups once the kernels are baked during startup.
	
	[ThreadStatic]
	private static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, ulong, ulong>>? _fromMont64Kernel; // TODO: Replace this concurrent cache with the prewarmed accelerator-indexed tables from GpuModularArithmeticBenchmarks so kernel launches avoid dictionary lookups once the kernels are baked during startup.
	
	[ThreadStatic]
	private static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, GpuUInt128, ulong, ulong, ulong, ulong>>? _scaleBarrett128Kernel; // TODO: Replace this concurrent cache with the prewarmed accelerator-indexed tables from GpuModularArithmeticBenchmarks so kernel launches avoid dictionary lookups once the kernels are baked during startup.
	
	[ThreadStatic]
	private static readonly Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, int, int, int, ArrayView<GpuUInt128>, GpuUInt128>>? _squareMont64Kernel; // TODO: Replace this concurrent cache with the prewarmed accelerator-indexed tables from GpuModularArithmeticBenchmarks so kernel launches avoid dictionary lookups once the kernels are baked during startup.
	
	[ThreadStatic]
	private static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, ulong, ulong, ulong>>? _scaleMont64Kernel; // TODO: Replace this concurrent cache with the prewarmed accelerator-indexed tables from GpuModularArithmeticBenchmarks so kernel launches avoid dictionary lookups once the kernels are baked during startup.
	
	[ThreadStatic]
	private static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>, int, GpuUInt128, GpuUInt128>>? _forwardKernel; // TODO: Replace this concurrent cache with the prewarmed accelerator-indexed tables from GpuModularArithmeticBenchmarks so kernel launches avoid dictionary lookups once the kernels are baked during startup.
	
	[ThreadStatic]
	private static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>, int, GpuUInt128, GpuUInt128, GpuUInt128>>? _inverseKernel; // TODO: Replace this concurrent cache with the prewarmed accelerator-indexed tables from GpuModularArithmeticBenchmarks so kernel launches avoid dictionary lookups once the kernels are baked during startup.
	
	[ThreadStatic]
	private static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, int>>? _bitReverseKernel; // TODO: Replace this concurrent cache with the prewarmed accelerator-indexed tables from GpuModularArithmeticBenchmarks so kernel launches avoid dictionary lookups once the kernels are baked during startup.

	private static Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, int, int, int, ArrayView<GpuUInt128>, GpuUInt128> GetStageKernel(Accelerator accelerator)
	{
		var pool = _stageKernel ??= [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}
		
		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, int, int, int, ArrayView<GpuUInt128>, GpuUInt128>(NttButterflyKernels.StageKernel);
		var kernel = KernelUtil.GetKernel(loaded);
		cached = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, int, int, int, ArrayView<GpuUInt128>, GpuUInt128>>();
		pool[accelerator] = cached;
		return cached;
	}

	private sealed class SquareCacheEntry
    {
        public MemoryBuffer1D<GpuUInt128, Stride1D.Dense> Buffer { get; }
        public GpuUInt128 Root { get; }
        public GpuUInt128 RootInv { get; }
        public GpuUInt128 NInv { get; }
        public int Bits { get; }
        public int[] StageOffsets { get; }
        public MemoryBuffer1D<GpuUInt128, Stride1D.Dense> Twiddles { get; }
        public MemoryBuffer1D<GpuUInt128, Stride1D.Dense> TwiddlesInv { get; }
        // TODO(MOD-OPT): Precompute and store Montgomery/Barrett constants for modulus here
        // (e.g., Montgomery n', R2; Barrett mu) to avoid recomputation.
        public bool UseMontgomery64 { get; }
        public bool UseBarrett128 { get; }
        public ulong ModulusLow { get; }
        public ulong ModulusHigh { get; }
        public ulong MontNPrime64 { get; }
        public ulong MontRMod64 { get; }
        public ulong MontR2Mod64 { get; }
        public ulong MontNInvR64 { get; }
        public MemoryBuffer1D<GpuUInt128, Stride1D.Dense>? TwiddlesMont { get; }
        public MemoryBuffer1D<GpuUInt128, Stride1D.Dense>? TwiddlesInvMont { get; }
        public ulong BarrettMuHigh { get; }
        public ulong BarrettMuLow { get; }

        public SquareCacheEntry(Accelerator accelerator, AcceleratorStream stream, int length, GpuUInt128 modulus, GpuUInt128 primitiveRoot)
        {
            Buffer = accelerator.Allocate1D<GpuUInt128>(length);
            Root = new GpuUInt128(primitiveRoot);
            RootInv = new GpuUInt128(primitiveRoot);
            if (modulus.High == 0UL)
            {
                RootInv.ModInv(modulus.Low);
            }
            else
            {
                RootInv.ModInv(modulus);
            }

            NInv = new GpuUInt128((UInt128)length);
            if (modulus.High == 0UL)
            {
                NInv.ModInv(modulus.Low);
            }
            else
            {
                NInv.ModInv(modulus);
            }
            
            // Precompute stage twiddle factors (forward and inverse) and upload to GPU.
            Bits = (int)Math.Log2(length);
            int twiddleCount = length - 1;
            var pool = ThreadStaticPools.GpuUInt128Pool;
            var forward = pool.Rent(twiddleCount);
            var inverse = pool.Rent(twiddleCount);
            StageOffsets = new int[Bits];
            int offset = 0;
            UInt128 modValue = modulus;
            for (int stage = 1; stage <= Bits; stage++)
            {
                int len = 1 << stage;
                int half = len >> 1;
                StageOffsets[stage - 1] = offset;
                UInt128 exp = (modValue - 1UL) / (ulong)len;
                // TODO: Pull these stage roots from the precomputed tables measured fastest in
                // Pow2MontgomeryModCycleComputationBenchmarks instead of recomputing ModPow for
                // every stage.
                var wLen = new GpuUInt128(Root);
                wLen.ModPow((ulong)exp, modulus);
                var w = new GpuUInt128(0UL, 1UL);
                for (int j = 0; j < half; j++)
                {
                    forward[offset + j] = w;
                    w.MulMod(wLen, modulus);
                }
                // TODO: Reuse the same precomputed tables for inverse roots so we do not repeat the
                // expensive ModPow calls highlighted as bottlenecks in Pow2MontgomeryModCycleComputationBenchmarks.
                var wLenInv = new GpuUInt128(RootInv);
                wLenInv.ModPow((ulong)exp, modulus);
                var wi = new GpuUInt128(0UL, 1UL);
                for (int j = 0; j < half; j++)
                {
                    inverse[offset + j] = wi;
                    wi.MulMod(wLenInv, modulus);
                }
                offset += half;
            }

            Twiddles = accelerator.Allocate1D<GpuUInt128>(twiddleCount);
            Twiddles.View.CopyFromCPU(stream, ref forward[0], twiddleCount);
            TwiddlesInv = accelerator.Allocate1D<GpuUInt128>(twiddleCount);
            TwiddlesInv.View.CopyFromCPU(stream, ref inverse[0], twiddleCount);
            
            // Prepare Montgomery constants for 64-bit moduli and twiddles in Montgomery domain.
            if (modulus.High == 0UL && (ReductionMode == ModReductionMode.Auto || ReductionMode == ModReductionMode.Mont64))
            {
                UseMontgomery64 = true;
                ModulusLow = modulus.Low;
                // TODO: Preload MontNPrime64/MontR values using the MontgomeryMultiplyBenchmarks helper so
                // startup reuses the fastest constant generation routine instead of recomputing here.
                MontNPrime64 = ComputeMontgomeryNPrime64(ModulusLow);
                var mBI = new BigInteger(ModulusLow);
                MontRMod64 = (ulong)((BigInteger.One << 64) % mBI);
                MontR2Mod64 = (ulong)((BigInteger.One << 128) % mBI);
                // nInvR = (nInv * R) mod m where nInv is scalar n^{-1} mod m
                ulong nInv64 = NInv.Low % ModulusLow;
                MontNInvR64 = (ulong)((((BigInteger)nInv64) * MontRMod64) % mBI);

                var forwardMont = pool.Rent(twiddleCount);
                var inverseMont = pool.Rent(twiddleCount);
                for (int i = 0; i < twiddleCount; i++)
                {
                    ulong w = forward[i].Low % ModulusLow;
                    ulong wi = inverse[i].Low % ModulusLow;
                    forwardMont[i] = new GpuUInt128(0UL, (ulong)((((BigInteger)w) * MontRMod64) % mBI));
                    inverseMont[i] = new GpuUInt128(0UL, (ulong)((((BigInteger)wi) * MontRMod64) % mBI));
                }

                TwiddlesMont = accelerator.Allocate1D<GpuUInt128>(twiddleCount);
                TwiddlesMont.View.CopyFromCPU(stream, ref forwardMont[0], twiddleCount);
                TwiddlesInvMont = accelerator.Allocate1D<GpuUInt128>(twiddleCount);
				TwiddlesInvMont.View.CopyFromCPU(stream, ref inverseMont[0], twiddleCount);

                pool.Return(forwardMont, clearArray: true);
                pool.Return(inverseMont, clearArray: true);
            }
            else if (modulus.High != 0UL && (ReductionMode == ModReductionMode.Auto || ReductionMode == ModReductionMode.Barrett128))
            {
                // Enable Barrett for 128-bit moduli
                UseBarrett128 = true;
                ModulusHigh = modulus.High;
                ModulusLow = modulus.Low;
                // mu = floor(2^256 / n)
                var nBI = ((BigInteger)modulus.High << 64) | modulus.Low;
                var muBI = (BigInteger.One << 256) / nBI;
                BarrettMuHigh = (ulong)(muBI >> 64);
                BarrettMuLow = (ulong)muBI;
            }
            else
            {
                // Fallback to generic GpuUInt128 path in staged butterflies
                UseMontgomery64 = false;
                UseBarrett128 = false;
                ModulusHigh = modulus.High;
                ModulusLow = modulus.Low;
            }

            pool.Return(forward, clearArray: true);
            pool.Return(inverse, clearArray: true);
        }

        public void Dispose()
        {
            Buffer.Dispose();
            Twiddles.Dispose();
            TwiddlesInv.Dispose();
            TwiddlesMont?.Dispose();
            TwiddlesInvMont?.Dispose();
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong ComputeMontgomeryNPrime64(ulong modulus)
    {
        // Compute inv = modulus^{-1} mod 2^64 via Newton–Raphson, then return -inv mod 2^64.
        // Works because modulus is odd for NTT-friendly primes.
        ulong inv = 1UL;
        for (int i = 0; i < 6; i++)
        {
            // inv = inv * (2 - m * inv) mod 2^64
            ulong tHigh, tLow;
            Mul64Parts(modulus, inv, out tHigh, out tLow);
            ulong twoMinus = unchecked(2UL - tLow);
            ulong h2, low2;
            Mul64Parts(inv, twoMinus, out h2, out low2);
            inv = low2;
        }

        return 0UL - inv;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Mul64Parts(ulong a, ulong b, out ulong high, out ulong low)
    {
        // 64x64 -> 128 using 32-bit limbs
        ulong a0 = (uint)a;
        ulong a1 = a >> 32;
        ulong b0 = (uint)b;
        ulong b1 = b >> 32;

        ulong lo = a0 * b0;
        ulong mid1 = a1 * b0;
        ulong mid2 = a0 * b1;
        ulong hi = a1 * b1;

        ulong carry = (lo >> 32) + (uint)mid1 + (uint)mid2;
        low = (lo & 0xFFFFFFFFUL) | (carry << 32);
        hi += (mid1 >> 32) + (mid2 >> 32) + (carry >> 32);
        high = hi;
    }

    private readonly record struct SquareCacheKey(int Length, GpuUInt128 Modulus, GpuUInt128 PrimitiveRoot);

    private static readonly ConcurrentDictionary<SquareCacheKey, SquareCacheEntry> SquareCache = [];

    // TODO(NTT-OPT): Add a TwiddleCache for stage-wise Cooley–Tukey NTT.
    // Design:
    // - Keyed by (Length, Modulus, PrimitiveRoot) per Accelerator.
    // - Store one GPU buffer with all stage twiddles, or per-stage slices.
    // - Expose accessors to retrieve per-stage views used by butterfly kernels.

    private static SquareCacheEntry GetSquareCache(Accelerator accelerator, AcceleratorStream stream, int length, GpuUInt128 modulus, GpuUInt128 primitiveRoot)
    {
        var key = new SquareCacheKey(length, modulus, primitiveRoot);
        return SquareCache.GetOrAdd(key, _ => new SquareCacheEntry(accelerator, stream, length, modulus, primitiveRoot));
    }

    private static readonly ulong[] SmallPrimes = GenerateSmallPrimes(1000);

    public static bool TryGenerateParameters(int length, out GpuUInt128 modulus, out GpuUInt128 primitiveRoot)
    {
        GpuUInt128 foundMod = default;
        GpuUInt128 foundRoot = default;
        using CancellationTokenSource cts = new();
        Parallel.For(1, int.MaxValue, new ParallelOptions { CancellationToken = cts.Token }, (i, state) =>
        {
            ulong candidate;
            try
            {
                candidate = checked((ulong)length * (ulong)i + 1UL);
            }
            catch (OverflowException)
            {
                state.Stop();
                cts.Cancel();
                return;
            }

            bool divisible = false;
            foreach (ulong p in SmallPrimes)
            {
                if (candidate % p == 0UL)
                {
                    divisible = true;
                    break;
                }
            }

            if (divisible || !IsPrime(candidate))
            {
                return;
            }

            ulong root = FindPrimitiveRoot(candidate, (ulong)length);
            foundMod = new GpuUInt128(0UL, candidate);
            foundRoot = new GpuUInt128(0UL, root);
            state.Stop();
            cts.Cancel();
        });

        if (foundMod.High != 0UL || foundMod.Low != 0UL)
        {
            modulus = foundMod;
            primitiveRoot = foundRoot;
            return true;
        }

        if (TryFindLargeModulus(length, SmallPrimes, out modulus, out primitiveRoot))
        {
            return true;
        }

        modulus = new GpuUInt128(0UL, 18446744069414584321UL);
        primitiveRoot = new GpuUInt128(0UL, 7UL);
        // Moduli larger than 128 bits are not supported by the current implementation.
        return false;
    }

    public static void ClearCaches(Accelerator? accelerator = null)
    {
        if (accelerator is null)
        {
            foreach (var entry in SquareCache.Values)
            {
				entry.Dispose();
            }

            SquareCache.Clear();
            // TODO(NTT-OPT): Clear TwiddleCache once introduced.
            return;
        }

		var squareCache = SquareCache; 
		foreach (var entry in squareCache.Values)
		{
			entry.Dispose();
		}

        // TODO(NTT-OPT): Remove per-accelerator twiddle buffers once added.
    }

	private static Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, int> GetBitReverseKernel(Accelerator accelerator)
	{
		var pool = _bitReverseKernel ??= [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}

		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, int>(NttTransformKernels.BitReverseKernel);
		
		var kernel = KernelUtil.GetKernel(loaded);

		cached = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, int>>();
		pool[accelerator] = cached;
		return cached;
	}

	private static Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>, int, GpuUInt128, GpuUInt128> GetForwardKernel(Accelerator accelerator)
	{
		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>, int, GpuUInt128, GpuUInt128>(NttTransformKernels.ForwardKernel);
		var kernel = KernelUtil.GetKernel(loaded);
		return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>, int, GpuUInt128, GpuUInt128>>();
	}

	private static Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, ulong, ulong> GetFromMont64Kernel(Accelerator accelerator)
	{
		var pool = _fromMont64Kernel ??= [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}

		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, ulong, ulong>(NttMontgomeryKernels.FromMont64Kernel);

		var kernel = KernelUtil.GetKernel(loaded);

		cached = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, ulong, ulong>>();
		pool[accelerator] = cached;
		return cached;
	}

	private static Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>, int, GpuUInt128, GpuUInt128, GpuUInt128> GetInverseKernel(Accelerator accelerator)
	{
		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>, int, GpuUInt128, GpuUInt128, GpuUInt128>(NttTransformKernels.InverseKernel);
		var kernel = KernelUtil.GetKernel(loaded);
		return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>, int, GpuUInt128, GpuUInt128, GpuUInt128>>();
	}

	private static Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>, GpuUInt128> GetMulKernel(Accelerator accelerator)
	{
		var pool = _mulKernel ??= [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}

		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>, GpuUInt128>(NttPointwiseKernels.MulKernel);
		
		var kernel = KernelUtil.GetKernel(loaded);

		cached = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>, GpuUInt128>>();
		pool[accelerator] = cached;
		return cached;
	}

	private static Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, GpuUInt128, GpuUInt128> GetScaleKernel(Accelerator accelerator)
	{
		var pool = _scaleKernel ??= [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}

		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, GpuUInt128, GpuUInt128>(NttPointwiseKernels.ScaleKernel);
		
		var kernel = KernelUtil.GetKernel(loaded);

		cached = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, GpuUInt128, GpuUInt128>>();
		pool[accelerator] = cached;
		return cached;
	}

	private static Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, GpuUInt128, ulong, ulong, ulong, ulong> GetScaleBarrett128Kernel(Accelerator accelerator)
	{
		var pool = _scaleBarrett128Kernel ??= [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}

		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, GpuUInt128, ulong, ulong, ulong, ulong>(NttBarrettKernels.ScaleBarrett128Kernel);
		
		var kernel = KernelUtil.GetKernel(loaded);

		cached = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, GpuUInt128, ulong, ulong, ulong, ulong>>();
		pool[accelerator] = cached;
		return cached;
	}

	private static Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, ulong, ulong, ulong> GetScaleMont64Kernel(Accelerator accelerator)
	{
		var pool = _scaleMont64Kernel ??= [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}

		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, ulong, ulong, ulong>(NttMontgomeryKernels.ScaleMont64Kernel);
		
		var kernel = KernelUtil.GetKernel(loaded);

		cached = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, ulong, ulong, ulong>>();
		pool[accelerator] = cached;
		return cached;
	}

	private static Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, int, int, int, ArrayView<GpuUInt128>, ulong, ulong, ulong, ulong> GetStageBarrett128Kernel(Accelerator accelerator)
	{
		var pool = _stageBarrett128Kernel ??= [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}

		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, int, int, int, ArrayView<GpuUInt128>, ulong, ulong, ulong, ulong>(NttBarrettKernels.StageBarrett128Kernel);

		var kernel = KernelUtil.GetKernel(loaded);
		
		cached = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, int, int, int, ArrayView<GpuUInt128>, ulong, ulong, ulong, ulong>>();
		pool[accelerator] = cached;
		return cached;
	}

	private static Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, int, int, int, ArrayView<GpuUInt128>, ulong, ulong> GetStageMontKernel(Accelerator accelerator)
	{
		var pool = _stageMontKernel ??= [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}

		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, int, int, int, ArrayView<GpuUInt128>, ulong, ulong>(NttMontgomeryKernels.StageMontKernel);

		var kernel = KernelUtil.GetKernel(loaded);

		cached = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, int, int, int, ArrayView<GpuUInt128>, ulong, ulong>>();
		pool[accelerator] = cached;
		return cached;
	}

	private static Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, ulong, ulong, ulong> GetToMont64Kernel(Accelerator accelerator)
	{
		var pool = _toMont64Kernel ??= [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}

		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, ulong, ulong, ulong>(NttMontgomeryKernels.ToMont64Kernel);
		
		var kernel = KernelUtil.GetKernel(loaded);

		cached = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<GpuUInt128>, ulong, ulong, ulong>>();
		pool[accelerator] = cached;
		return cached;
	}

	// TODO(NTT-OPT): Mirror the forward stage-wise design for the inverse
	// transform. Precompute inverse twiddles and a normalization factor
	// (n^-1 mod m). Launch one short kernel per stage to avoid TDR.

	public static void BitReverse(Span<GpuUInt128> values)
    {
        int n = values.Length;
        int bits = (int)Math.Log2(n);
        for (int i = 0; i < n; i++)
        {
            int j = NttTransformKernels.ReverseBits(i, bits);
            if (i < j)
            {
                (values[j], values[i]) = (values[i], values[j]);
            }
        }
    }

    public static void BitReverseGpu(Span<GpuUInt128> values)
    {
        int n = values.Length;
        int bits = (int)Math.Log2(n);
        var acceleratorIndex = AcceleratorPool.Shared.Rent();
		var accelerator = _accelerators[acceleratorIndex];
        var pool = ThreadStaticPools.GpuUInt128Pool;
        var array = pool.Rent(n);
        values.CopyTo(array);
        var buffer = accelerator.Allocate1D<GpuUInt128>(n);
        var stream = accelerator.CreateStream();
		buffer.View.CopyFromCPU(stream, ref array[0], n);
		var bitReverseKernel = GetBitReverseKernel(accelerator);
        bitReverseKernel(stream, n, buffer.View, bits);
        buffer.View.CopyToCPU(stream, ref array[0], n);
        stream.Synchronize();
        array.AsSpan(0, n).CopyTo(values);
        pool.Return(array, clearArray: true);
        buffer.Dispose();
        stream.Dispose();

    }

    public static void Forward(Span<GpuUInt128> values, GpuUInt128 modulus, GpuUInt128 primitiveRoot)
    {
        int n = values.Length;
        BitReverse(values);
        UInt128 modValue = modulus;
        for (int len = 2; len <= n; len <<= 1)
        {
            UInt128 expBase = (modValue - 1UL) / (ulong)len;
            var wLen = new GpuUInt128(primitiveRoot);
            wLen.ModPow((ulong)expBase, modulus);
            for (int i = 0; i < n; i += len)
            {
                var w = new GpuUInt128(0UL, 1UL);
                int half = len >> 1;
                for (int j = 0; j < half; j++)
                {
                    var u = values[i + j];
                    var v = values[i + j + half];
                    v.MulMod(w, modulus);
                    var sum = new GpuUInt128(u);
                    sum.AddMod(v, modulus);
                    u.SubMod(v, modulus);
                    values[i + j] = sum;
                    values[i + j + half] = u;
                    w.MulMod(wLen, modulus);
                }
            }
        }
    }

    public static void ForwardGpu(Span<GpuUInt128> values, GpuUInt128 modulus, GpuUInt128 primitiveRoot)
    {
        if (GpuTransformBackend == NttBackend.Reference)
        {
            ForwardGpuReference(values, modulus, primitiveRoot);
            return;
        }

        ForwardGpuStaged(values, modulus, primitiveRoot);
    }

    private static void ForwardGpuReference(Span<GpuUInt128> values, GpuUInt128 modulus, GpuUInt128 primitiveRoot)
    {
        int n = (int)values.Length;
        var acceleratorIndex = AcceleratorPool.Shared.Rent();
		var accelerator = _accelerators[acceleratorIndex];
        var stream = accelerator.CreateStream();
        var pool = ThreadStaticPools.GpuUInt128Pool;
        var inputArray = pool.Rent(n);
        var outputArray = pool.Rent(n);
        values.CopyTo(inputArray);
        var inputBuffer = accelerator.Allocate1D<GpuUInt128>(n);
        var outputBuffer = accelerator.Allocate1D<GpuUInt128>(n);
        inputBuffer.View.CopyFromCPU(stream, ref inputArray[0], n);
        UInt128 modValue = modulus;
        UInt128 expBase = (modValue - 1UL) / (ulong)n;
        var root = new GpuUInt128(primitiveRoot);
		root.ModPow((ulong)expBase, modulus);
		var forwardKernel = GetForwardKernel(accelerator);
        forwardKernel(stream, n, inputBuffer.View, outputBuffer.View, n, modulus, root);
        outputBuffer.View.CopyToCPU(stream, ref outputArray[0], n);
        stream.Synchronize();
        outputArray.AsSpan(0, n).CopyTo(values);
        pool.Return(inputArray, clearArray: true);
        pool.Return(outputArray, clearArray: true);
        outputBuffer.Dispose();
        inputBuffer.Dispose();
        stream.Dispose();
    }

    public static void Inverse(Span<GpuUInt128> values, GpuUInt128 modulus, GpuUInt128 primitiveRoot)
    {
        int n = values.Length;
        BitReverse(values);
        UInt128 modValue = modulus;
        var root = new GpuUInt128(primitiveRoot);
        if (modulus.High == 0UL)
        {
            root.ModInv(modulus.Low);
        }
        else
        {
            root.ModInv(modulus);
        }
		
        for (int len = 2; len <= n; len <<= 1)
		{
                        UInt128 expBase = (modValue - 1UL) / (ulong)len;
                        var wLen = new GpuUInt128(root);
                        wLen.ModPow((ulong)expBase, modulus);
			for (int i = 0; i < n; i += len)
			{
				var w = new GpuUInt128(0UL, 1UL);
				int half = len >> 1;
				for (int j = 0; j < half; j++)
				{
                                        var u = values[i + j];
                                        var v = values[i + j + half];
                                        v.MulMod(w, modulus);
                                        var sum = new GpuUInt128(u);
                                        sum.AddMod(v, modulus);
                                        u.SubMod(v, modulus);
                                        values[i + j] = sum;
                                        values[i + j + half] = u;
                                        w.MulMod(wLen, modulus);
				}
			}
		}

        var nInv = new GpuUInt128((UInt128)n);
        if (modulus.High == 0UL)
        {
            nInv.ModInv(modulus.Low);
        }
        else
        {
            nInv.ModInv(modulus);
        }

        for (int i = 0; i < n; i++)
        {
            var val = values[i];
            val.MulMod(nInv, modulus);
            values[i] = val;
        }
    }

    public static void InverseGpu(Span<GpuUInt128> values, GpuUInt128 modulus, GpuUInt128 primitiveRoot)
    {
        if (GpuTransformBackend == NttBackend.Reference)
        {
            InverseGpuReference(values, modulus, primitiveRoot);
            return;
        }

        InverseGpuStaged(values, modulus, primitiveRoot);
    }

    private static void InverseGpuReference(Span<GpuUInt128> values, GpuUInt128 modulus, GpuUInt128 primitiveRoot)
    {
        int n = values.Length;
        var acceleratorIndex = AcceleratorPool.Shared.Rent();
		var accelerator = _accelerators[acceleratorIndex];
        var stream = accelerator.CreateStream();
        var pool = ThreadStaticPools.GpuUInt128Pool;
        var inputArray = pool.Rent(n);
        var outputArray = pool.Rent(n);
        values.CopyTo(inputArray);
        var inputBuffer = accelerator.Allocate1D<GpuUInt128>(n);
        var outputBuffer = accelerator.Allocate1D<GpuUInt128>(n);
        inputBuffer.View.CopyFromCPU(stream, ref inputArray[0], n);
        UInt128 modValue = modulus;
        UInt128 expBase = (modValue - 1UL) / (ulong)n;
        var root = new GpuUInt128(primitiveRoot);
        root.ModPow((ulong)expBase, modulus);
        var rootInv = new GpuUInt128(root);
        if (modulus.High == 0UL)
        {
            rootInv.ModInv(modulus.Low);
        }
        else
        {
            rootInv.ModInv(modulus);
        }

        var nInv = new GpuUInt128((UInt128)n);
		if (modulus.High == 0UL)
		{
			nInv.ModInv(modulus.Low);
		}
		else
		{
			nInv.ModInv(modulus);
		}

		var inverseKernel = GetInverseKernel(accelerator);
        inverseKernel(stream, n, inputBuffer.View, outputBuffer.View, n, modulus, rootInv, nInv);
        outputBuffer.View.CopyToCPU(stream, ref outputArray[0], n);
        stream.Synchronize();
        outputArray.AsSpan(0, n).CopyTo(values);
        pool.Return(inputArray, clearArray: true);
        pool.Return(outputArray, clearArray: true);
        outputBuffer.Dispose();
        inputBuffer.Dispose();
        stream.Dispose();
    }

    private static void ForwardGpuStaged(Span<GpuUInt128> values, GpuUInt128 modulus, GpuUInt128 primitiveRoot)
    {
        int n = values.Length;
        var acceleratorIndex = AcceleratorPool.Shared.Rent();
		var accelerator = _accelerators[acceleratorIndex];
        var stream = accelerator.CreateStream();
        var buffer = accelerator.Allocate1D<GpuUInt128>(n);
        buffer.View.CopyFromCPU(stream, ref values[0], n);
        var cache = GetSquareCache(accelerator, stream, n, modulus, primitiveRoot);
        ForwardDevice(accelerator, stream, buffer.View, n, modulus, cache);
		buffer.View.CopyToCPU(stream, ref values[0], n);
		stream.Synchronize();
        stream.Dispose();
        buffer.Dispose();
    }

    private static void InverseGpuStaged(Span<GpuUInt128> values, GpuUInt128 modulus, GpuUInt128 primitiveRoot)
    {
        int n = values.Length;
        var acceleratorIndex = AcceleratorPool.Shared.Rent();
		var accelerator = _accelerators[acceleratorIndex];
        var stream = accelerator.CreateStream();
        var buffer = accelerator.Allocate1D<GpuUInt128>(n);
        buffer.View.CopyFromCPU(stream, ref values[0], n);
        var cache = GetSquareCache(accelerator, stream, n, modulus, primitiveRoot);
        InverseDevice(accelerator, stream, buffer.View, n, modulus, cache);
        buffer.View.CopyToCPU(stream, ref values[0], n);
		stream.Synchronize();
        stream.Dispose();
        buffer.Dispose();
    }

    private static void ForwardDevice(Accelerator accelerator, AcceleratorStream stream, ArrayView<GpuUInt128> data, int length, GpuUInt128 modulus, SquareCacheEntry cache)
	{
		var bitReverseKernel = GetBitReverseKernel(accelerator);
        bitReverseKernel(stream, length, data, cache.Bits);

        if (cache.UseMontgomery64)
		{
			var toMont64Kernel = GetToMont64Kernel(accelerator);
            toMont64Kernel(stream, length, data, cache.ModulusLow, cache.MontNPrime64, cache.MontR2Mod64);
            var stageKernel = GetStageMontKernel(accelerator);
			int butterflies = length >> 1;
			for (int s = 0; s < cache.Bits; s++)
			{
				int len = 1 << (s + 1);
				int half = len >> 1;
				stageKernel(stream, butterflies, data, len, half, cache.StageOffsets[s], cache.TwiddlesMont!.View, cache.ModulusLow, cache.MontNPrime64);
			}
			var fromMont64Kernel = GetFromMont64Kernel(accelerator);
            fromMont64Kernel(stream, length, data, cache.ModulusLow, cache.MontNPrime64);
        }
        else if (cache.UseBarrett128)
        {
            var stageKernel = GetStageBarrett128Kernel(accelerator);
            int butterflies = length >> 1;
            for (int s = 0; s < cache.Bits; s++)
            {
                int len = 1 << (s + 1);
                int half = len >> 1;
                stageKernel(stream, butterflies, data, len, half, cache.StageOffsets[s], cache.Twiddles.View, cache.ModulusHigh, cache.ModulusLow, cache.BarrettMuHigh, cache.BarrettMuLow);
            }
        }
        else
        {
            var stageKernel = GetStageKernel(accelerator);
            int butterflies = length >> 1;
            for (int s = 0; s < cache.Bits; s++)
            {
                int len = 1 << (s + 1);
                int half = len >> 1;
                stageKernel(stream, butterflies, data, len, half, cache.StageOffsets[s], cache.Twiddles.View, modulus);
            }
        }

        stream.Synchronize();
    }

    private static void InverseDevice(Accelerator accelerator, AcceleratorStream stream, ArrayView<GpuUInt128> data, int length, GpuUInt128 modulus, SquareCacheEntry cache)
	{
		var bitReverseKernel = GetBitReverseKernel(accelerator);
        bitReverseKernel(stream, length, data, cache.Bits);

        if (cache.UseMontgomery64)
		{
			var toMont64Kernel = GetToMont64Kernel(accelerator);
            toMont64Kernel(stream, length, data, cache.ModulusLow, cache.MontNPrime64, cache.MontR2Mod64);
            var stageKernel = GetStageMontKernel(accelerator);
            int butterflies = length >> 1;
			for (int s = 0; s < cache.Bits; s++)
			{
				int len = 1 << (s + 1);
				int half = len >> 1;
				stageKernel(stream, butterflies, data, len, half, cache.StageOffsets[s], cache.TwiddlesInvMont!.View, cache.ModulusLow, cache.MontNPrime64);
			}

			var scaleMont64Kernel = GetScaleMont64Kernel(accelerator);
			scaleMont64Kernel(stream, length, data, cache.ModulusLow, cache.MontNPrime64, cache.MontNInvR64);

			var fromMont64Kernel = GetFromMont64Kernel(accelerator);
            fromMont64Kernel(stream, length, data, cache.ModulusLow, cache.MontNPrime64);
        }
        else if (cache.UseBarrett128)
        {
            var stageKernel = GetStageBarrett128Kernel(accelerator);
            int butterflies = length >> 1;
			for (int s = 0; s < cache.Bits; s++)
			{
				int len = 1 << (s + 1);
				int half = len >> 1;
				stageKernel(stream, butterflies, data, len, half, cache.StageOffsets[s], cache.TwiddlesInv.View, cache.ModulusHigh, cache.ModulusLow, cache.BarrettMuHigh, cache.BarrettMuLow);
			}

			var scaleBarrett128Kernel = GetScaleBarrett128Kernel(accelerator);
            scaleBarrett128Kernel(stream, length, data, cache.NInv, cache.ModulusHigh, cache.ModulusLow, cache.BarrettMuHigh, cache.BarrettMuLow);
        }
        else
        {
            var stageKernel = GetStageKernel(accelerator);
            int butterflies = length >> 1;
			for (int s = 0; s < cache.Bits; s++)
			{
				int len = 1 << (s + 1);
				int half = len >> 1;
				stageKernel(stream, butterflies, data, len, half, cache.StageOffsets[s], cache.TwiddlesInv.View, modulus);
			}

			var scaleKernel = GetScaleKernel(accelerator);
            scaleKernel(stream, length, data, cache.NInv, modulus);
        }

        stream.Synchronize();
    }

    public static void Convolve(Span<GpuUInt128> left, Span<GpuUInt128> right, GpuUInt128 modulus, GpuUInt128 primitiveRoot)
    {
        int n = 1;
        int resultLength = left.Length + right.Length;
        while (n < resultLength)
        {
            n <<= 1;
        }

        var pool = ThreadStaticPools.GpuUInt128Pool;
        var aBuffer = pool.Rent(n);
        var bBuffer = pool.Rent(n);
        var a = aBuffer.AsSpan(0, n);
        var b = bBuffer.AsSpan(0, n);
        left.CopyTo(a);
        right.CopyTo(b);
        for (int i = left.Length; i < n; i++)
        {
            a[i] = new GpuUInt128(0UL, 0UL);
        }

        for (int i = right.Length; i < n; i++)
        {
            b[i] = new GpuUInt128(0UL, 0UL);
        }

        Forward(a, modulus, primitiveRoot);
        Forward(b, modulus, primitiveRoot);
        PointwiseMultiply(a, b, modulus);

        Inverse(a, modulus, primitiveRoot);
        int copyLength = Math.Min(left.Length, resultLength);
        a.Slice(0, copyLength).CopyTo(left);
        pool.Return(aBuffer, clearArray: true);
        pool.Return(bBuffer, clearArray: true);
    }

    public static void ConvolveGpu(Span<GpuUInt128> left, Span<GpuUInt128> right, GpuUInt128 modulus, GpuUInt128 primitiveRoot)
    {
        int n = 1;
        int resultLength = left.Length + right.Length;
        while (n < resultLength)
        {
            n <<= 1;
        }

        var pool = ThreadStaticPools.GpuUInt128Pool;
        var aBuffer = pool.Rent(n);
        var bBuffer = pool.Rent(n);
        var a = aBuffer.AsSpan(0, n);
        var b = bBuffer.AsSpan(0, n);
        left.CopyTo(a);
        right.CopyTo(b);
        for (int i = left.Length; i < n; i++)
        {
            a[i] = new GpuUInt128(0UL, 0UL);
        }

        for (int i = right.Length; i < n; i++)
        {
            b[i] = new GpuUInt128(0UL, 0UL);
        }

        ForwardGpu(a, modulus, primitiveRoot);
        ForwardGpu(b, modulus, primitiveRoot);
        PointwiseMultiply(a, b, modulus);
        InverseGpu(a, modulus, primitiveRoot);

        int copyLength = Math.Min(left.Length, resultLength);
        a.Slice(0, copyLength).CopyTo(left);
        pool.Return(aBuffer, clearArray: true);
        pool.Return(bBuffer, clearArray: true);
    }

    public static void Square(Span<GpuUInt128> values, GpuUInt128 modulus, GpuUInt128 primitiveRoot) =>
        Convolve(values, values, modulus, primitiveRoot);

    public static void SquareGpu(Span<GpuUInt128> values, GpuUInt128 modulus, GpuUInt128 primitiveRoot) =>
        ConvolveGpu(values, values, modulus, primitiveRoot);

    public static void SquareDevice(Accelerator accelerator, AcceleratorStream stream, ArrayView<GpuUInt128> values, GpuUInt128 modulus, GpuUInt128 primitiveRoot)
    {
        var cache = GetSquareCache(accelerator, stream, (int)values.Length, modulus, primitiveRoot);
		ForwardDevice(accelerator, stream, values, (int)values.Length, modulus, cache);
		var mulKernel = GetMulKernel(accelerator);
        mulKernel(stream, (int)values.Length, values, values, modulus);
        InverseDevice(accelerator, stream, values, (int)values.Length, modulus, cache);
    }

	private static readonly Accelerator[] _accelerators = AcceleratorPool.Shared.Accelerators;

    public static void PointwiseMultiply(Span<GpuUInt128> left, Span<GpuUInt128> right, GpuUInt128 modulus)
    {
        var acceleratorIndex = AcceleratorPool.Shared.Rent();
		var accelerator = _accelerators[acceleratorIndex];
        var stream = accelerator.CreateStream();
        var pool = ThreadStaticPools.GpuUInt128Pool;
        var leftArray = pool.Rent(left.Length);
        var rightArray = pool.Rent(right.Length);
        left.CopyTo(leftArray);
        right.CopyTo(rightArray);
        var leftBuffer = accelerator.Allocate1D<GpuUInt128>(left.Length);
        var rightBuffer = accelerator.Allocate1D<GpuUInt128>(right.Length);
        leftBuffer.View.CopyFromCPU(stream, ref leftArray[0], left.Length);
		rightBuffer.View.CopyFromCPU(stream, ref rightArray[0], right.Length);
		var mulKernel = GetMulKernel(accelerator);
        mulKernel(stream, left.Length, leftBuffer.View, rightBuffer.View, modulus);
        leftBuffer.View.CopyToCPU(stream, ref leftArray[0], left.Length);
        rightBuffer.View.CopyToCPU(stream, ref rightArray[0], right.Length);
        stream.Synchronize();
        leftArray.AsSpan(0, left.Length).CopyTo(left);
        rightArray.AsSpan(0, right.Length).CopyTo(right);
        pool.Return(leftArray, clearArray: true);
        pool.Return(rightArray, clearArray: true);
        rightBuffer.Dispose();
        leftBuffer.Dispose();
        stream.Dispose();
    }

    private static ulong[] GenerateSmallPrimes(int limit)
    {
        var isComposite = new bool[limit + 1];
        var primes = new List<ulong>();
        for (int p = 2; p <= limit; p++)
        {
            if (isComposite[p])
            {
                continue;
            }

            primes.Add((ulong)p);
            long square = (long)p * p;
            for (long multiple = square; multiple <= limit; multiple += p)
            {
                isComposite[(int)multiple] = true;
            }
        }

        var result = new ulong[primes.Count];
        primes.CopyTo(result, 0);
        return result;
    }

    private static bool IsPrime(ulong n)
    {
        if (n < 2UL)
        {
            return false;
        }

        ulong[] bases = [2UL, 3UL, 5UL, 7UL, 11UL, 13UL];
        ulong d = n - 1UL;
        int s = 0;
        while ((d & 1UL) == 0UL)
        {
            d >>= 1;
            s++;
        }

        foreach (ulong a in bases)
        {
            if (a >= n)
            {
                continue;
            }

            ulong x = ModPow(a, d, n);
            if (x == 1UL || x == n - 1UL)
            {
                continue;
            }

            bool cont = false;
            for (int r = 1; r < s; r++)
            {
                x = MulMod(x, x, n);
                if (x == n - 1UL)
                {
                    cont = true;
                    break;
                }
            }

            if (!cont)
            {
                return false;
            }
        }

        return true;
    }

    private static ulong FindPrimitiveRoot(ulong modulus, ulong order)
    {
        ulong phi = modulus - 1UL;
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

    private static ulong MulMod(ulong a, ulong b, ulong modulus)
    {
        return (ulong)(((UInt128)a * b) % modulus);
    }

    private static bool TryFindLargeModulus(int length, ulong[] smallPrimes, out GpuUInt128 modulus, out GpuUInt128 primitiveRoot)
    {
        BigInteger bigLength = new BigInteger(length);
        BigInteger k = (BigInteger)(ulong.MaxValue / (ulong)length) + 1;
        while (true)
        {
            BigInteger candidate = bigLength * k + BigInteger.One;
            if (candidate.GetBitLength() > 128)
            {
                break;
            }

            bool divisible = false;
            foreach (ulong p in smallPrimes)
            {
                if (candidate % p == 0)
                {
                    divisible = true;
                    break;
                }
            }

            if (!divisible && IsPrime(candidate))
            {
                BigInteger root = FindPrimitiveRoot(candidate, length);
                modulus = ToGpuUInt128(candidate);
                primitiveRoot = ToGpuUInt128(root);
                return true;
            }

            k++;
        }

        modulus = new GpuUInt128(0UL, 0UL);
        primitiveRoot = new GpuUInt128(0UL, 0UL);
        return false;
    }

    private static bool IsPrime(BigInteger n)
    {
        if (n < 2)
        {
            return false;
        }

        ulong[] bases = [2UL, 3UL, 5UL, 7UL, 11UL, 13UL, 17UL];
        BigInteger d = n - 1;
        int s = 0;
        while ((d & 1) == 0)
        {
            d >>= 1;
            s++;
        }

        foreach (ulong a in bases)
        {
            if (a >= n)
            {
                continue;
            }

            BigInteger x = BigInteger.ModPow(a, d, n);
            if (x == 1 || x == n - 1)
            {
                continue;
            }

            bool cont = false;
            for (int r = 1; r < s; r++)
            {
                x = (x * x) % n;
                if (x == n - 1)
                {
                    cont = true;
                    break;
                }
            }

            if (!cont)
            {
                return false;
            }
        }

        return true;
    }

    private static BigInteger FindPrimitiveRoot(BigInteger modulus, int order)
    {
        BigInteger phi = modulus - 1;
        var factors = FactorizeBig(phi);
        for (BigInteger g = 2; g < modulus; g++)
        {
            bool ok = true;
            foreach (BigInteger f in factors)
            {
                if (BigInteger.ModPow(g, phi / f, modulus) == 1)
                {
                    ok = false;
                    break;
                }
            }

            if (ok)
            {
                return BigInteger.ModPow(g, phi / order, modulus);
            }
        }

        throw new InvalidOperationException("Primitive root not found.");
    }

    private static List<BigInteger> FactorizeBig(BigInteger n)
    {
        var factors = new List<BigInteger>();
        for (BigInteger p = 2; p * p <= n; p += p == 2 ? 1 : 2)
        {
            if (n % p == 0)
            {
                factors.Add(p);
                while (n % p == 0)
                {
                    n /= p;
                }
            }
        }

        if (n > 1)
        {
            factors.Add(n);
        }

        return factors;
    }

    private static GpuUInt128 ToGpuUInt128(BigInteger value)
    {
        Span<byte> bytes = stackalloc byte[16];
        value.TryWriteBytes(bytes, out int written, isUnsigned: true, isBigEndian: true);
        ulong high = 0UL;
        ulong low = 0UL;
        int start = 16 - written;
        for (int i = 0; i < 8; i++)
        {
            high = (high << 8) | bytes[start + i];
        }

        for (int i = 8; i < 16; i++)
        {
            low = (low << 8) | bytes[start + i];
        }

        return new GpuUInt128(high, low);
    }
}

