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
    private static readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>, GpuUInt128>> MulKernelCache = new();
    private static readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<GpuUInt128>, int, int, int, ArrayView<GpuUInt128>, GpuUInt128>> StageKernelCache = new();
    private static readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<GpuUInt128>, GpuUInt128, GpuUInt128>> ScaleKernelCache = new();
    private static readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<GpuUInt128>, int, int, int, ArrayView<GpuUInt128>, ulong, ulong>> StageMontKernelCache = new();
    private static readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<GpuUInt128>, int, int, int, ArrayView<GpuUInt128>, ulong, ulong, ulong, ulong>> StageBarrett128KernelCache = new();
    private static readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<GpuUInt128>, GpuUInt128, ulong, ulong, ulong, ulong>> ScaleBarrett128KernelCache = new();
    private static readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<GpuUInt128>, ulong, ulong, ulong, ulong>> SquareBarrett128KernelCache = new();
    private static readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<GpuUInt128>, ulong, ulong, ulong>> ToMont64KernelCache = new();
    private static readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<GpuUInt128>, ulong, ulong>> FromMont64KernelCache = new();
    private static readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<GpuUInt128>, ulong, ulong>> SquareMont64KernelCache = new();
    private static readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<GpuUInt128>, ulong, ulong, ulong>> ScaleMont64KernelCache = new();

    private static readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>, int, GpuUInt128, GpuUInt128>> ForwardKernelCache = new();

    private static readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>, int, GpuUInt128, GpuUInt128, GpuUInt128>> InverseKernelCache = new();

    private static readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<GpuUInt128>, int>> BitReverseKernelCache = new();

    private sealed class SquareCacheEntry : IDisposable
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

        public SquareCacheEntry(Accelerator accelerator, int length, GpuUInt128 modulus, GpuUInt128 primitiveRoot)
        {
            Buffer = accelerator.Allocate1D<GpuUInt128>(length);
            Root = new GpuUInt128(primitiveRoot);
            RootInv = new GpuUInt128(primitiveRoot);
            RootInv = modulus.High == 0UL
                ? RootInv.ModInv(modulus.Low)
                : RootInv.ModInv(modulus);

            NInv = new GpuUInt128((UInt128)length);
            NInv = modulus.High == 0UL
                ? NInv.ModInv(modulus.Low)
                : NInv.ModInv(modulus);
            
            // Precompute stage twiddle factors (forward and inverse) and upload to GPU.
            Bits = (int)Math.Log2(length);
            int twiddleCount = length - 1;
            var pool = ArrayPool<GpuUInt128>.Shared;
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
                var wLen = new GpuUInt128(Root);
                wLen = wLen.ModPow((ulong)exp, modulus);
                var w = new GpuUInt128(0UL, 1UL);
                for (int j = 0; j < half; j++)
                {
                    forward[offset + j] = w;
                    w = w.MulMod(wLen, modulus);
                }
                var wLenInv = new GpuUInt128(RootInv);
                wLenInv = wLenInv.ModPow((ulong)exp, modulus);
                var wi = new GpuUInt128(0UL, 1UL);
                for (int j = 0; j < half; j++)
                {
                    inverse[offset + j] = wi;
                    wi = wi.MulMod(wLenInv, modulus);
                }
                offset += half;
            }

            Twiddles = accelerator.Allocate1D<GpuUInt128>(twiddleCount);
            Twiddles.View.CopyFromCPU(ref forward[0], twiddleCount);
            TwiddlesInv = accelerator.Allocate1D<GpuUInt128>(twiddleCount);
            TwiddlesInv.View.CopyFromCPU(ref inverse[0], twiddleCount);
            
            // Prepare Montgomery constants for 64-bit moduli and twiddles in Montgomery domain.
            if (modulus.High == 0UL && (ReductionMode == ModReductionMode.Auto || ReductionMode == ModReductionMode.Mont64))
            {
                UseMontgomery64 = true;
                ModulusLow = modulus.Low;
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
                TwiddlesMont.View.CopyFromCPU(ref forwardMont[0], twiddleCount);
                TwiddlesInvMont = accelerator.Allocate1D<GpuUInt128>(twiddleCount);
                TwiddlesInvMont.View.CopyFromCPU(ref inverseMont[0], twiddleCount);

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

    private readonly record struct SquareCacheKey(int Length, GpuUInt128 Modulus, GpuUInt128 PrimitiveRoot);

    private static readonly ConcurrentDictionary<Accelerator, ConcurrentDictionary<SquareCacheKey, SquareCacheEntry>> SquareCache = new();

    // TODO(NTT-OPT): Add a TwiddleCache for stage-wise Cooley–Tukey NTT.
    // Design:
    // - Keyed by (Length, Modulus, PrimitiveRoot) per Accelerator.
    // - Store one GPU buffer with all stage twiddles, or per-stage slices.
    // - Expose accessors to retrieve per-stage views used by butterfly kernels.

    private static SquareCacheEntry GetSquareCache(Accelerator accelerator, int length, GpuUInt128 modulus, GpuUInt128 primitiveRoot)
    {
        var perAccel = SquareCache.GetOrAdd(accelerator, _ => new ConcurrentDictionary<SquareCacheKey, SquareCacheEntry>());
        var key = new SquareCacheKey(length, modulus, primitiveRoot);
        return perAccel.GetOrAdd(key, _ => new SquareCacheEntry(accelerator, length, modulus, primitiveRoot));
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
            foreach (var perAccel in SquareCache.Values)
            {
                foreach (var entry in perAccel.Values)
                {
                    entry.Dispose();
                }
            }

            SquareCache.Clear();
            MulKernelCache.Clear();
            StageKernelCache.Clear();
            ScaleKernelCache.Clear();
            StageMontKernelCache.Clear();
            StageBarrett128KernelCache.Clear();
            ToMont64KernelCache.Clear();
            FromMont64KernelCache.Clear();
            SquareMont64KernelCache.Clear();
            ScaleMont64KernelCache.Clear();
            ForwardKernelCache.Clear();
            InverseKernelCache.Clear();
            // TODO(NTT-OPT): Clear TwiddleCache once introduced.
            return;
        }

        if (SquareCache.TryRemove(accelerator, out var cache))
        {
            foreach (var entry in cache.Values)
            {
                entry.Dispose();
            }
        }

        MulKernelCache.TryRemove(accelerator, out _);
        StageKernelCache.TryRemove(accelerator, out _);
        ScaleKernelCache.TryRemove(accelerator, out _);
        StageMontKernelCache.TryRemove(accelerator, out _);
        StageBarrett128KernelCache.TryRemove(accelerator, out _);
        ToMont64KernelCache.TryRemove(accelerator, out _);
        FromMont64KernelCache.TryRemove(accelerator, out _);
        SquareMont64KernelCache.TryRemove(accelerator, out _);
        ScaleMont64KernelCache.TryRemove(accelerator, out _);
        ForwardKernelCache.TryRemove(accelerator, out _);
        InverseKernelCache.TryRemove(accelerator, out _);
        // TODO(NTT-OPT): Remove per-accelerator twiddle buffers once added.
    }

    private static Action<Index1D, ArrayView<GpuUInt128>, int> GetBitReverseKernel(Accelerator accelerator) =>
        BitReverseKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, int>(BitReverseKernel));

    private static Action<Index1D, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>, GpuUInt128> GetMulKernel(Accelerator accelerator) =>
        MulKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>, GpuUInt128>(MulKernel));

    private static Action<Index1D, ArrayView<GpuUInt128>, int, int, int, ArrayView<GpuUInt128>, GpuUInt128> GetStageKernel(Accelerator accelerator) =>
        StageKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, int, int, int, ArrayView<GpuUInt128>, GpuUInt128>(StageKernel));

    private static Action<Index1D, ArrayView<GpuUInt128>, GpuUInt128, GpuUInt128> GetScaleKernel(Accelerator accelerator) =>
        ScaleKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, GpuUInt128, GpuUInt128>(ScaleKernel));

    private static Action<Index1D, ArrayView<GpuUInt128>, int, int, int, ArrayView<GpuUInt128>, ulong, ulong> GetStageMontKernel(Accelerator accelerator) =>
        StageMontKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, int, int, int, ArrayView<GpuUInt128>, ulong, ulong>(StageMontKernel));

    private static Action<Index1D, ArrayView<GpuUInt128>, int, int, int, ArrayView<GpuUInt128>, ulong, ulong, ulong, ulong> GetStageBarrett128Kernel(Accelerator accelerator) =>
        StageBarrett128KernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, int, int, int, ArrayView<GpuUInt128>, ulong, ulong, ulong, ulong>(StageBarrett128Kernel));

    private static Action<Index1D, ArrayView<GpuUInt128>, GpuUInt128, ulong, ulong, ulong, ulong> GetScaleBarrett128Kernel(Accelerator accelerator) =>
        ScaleBarrett128KernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, GpuUInt128, ulong, ulong, ulong, ulong>(ScaleBarrett128Kernel));

    private static Action<Index1D, ArrayView<GpuUInt128>, ulong, ulong, ulong, ulong> GetSquareBarrett128Kernel(Accelerator accelerator) =>
        SquareBarrett128KernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, ulong, ulong, ulong, ulong>(SquareBarrett128Kernel));

    private static Action<Index1D, ArrayView<GpuUInt128>, ulong, ulong, ulong> GetToMont64Kernel(Accelerator accelerator) =>
        ToMont64KernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, ulong, ulong, ulong>(ToMont64Kernel));

    private static Action<Index1D, ArrayView<GpuUInt128>, ulong, ulong> GetFromMont64Kernel(Accelerator accelerator) =>
        FromMont64KernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, ulong, ulong>(FromMont64Kernel));

    private static Action<Index1D, ArrayView<GpuUInt128>, ulong, ulong> GetSquareMont64Kernel(Accelerator accelerator) =>
        SquareMont64KernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, ulong, ulong>(SquareMont64Kernel));

    private static Action<Index1D, ArrayView<GpuUInt128>, ulong, ulong, ulong> GetScaleMont64Kernel(Accelerator accelerator) =>
        ScaleMont64KernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, ulong, ulong, ulong>(ScaleMont64Kernel));

    private static Action<Index1D, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>, int, GpuUInt128, GpuUInt128> GetForwardKernel(Accelerator accelerator) =>
        ForwardKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>, int, GpuUInt128, GpuUInt128>(ForwardKernel));

    // TODO(NTT-OPT): Introduce stage-wise Cooley–Tukey kernels and a
    // per-(accelerator,length,modulus,root) twiddle cache. The plan:
    // - Precompute twiddle factors for all stages once and cache on GPU.
    // - Replace the current O(n^2) ForwardKernel with O(n log n) butterflies.
    // - Provide accessors like GetStageKernel(...) and GetTwiddleBuffer(...).

    private static Action<Index1D, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>, int, GpuUInt128, GpuUInt128, GpuUInt128> GetInverseKernel(Accelerator accelerator) =>
        InverseKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>, int, GpuUInt128, GpuUInt128, GpuUInt128>(InverseKernel));

    // TODO(NTT-OPT): Mirror the forward stage-wise design for the inverse
    // transform. Precompute inverse twiddles and a normalization factor
    // (n^-1 mod m). Launch one short kernel per stage to avoid TDR.

    private static void MulKernel(Index1D index, ArrayView<GpuUInt128> a, ArrayView<GpuUInt128> b, GpuUInt128 modulus)
    {
        a[index] = a[index].MulMod(b[index], modulus);
    }

    private static void StageKernel(Index1D index, ArrayView<GpuUInt128> data, int len, int half, int stageOffset, ArrayView<GpuUInt128> twiddles, GpuUInt128 modulus)
    {
        // TODO(NTT-OPT): Consider shared-memory tiling to improve locality:
        // load a block of size `len` into shared memory, do butterflies, write back.
        // Requires explicit grouped kernels and chosen group size.
        int t = index.X;
        int j = t % half;
        int block = t / half;
        int k = block * len;
        int i1 = k + j;
        int i2 = i1 + half;
        var u = data[i1];
        var v = data[i2];
        var w = twiddles[stageOffset + j];
        v = v.MulMod(w, modulus);
        var sum = new GpuUInt128(u);
        sum = sum.AddMod(v, modulus);
        u = u.SubMod(v, modulus);
        data[i1] = sum;
        data[i2] = u;
    }

    private static void ScaleKernel(Index1D index, ArrayView<GpuUInt128> data, GpuUInt128 scale, GpuUInt128 modulus)
    {
        var v = data[index];
        v = v.MulMod(scale, modulus);
        data[index] = v;
    }

    // Barrett-based scaling: v = v * scale mod n for 128-bit n
    private static void ScaleBarrett128Kernel(Index1D index, ArrayView<GpuUInt128> data, GpuUInt128 scale, ulong modHigh, ulong modLow, ulong muHigh, ulong muLow)
    {
        var v = data[index];
        // 128x128 -> 256
        Mul128Full(v.High, v.Low, scale.High, scale.Low, out var z3, out var z2, out var z1, out var z0);
        // q approx
        Mul3x2Top(z3, z2, z1, muHigh, muLow, out var qHigh, out var qLow);
        // q*n
        Mul128Full(qHigh, qLow, modHigh, modLow, out var qq3, out var qq2, out var qq1, out var qq0);
        // r = z - q*n
        Sub256(z3, z2, z1, z0, qq3, qq2, qq1, qq0, out var r3, out var r2, out var r1, out var r0);
        ulong resHigh = r1;
        ulong resLow = r0;
        if (Geq128(resHigh, resLow, modHigh, modLow))
        {
            Sub128(ref resHigh, ref resLow, modHigh, modLow);
            if (Geq128(resHigh, resLow, modHigh, modLow))
            {
                Sub128(ref resHigh, ref resLow, modHigh, modLow);
            }
        }

        data[index] = new GpuUInt128(resHigh, resLow);
    }

    // Barrett-based squaring: v = v^2 mod n for 128-bit n
    private static void SquareBarrett128Kernel(Index1D index, ArrayView<GpuUInt128> data, ulong modHigh, ulong modLow, ulong muHigh, ulong muLow)
    {
        var v = data[index];
        Mul128Full(v.High, v.Low, v.High, v.Low, out var z3, out var z2, out var z1, out var z0);
        Mul3x2Top(z3, z2, z1, muHigh, muLow, out var qHigh, out var qLow);
        Mul128Full(qHigh, qLow, modHigh, modLow, out var qq3, out var qq2, out var qq1, out var qq0);
        Sub256(z3, z2, z1, z0, qq3, qq2, qq1, qq0, out var r3, out var r2, out var r1, out var r0);
        ulong resHigh = r1;
        ulong resLow = r0;
        if (Geq128(resHigh, resLow, modHigh, modLow))
        {
            Sub128(ref resHigh, ref resLow, modHigh, modLow);
            if (Geq128(resHigh, resLow, modHigh, modLow))
            {
                Sub128(ref resHigh, ref resLow, modHigh, modLow);
            }
        }
        data[index] = new GpuUInt128(resHigh, resLow);
    }

    // Barrett reduction based stage for 128-bit moduli (no '%').
    private static void StageBarrett128Kernel(Index1D index, ArrayView<GpuUInt128> data, int len, int half, int stageOffset, ArrayView<GpuUInt128> twiddles, ulong modHigh, ulong modLow, ulong muHigh, ulong muLow)
    {
        int t = index.X;
        int j = t % half;
        int block = t / half;
        int k = block * len;
        int i1 = k + j;
        int i2 = i1 + half;
        var u = data[i1];
        var v = data[i2];
        var w = twiddles[stageOffset + j];

        // Compute v * w (128x128) -> 256-bit z3..z0
        Mul128Full(v.High, v.Low, w.High, w.Low, out var z3, out var z2, out var z1, out var z0);

        // t = floor(z / b) where b=2^64 => take (z3,z2,z1)
        // Compute q = floor((t * mu) / b^3). Only need top two limbs of 5-limb product.
        Mul3x2Top(z3, z2, z1, muHigh, muLow, out var qHigh, out var qLow);

        // q * n (128x128) -> 256-bit qq3..qq0
        Mul128Full(qHigh, qLow, modHigh, modLow, out var qq3, out var qq2, out var qq1, out var qq0);

        // r = z - q*n (256-bit)
        Sub256(z3, z2, z1, z0, qq3, qq2, qq1, qq0, out var r3, out var r2, out var r1, out var r0);

        // Reduce r to [0, n) by at most two subtractions
        // Discard higher limbs (should be zero if r < 2n), keep low 128-bit
        ulong resHigh = r1; // r1 is limb1 after subtraction (since lower two limbs are r1 (high) and r0 (low))
        ulong resLow = r0;

        // while (res >= n) res -= n; (at most twice)
        if (Geq128(resHigh, resLow, modHigh, modLow))
        {
            Sub128(ref resHigh, ref resLow, modHigh, modLow);
            if (Geq128(resHigh, resLow, modHigh, modLow))
            {
                Sub128(ref resHigh, ref resLow, modHigh, modLow);
            }
        }

        var sum = new GpuUInt128(u);
        var mulred = new GpuUInt128(resHigh, resLow);
        sum = sum.AddMod(mulred, new GpuUInt128(modHigh, modLow));
        u = u.SubMod(mulred, new GpuUInt128(modHigh, modLow));
        data[i1] = sum;
        data[i2] = u;
    }

    // TODO(MOD-OPT): Stage kernel using 64-bit Montgomery multiplication. Requires data and twiddles in Montgomery domain.
    private static void StageMontKernel(Index1D index, ArrayView<GpuUInt128> data, int len, int half, int stageOffset, ArrayView<GpuUInt128> twiddlesMont, ulong modulus, ulong nPrime)
    {
        int t = index.X;
        int j = t % half;
        int block = t / half;
        int k = block * len;
        int i1 = k + j;
        int i2 = i1 + half;
        ulong u = data[i1].Low;
        ulong v = data[i2].Low;
        ulong wR = twiddlesMont[stageOffset + j].Low;
        v = MontMul64(v, wR, modulus, nPrime);
        ulong sum = u + v;
        if (sum >= modulus)
        {
            sum -= modulus;
        }
        ulong diff = u >= v ? u - v : unchecked(u + modulus - v);
        data[i1] = new GpuUInt128(0UL, sum);
        data[i2] = new GpuUInt128(0UL, diff);
    }

    // Computes Montgomery multiplication for 64-bit operands in Montgomery domain.
    // Returns a*b*R^{-1} mod modulus, where R=2^64 and nPrime = -modulus^{-1} mod 2^64.
    private static ulong MontMul64(ulong aR, ulong bR, ulong modulus, ulong nPrime)
    {
        // t = aR * bR (128-bit)
        ulong tLow, tHigh;
        Mul64Parts(aR, bR, out tHigh, out tLow);
        // m = (tLow * nPrime) mod 2^64 -> just low 64 bits of the product
        ulong mLow, mHigh;
        Mul64Parts(tLow, nPrime, out mHigh, out mLow);
        // u = (t + m*modulus) >> 64
        ulong mmLow, mmHigh;
        Mul64Parts(mLow, modulus, out mmHigh, out mmLow);
        // add tLow + mmLow, keep carry
        ulong carry = 0UL;
        ulong low = tLow + mmLow;
        if (low < tLow)
        {
            carry = 1UL;
        }
        // high = tHigh + mmHigh + carry
        ulong high = tHigh + mmHigh + carry;
        ulong u = high; // this is (t + m*n) >> 64
        if (u >= modulus)
        {
            u -= modulus;
        }
        return u;
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Mul128Full(ulong aHigh, ulong aLow, ulong bHigh, ulong bLow, out ulong p3, out ulong p2, out ulong p1, out ulong p0)
    {
        // Full 128x128 -> 256 multiply using 64-bit limbs
        Mul64Parts(aLow, bLow, out var h0, out var l0);
        Mul64Parts(aLow, bHigh, out var h1, out var l1);
        Mul64Parts(aHigh, bLow, out var h2, out var l2);
        Mul64Parts(aHigh, bHigh, out var h3, out var l3);

        p0 = l0;
        ulong carry = 0UL;
        ulong sum = h0;
        sum += l1; if (sum < l1) carry++;
        sum += l2; if (sum < l2) carry++;
        p1 = sum;

        sum = h1;
        sum += h2; ulong carry2 = sum < h2 ? 1UL : 0UL;
        sum += l3; if (sum < l3) carry2++;
        sum += carry; if (sum < carry) carry2++;
        p2 = sum;
        p3 = h3 + carry2;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Mul3x2Top(ulong t2, ulong t1, ulong t0, ulong muHigh, ulong muLow, out ulong top1, out ulong top0)
    {
        // Compute top two limbs of ( (t2,t1,t0) * (muHigh,muLow) ) after shifting right by 192 bits.
        // We build full 5 limbs and return p4 (top1) and p3 (top0).
        ulong p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0;

        void Add128To(ref ulong lo, ref ulong hi, ulong addLo, ulong addHi)
        {
            ulong s = lo + addLo;
            ulong c = s < lo ? 1UL : 0UL;
            lo = s;
            ulong s2 = hi + addHi + c;
            hi = s2;
        }

        // t0 * muLow -> contributes to p0,p1
        Mul64Parts(t0, muLow, out var h, out var l);
        Add128To(ref p0, ref p1, l, h);

        // t0 * muHigh -> to p1,p2
        Mul64Parts(t0, muHigh, out h, out l);
        Add128To(ref p1, ref p2, l, h);

        // t1 * muLow -> to p1,p2
        Mul64Parts(t1, muLow, out h, out l);
        Add128To(ref p1, ref p2, l, h);

        // t1 * muHigh -> to p2,p3
        Mul64Parts(t1, muHigh, out h, out l);
        Add128To(ref p2, ref p3, l, h);

        // t2 * muLow -> to p2,p3
        Mul64Parts(t2, muLow, out h, out l);
        Add128To(ref p2, ref p3, l, h);

        // t2 * muHigh -> to p3,p4
        Mul64Parts(t2, muHigh, out h, out l);
        Add128To(ref p3, ref p4, l, h);

        top1 = p4; top0 = p3;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Sub256(ulong a3, ulong a2, ulong a1, ulong a0, ulong b3, ulong b2, ulong b1, ulong b0, out ulong r3, out ulong r2, out ulong r1, out ulong r0)
    {
        ulong borrow = 0;
        r0 = SubWithBorrow(a0, b0, ref borrow);
        r1 = SubWithBorrow(a1, b1, ref borrow);
        r2 = SubWithBorrow(a2, b2, ref borrow);
        r3 = SubWithBorrow(a3, b3, ref borrow);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong SubWithBorrow(ulong a, ulong b, ref ulong borrow)
    {
        ulong res = a - b - borrow;
        borrow = ((a ^ b) & (a ^ res) & 0x8000_0000_0000_0000UL) != 0 ? 1UL : (a < b + borrow ? 1UL : 0UL);
        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool Geq128(ulong aHigh, ulong aLow, ulong bHigh, ulong bLow)
    {
        return aHigh > bHigh || (aHigh == bHigh && aLow >= bLow);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Sub128(ref ulong aHigh, ref ulong aLow, ulong bHigh, ulong bLow)
    {
        ulong borrow = 0;
        ulong low = aLow - bLow;
        borrow = aLow < bLow ? 1UL : 0UL;
        ulong high = aHigh - bHigh - borrow;
        aHigh = high; aLow = low;
    }

    private static void ToMont64Kernel(Index1D index, ArrayView<GpuUInt128> data, ulong modulus, ulong nPrime, ulong r2)
    {
        ulong a = data[index].Low;
        ulong v = MontMul64(a, r2, modulus, nPrime);
        data[index] = new GpuUInt128(0UL, v);
    }

    private static void FromMont64Kernel(Index1D index, ArrayView<GpuUInt128> data, ulong modulus, ulong nPrime)
    {
        ulong aR = data[index].Low;
        ulong v = MontMul64(aR, 1UL, modulus, nPrime);
        data[index] = new GpuUInt128(0UL, v);
    }

    private static void SquareMont64Kernel(Index1D index, ArrayView<GpuUInt128> data, ulong modulus, ulong nPrime)
    {
        ulong aR = data[index].Low;
        ulong v = MontMul64(aR, aR, modulus, nPrime);
        data[index] = new GpuUInt128(0UL, v);
    }

    private static void ScaleMont64Kernel(Index1D index, ArrayView<GpuUInt128> data, ulong modulus, ulong nPrime, ulong scaleMont)
    {
        ulong aR = data[index].Low;
        ulong v = MontMul64(aR, scaleMont, modulus, nPrime);
        data[index] = new GpuUInt128(0UL, v);
    }

    private static void ForwardKernel(Index1D index, ArrayView<GpuUInt128> input, ArrayView<GpuUInt128> output, int length, GpuUInt128 modulus, GpuUInt128 root)
    {
        // TODO(NTT-OPT): Replace this reference O(n^2) kernel with a stage-wise
        // Cooley–Tukey butterfly kernel (O(n log n)). This version uses a per-
        // element ModPow in the hot loop which is far too slow and causes long
        // single-kernel runtimes. After twiddle precomputation, each butterfly
        // should perform: (u, v) -> (u+v*w, u-v*w) with a single MulMod.
        var sum = new GpuUInt128(0UL, 0UL);
        for (int i = 0; i < length; i++)
        {
            var term = input[i];
            var power = new GpuUInt128(root);
            power = power.ModPow((ulong)index.X * (ulong)i, modulus);
            term = term.MulMod(power, modulus);
            sum = sum.AddMod(term, modulus);
        }

        output[index] = sum;
    }

    private static void InverseKernel(Index1D index, ArrayView<GpuUInt128> input, ArrayView<GpuUInt128> output, int length, GpuUInt128 modulus, GpuUInt128 rootInv, GpuUInt128 nInv)
    {
        // TODO(NTT-OPT): Replace this O(n^2) inverse with stage-wise butterflies
        // using precomputed inverse twiddles. Apply final scaling by nInv after
        // all stages, preferably in a small dedicated kernel.
        var sum = new GpuUInt128(0UL, 0UL);
        for (int i = 0; i < length; i++)
        {
            var term = input[i];
            var power = new GpuUInt128(rootInv);
            power = power.ModPow((ulong)index.X * (ulong)i, modulus);
            term = term.MulMod(power, modulus);
            sum = sum.AddMod(term, modulus);
        }

        sum = sum.MulMod(nInv, modulus);
        output[index] = sum;
    }

    private static void BitReverseKernel(Index1D index, ArrayView<GpuUInt128> data, int bits)
    {
        int i = index.X;
        int j = ReverseBits(i, bits);
        if (i < j)
        {
            var tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }

    private static int ReverseBits(int value, int bits)
    {
        int result = 0;
        for (int i = 0; i < bits; i++)
        {
            result = (result << 1) | (value & 1);
            value >>= 1;
        }

        return result;
    }

    public static void BitReverse(Span<GpuUInt128> values)
    {
        int n = values.Length;
        int bits = (int)Math.Log2(n);
        for (int i = 0; i < n; i++)
        {
            int j = ReverseBits(i, bits);
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
        using var gpu = GpuContextPool.Rent();
        var accelerator = gpu.Accelerator;
        var kernel = GetBitReverseKernel(accelerator);
        var pool = ArrayPool<GpuUInt128>.Shared;
        var array = pool.Rent(n);
        values.CopyTo(array);
        using var buffer = accelerator.Allocate1D<GpuUInt128>(n);
        buffer.View.CopyFromCPU(ref array[0], n);
        kernel(n, buffer.View, bits);
        accelerator.Synchronize();
        buffer.View.CopyToCPU(ref array[0], n);
        array.AsSpan(0, n).CopyTo(values);
        pool.Return(array, clearArray: true);
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
            wLen = wLen.ModPow((ulong)expBase, modulus);
            for (int i = 0; i < n; i += len)
            {
                var w = new GpuUInt128(0UL, 1UL);
                int half = len >> 1;
                for (int j = 0; j < half; j++)
                {
                    var u = values[i + j];
                    var v = values[i + j + half];
                    v = v.MulMod(w, modulus);
                    var sum = new GpuUInt128(u);
                    sum = sum.AddMod(v, modulus);
                    u = u.SubMod(v, modulus);
                    values[i + j] = sum;
                    values[i + j + half] = u;
                    w = w.MulMod(wLen, modulus);
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
        using var gpu = GpuContextPool.Rent();
        var accelerator = gpu.Accelerator;
        var kernel = GetForwardKernel(accelerator);
        var pool = ArrayPool<GpuUInt128>.Shared;
        var inputArray = pool.Rent(n);
        var outputArray = pool.Rent(n);
        values.CopyTo(inputArray);
        using var inputBuffer = accelerator.Allocate1D<GpuUInt128>(n);
        using var outputBuffer = accelerator.Allocate1D<GpuUInt128>(n);
        inputBuffer.View.CopyFromCPU(ref inputArray[0], n);
        UInt128 modValue = modulus;
        UInt128 expBase = (modValue - 1UL) / (ulong)n;
        var root = new GpuUInt128(primitiveRoot);
        root = root.ModPow((ulong)expBase, modulus);
        kernel(n, inputBuffer.View, outputBuffer.View, n, modulus, root);
        accelerator.Synchronize();
        outputBuffer.View.CopyToCPU(ref outputArray[0], n);
        outputArray.AsSpan(0, n).CopyTo(values);
        pool.Return(inputArray, clearArray: true);
        pool.Return(outputArray, clearArray: true);
    }

    public static void Inverse(Span<GpuUInt128> values, GpuUInt128 modulus, GpuUInt128 primitiveRoot)
    {
        int n = values.Length;
        BitReverse(values);
        UInt128 modValue = modulus;
        var root = new GpuUInt128(primitiveRoot);
        if (modulus.High == 0UL)
        {
            root = root.ModInv(modulus.Low);
        }
        else
        {
            root = root.ModInv(modulus);
        }
		
        for (int len = 2; len <= n; len <<= 1)
		{
			UInt128 expBase = (modValue - 1UL) / (ulong)len;
			var wLen = new GpuUInt128(root);
			wLen = wLen.ModPow((ulong)expBase, modulus);
			for (int i = 0; i < n; i += len)
			{
				var w = new GpuUInt128(0UL, 1UL);
				int half = len >> 1;
				for (int j = 0; j < half; j++)
				{
					var u = values[i + j];
					var v = values[i + j + half];
					v = v.MulMod(w, modulus);
					var sum = new GpuUInt128(u);
					sum = sum.AddMod(v, modulus);
					u = u.SubMod(v, modulus);
					values[i + j] = sum;
					values[i + j + half] = u;
					w = w.MulMod(wLen, modulus);
				}
			}
		}

        var nInv = new GpuUInt128((UInt128)n);
        if (modulus.High == 0UL)
        {
            nInv = nInv.ModInv(modulus.Low);
        }
        else
        {
            nInv = nInv.ModInv(modulus);
        }

        for (int i = 0; i < n; i++)
        {
            values[i] = values[i].MulMod(nInv, modulus);
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
        using var gpu = GpuContextPool.Rent();
        var accelerator = gpu.Accelerator;
        var kernel = GetInverseKernel(accelerator);
        var pool = ArrayPool<GpuUInt128>.Shared;
        var inputArray = pool.Rent(n);
        var outputArray = pool.Rent(n);
        values.CopyTo(inputArray);
        using var inputBuffer = accelerator.Allocate1D<GpuUInt128>(n);
        using var outputBuffer = accelerator.Allocate1D<GpuUInt128>(n);
        inputBuffer.View.CopyFromCPU(ref inputArray[0], n);
        UInt128 modValue = modulus;
        UInt128 expBase = (modValue - 1UL) / (ulong)n;
        var root = new GpuUInt128(primitiveRoot);
        root = root.ModPow((ulong)expBase, modulus);
        var rootInv = new GpuUInt128(root);
        if (modulus.High == 0UL)
        {
            rootInv = rootInv.ModInv(modulus.Low);
        }
        else
        {
            rootInv = rootInv.ModInv(modulus);
        }

        var nInv = new GpuUInt128((UInt128)n);
        if (modulus.High == 0UL)
        {
            nInv = nInv.ModInv(modulus.Low);
        }
        else
        {
            nInv = nInv.ModInv(modulus);
        }
        kernel(n, inputBuffer.View, outputBuffer.View, n, modulus, rootInv, nInv);
        accelerator.Synchronize();
        outputBuffer.View.CopyToCPU(ref outputArray[0], n);
        outputArray.AsSpan(0, n).CopyTo(values);
        pool.Return(inputArray, clearArray: true);
        pool.Return(outputArray, clearArray: true);
    }

    private static void ForwardGpuStaged(Span<GpuUInt128> values, GpuUInt128 modulus, GpuUInt128 primitiveRoot)
    {
        int n = values.Length;
        using var gpu = GpuContextPool.Rent();
        var accelerator = gpu.Accelerator;
        using var buffer = accelerator.Allocate1D<GpuUInt128>(n);
        buffer.View.CopyFromCPU(ref values[0], n);
        var cache = GetSquareCache(accelerator, n, modulus, primitiveRoot);
        ForwardDevice(accelerator, buffer.View, n, modulus, cache);
        buffer.View.CopyToCPU(ref values[0], n);
    }

    private static void InverseGpuStaged(Span<GpuUInt128> values, GpuUInt128 modulus, GpuUInt128 primitiveRoot)
    {
        int n = values.Length;
        using var gpu = GpuContextPool.Rent();
        var accelerator = gpu.Accelerator;
        using var buffer = accelerator.Allocate1D<GpuUInt128>(n);
        buffer.View.CopyFromCPU(ref values[0], n);
        var cache = GetSquareCache(accelerator, n, modulus, primitiveRoot);
        InverseDevice(accelerator, buffer.View, n, modulus, cache);
        buffer.View.CopyToCPU(ref values[0], n);
    }

    private static void ForwardDevice(Accelerator accelerator, ArrayView<GpuUInt128> data, int length, GpuUInt128 modulus, SquareCacheEntry cache)
    {
        var bitKernel = GetBitReverseKernel(accelerator);
        bitKernel(length, data, cache.Bits);
        accelerator.Synchronize();

        if (cache.UseMontgomery64)
        {
            var toMont = GetToMont64Kernel(accelerator);
            toMont(length, data, cache.ModulusLow, cache.MontNPrime64, cache.MontR2Mod64);
            accelerator.Synchronize();
            var stageKernel = GetStageMontKernel(accelerator);
            int butterflies = length >> 1;
            for (int s = 0; s < cache.Bits; s++)
            {
                int len = 1 << (s + 1);
                int half = len >> 1;
                stageKernel(butterflies, data, len, half, cache.StageOffsets[s], cache.TwiddlesMont!.View, cache.ModulusLow, cache.MontNPrime64);
                accelerator.Synchronize();
            }
            var fromMont = GetFromMont64Kernel(accelerator);
            fromMont(length, data, cache.ModulusLow, cache.MontNPrime64);
        }
        else if (cache.UseBarrett128)
        {
            var stageKernel = GetStageBarrett128Kernel(accelerator);
            int butterflies = length >> 1;
            for (int s = 0; s < cache.Bits; s++)
            {
                int len = 1 << (s + 1);
                int half = len >> 1;
                stageKernel(butterflies, data, len, half, cache.StageOffsets[s], cache.Twiddles.View, cache.ModulusHigh, cache.ModulusLow, cache.BarrettMuHigh, cache.BarrettMuLow);
                accelerator.Synchronize();
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
                stageKernel(butterflies, data, len, half, cache.StageOffsets[s], cache.Twiddles.View, modulus);
                accelerator.Synchronize();
            }
        }

        accelerator.Synchronize();
    }

    private static void InverseDevice(Accelerator accelerator, ArrayView<GpuUInt128> data, int length, GpuUInt128 modulus, SquareCacheEntry cache)
    {
        var bitKernel = GetBitReverseKernel(accelerator);
        bitKernel(length, data, cache.Bits);
        accelerator.Synchronize();

        if (cache.UseMontgomery64)
        {
            var toMont = GetToMont64Kernel(accelerator);
            toMont(length, data, cache.ModulusLow, cache.MontNPrime64, cache.MontR2Mod64);
            accelerator.Synchronize();
            var stageKernel = GetStageMontKernel(accelerator);
            int butterflies = length >> 1;
            for (int s = 0; s < cache.Bits; s++)
            {
                int len = 1 << (s + 1);
                int half = len >> 1;
                stageKernel(butterflies, data, len, half, cache.StageOffsets[s], cache.TwiddlesInvMont!.View, cache.ModulusLow, cache.MontNPrime64);
                accelerator.Synchronize();
            }
            var scaleKernel = GetScaleMont64Kernel(accelerator);
            scaleKernel(length, data, cache.ModulusLow, cache.MontNPrime64, cache.MontNInvR64);
            accelerator.Synchronize();
            var fromMont = GetFromMont64Kernel(accelerator);
            fromMont(length, data, cache.ModulusLow, cache.MontNPrime64);
        }
        else if (cache.UseBarrett128)
        {
            var stageKernel = GetStageBarrett128Kernel(accelerator);
            int butterflies = length >> 1;
            for (int s = 0; s < cache.Bits; s++)
            {
                int len = 1 << (s + 1);
                int half = len >> 1;
                stageKernel(butterflies, data, len, half, cache.StageOffsets[s], cache.TwiddlesInv.View, cache.ModulusHigh, cache.ModulusLow, cache.BarrettMuHigh, cache.BarrettMuLow);
                accelerator.Synchronize();
            }
            var scaleKernel = GetScaleBarrett128Kernel(accelerator);
            scaleKernel(length, data, cache.NInv, cache.ModulusHigh, cache.ModulusLow, cache.BarrettMuHigh, cache.BarrettMuLow);
            accelerator.Synchronize();
        }
        else
        {
            var stageKernel = GetStageKernel(accelerator);
            int butterflies = length >> 1;
            for (int s = 0; s < cache.Bits; s++)
            {
                int len = 1 << (s + 1);
                int half = len >> 1;
                stageKernel(butterflies, data, len, half, cache.StageOffsets[s], cache.TwiddlesInv.View, modulus);
                accelerator.Synchronize();
            }
            var scaleKernel = GetScaleKernel(accelerator);
            scaleKernel(length, data, cache.NInv, modulus);
            accelerator.Synchronize();
        }

        accelerator.Synchronize();
    }

    public static void Convolve(Span<GpuUInt128> left, Span<GpuUInt128> right, GpuUInt128 modulus, GpuUInt128 primitiveRoot)
    {
        int n = 1;
        int resultLength = left.Length + right.Length;
        while (n < resultLength)
        {
            n <<= 1;
        }

        var pool = ArrayPool<GpuUInt128>.Shared;
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

        var pool = ArrayPool<GpuUInt128>.Shared;
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

    public static void SquareDevice(Accelerator accelerator, ArrayView<GpuUInt128> values, GpuUInt128 modulus, GpuUInt128 primitiveRoot)
    {
        var cache = GetSquareCache(accelerator, (int)values.Length, modulus, primitiveRoot);
        ForwardDevice(accelerator, values, (int)values.Length, modulus, cache);
        var mul = GetMulKernel(accelerator);
        mul((int)values.Length, values, values, modulus);
        InverseDevice(accelerator, values, (int)values.Length, modulus, cache);
    }

    public static void PointwiseMultiply(Span<GpuUInt128> left, Span<GpuUInt128> right, GpuUInt128 modulus)
    {
        using var gpu = GpuContextPool.Rent();
        var accelerator = gpu.Accelerator;
        var kernel = GetMulKernel(accelerator);
        var pool = ArrayPool<GpuUInt128>.Shared;
        var leftArray = pool.Rent(left.Length);
        var rightArray = pool.Rent(right.Length);
        left.CopyTo(leftArray);
        right.CopyTo(rightArray);
        using var leftBuffer = accelerator.Allocate1D<GpuUInt128>(left.Length);
        using var rightBuffer = accelerator.Allocate1D<GpuUInt128>(right.Length);
        leftBuffer.View.CopyFromCPU(ref leftArray[0], left.Length);
        rightBuffer.View.CopyFromCPU(ref rightArray[0], right.Length);
        kernel(left.Length, leftBuffer.View, rightBuffer.View, modulus);
        accelerator.Synchronize();
        leftBuffer.View.CopyToCPU(ref leftArray[0], left.Length);
        rightBuffer.View.CopyToCPU(ref rightArray[0], right.Length);
        leftArray.AsSpan(0, left.Length).CopyTo(left);
        rightArray.AsSpan(0, right.Length).CopyTo(right);
        pool.Return(leftArray, clearArray: true);
        pool.Return(rightArray, clearArray: true);
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

