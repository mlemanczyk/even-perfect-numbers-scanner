using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
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
        return KernelCache.GetOrAdd(accelerator, accel => accel.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, ArrayView<GpuUInt128>>(Kernel));
    }

    private Action<Index1D, ulong, ArrayView<GpuUInt128>> GetAddSmallKernel(Accelerator accelerator) =>
        AddSmallKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ulong, ArrayView<GpuUInt128>>(AddSmallKernel));

    private Action<Index1D, ulong, ArrayView<GpuUInt128>> GetSubSmallKernel(Accelerator accelerator) =>
        SubSmallKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ulong, ArrayView<GpuUInt128>>(SubtractSmallKernel));

    private Action<Index1D, ulong, ArrayView<GpuUInt128>> GetReduceKernel(Accelerator accelerator) =>
        ReduceKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ulong, ArrayView<GpuUInt128>>(ReduceModMersenneKernel));

    private Action<Index1D, ArrayView<GpuUInt128>, ArrayView<byte>> GetIsZeroKernel(Accelerator accelerator) =>
        IsZeroKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuUInt128>, ArrayView<byte>>(IsZeroKernel));

    private Action<Index1D, ArrayView<ulong>, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>> GetBatchKernel(Accelerator accelerator) =>
        BatchKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<GpuUInt128>, ArrayView<GpuUInt128>>(KernelBatch));

    // Configurable LL slice size to keep kernels short. Default 32.
    public int SliceSize = 32;

    public bool IsMersennePrime(ulong exponent)
    {
        // Default to global kernel preference for backward compatibility.
        bool runOnGpu = !GpuContextPool.ForceCpu;
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
        var kernel = GetBatchKernel(accelerator); // TODO: Switch this batch path to the ProcessEightBitWindows residue kernel once Lucas–Lehmer integrates the benchmarked windowed pow2 helper for small exponents.

        var expBuffer = accelerator.Allocate1D<ulong>(count);
        ulong[] expArray = ArrayPool<ulong>.Shared.Rent(count);
        exponents.CopyTo(expArray);
        expBuffer.View.CopyFromCPU(ref expArray[0], count);

        GpuUInt128[] modulusArray = ArrayPool<GpuUInt128>.Shared.Rent(count);
        for (int i = 0; i < count; i++)
        {
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

            if (!int.TryParse(parts[0], out var storedLength) || storedLength != length)
            {
                continue;
            }

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

    private static ulong ModPow(ulong value, ulong exponent, ulong modulus)
    {
        ulong result = 1UL;
        while (exponent > 0UL)
        {
            if ((exponent & 1UL) != 0UL)
            {
                // TODO: Route these powmods through the ProcessEightBitWindows helper once it lands for GPU
                // host-side fallbacks so the Lucas–Lehmer setup reuses the benchmarked windowed ladder.
                result = MulMod(result, value, modulus);
            }

            value = MulMod(value, value, modulus);
            exponent >>= 1;
        }

        return result;
    }

    private static ulong MulMod(ulong a, ulong b, ulong modulus)
    {
        // TODO: Swap this UInt128 `%` reduction for the GPU-compatible MulMod helper once it adopts the faster
        // inline UInt128 path benchmarked in MulMod64Benchmarks so host/GPU parity avoids BigInteger-style fallbacks.
        return (ulong)(((UInt128)a * b) % modulus);
    }

    private static void AddSmallKernel(Index1D index, ulong add, ArrayView<GpuUInt128> value)
    {
        ulong carry = add;
        for (int i = 0; i < value.Length && carry != 0UL; i++)
        {
            var limb = value[i];
            ulong low = limb.Low + carry;
            ulong carryOut = low < carry ? 1UL : 0UL;
            ulong high = limb.High + carryOut;
            carry = high < limb.High ? 1UL : 0UL;
            value[i] = new GpuUInt128(high, low);
        }
    }

    private static void SubtractSmallKernel(Index1D index, ulong subtract, ArrayView<GpuUInt128> value)
    {
        ulong borrow = subtract;
        for (int i = 0; i < value.Length && borrow != 0UL; i++)
        {
            var limb = value[i];
            ulong low = limb.Low;
            ulong high = limb.High;
            if (low >= borrow)
            {
                low -= borrow;
                borrow = 0UL;
            }
            else
            {
                low = unchecked(low - borrow);
                borrow = 1UL;
                if (high != 0UL)
                {
                    high--;
                    borrow = 0UL;
                }
                else
                {
                    high = ulong.MaxValue;
                }
            }

            value[i] = new GpuUInt128(high, low);
        }
    }

    private static void ReduceModMersenneKernel(Index1D index, ulong exponent, ArrayView<GpuUInt128> value)
    {
        int limbCount = (int)((exponent + 127UL) / 128UL);
        for (int i = limbCount; i < value.Length; i++)
        {
            var limb = value[i];
            if (limb.IsZero)
            {
                continue;
            }

            int target = i - limbCount;
            var sum = value[target];
            var original = sum;
            sum.Add(limb);
            value[target] = sum;
            value[i] = new GpuUInt128(0UL, 0UL);
            if (sum.CompareTo(original) < 0)
            {
                int j = target + 1;
                while (j < limbCount)
                {
                    var next = value[j];
                    var nextOriginal = next;
                    next.Add(1UL);
                    value[j] = next;
                    if (next.CompareTo(nextOriginal) >= 0)
                    {
                        break;
                    }

                    j++;
                }
            }
        }

        int topBits = (int)exponent.Mod128();
        if (topBits == 0)
        {
            topBits = 128;
        }

        int topIndex = limbCount - 1;
        ulong topLowMask = topBits >= 64 ? ulong.MaxValue : (1UL << topBits) - 1UL;
        ulong topHighMask = topBits > 64 ? (1UL << (topBits - 64)) - 1UL : 0UL;
        var top = value[topIndex];
        var carryBits = top >> topBits;
        top = new(top.High & topHighMask, top.Low & topLowMask);
        value[topIndex] = top;

        if (!carryBits.IsZero)
        {
            int j = 0;
            while (j < limbCount && !carryBits.IsZero)
            {
                var cur = value[j];
                var curOriginal = cur;
                cur.Add(carryBits);
                value[j] = cur;
                carryBits = cur.CompareTo(curOriginal) < 0 ? new GpuUInt128(0UL, 1UL) : new GpuUInt128(0UL, 0UL);
                j++;
            }
        }

        bool geq = true;
        for (int i = topIndex; i >= 0; i--)
        {
            var limb = value[i];
            ulong modHigh = i == topIndex ? topHighMask : ulong.MaxValue;
            ulong modLow = i == topIndex ? topLowMask : ulong.MaxValue;
            if (limb.High > modHigh || (limb.High == modHigh && limb.Low > modLow))
            {
                geq = true;
                break;
            }

            if (limb.High < modHigh || (limb.High == modHigh && limb.Low < modLow))
            {
                geq = false;
                break;
            }
        }

        if (geq)
        {
            ulong borrow = 0UL;
            for (int i = 0; i < limbCount - 1; i++)
            {
                var limb = value[i];
                ulong newLow = limb.Low - ulong.MaxValue - borrow;
                borrow = limb.Low < ulong.MaxValue + borrow ? 1UL : 0UL;
                ulong newHigh = limb.High - ulong.MaxValue - borrow;
                borrow = limb.High < ulong.MaxValue + borrow ? 1UL : 0UL;
                value[i] = new GpuUInt128(newHigh, newLow);
            }

            var topLimb = value[topIndex];
            ulong newTopLow = topLimb.Low - topLowMask - borrow;
            borrow = topLimb.Low < topLowMask + borrow ? 1UL : 0UL;
            ulong newTopHigh = topLimb.High - topHighMask - borrow;
            value[topIndex] = new GpuUInt128(newTopHigh, newTopLow);
        }

        for (int i = limbCount; i < value.Length; i++)
        {
            value[i] = new GpuUInt128(0UL, 0UL);
        }
    }

    private static void IsZeroKernel(Index1D index, ArrayView<GpuUInt128> value, ArrayView<byte> result)
    {
        byte isZero = 1;
        for (int i = 0; i < value.Length; i++)
        {
            var limb = value[i];
            if (limb.High != 0UL || limb.Low != 0UL)
            {
                isZero = 0;
                break;
            }
        }

        result[0] = isZero;
    }

    private static void ReduceModMersenne(Span<GpuUInt128> value, ulong exponent)
    {
        int limbCount = (int)((exponent + 127UL) / 128UL);
        for (int i = limbCount; i < value.Length; i++)
        {
            var limb = value[i];
            if (limb.IsZero)
            {
                continue;
            }

            int target = i - limbCount;
            UInt128 original = value[target];
            UInt128 sum = original + (UInt128)limb;
            bool carry = sum < original;
            value[target] = new GpuUInt128(sum);
            value[i] = new GpuUInt128(0UL, 0UL);
            if (carry)
            {
                int j = target + 1;
                while (j < limbCount)
                {
                    UInt128 next = value[j];
                    UInt128 nextSum = next + 1UL;
                    value[j] = new GpuUInt128(nextSum);
                    if (nextSum != 0UL)
                    {
                        break;
                    }

                    j++;
                }
            }
        }

        int topBits = (int)exponent.Mod128();
        if (topBits == 0)
        {
            topBits = 128;
        }

        int topIndex = limbCount - 1;
        ulong topLowMask = topBits >= 64 ? ulong.MaxValue : (1UL << topBits) - 1UL;
        ulong topHighMask = topBits > 64 ? (1UL << (topBits - 64)) - 1UL : 0UL;
        UInt128 mask = ((UInt128)topHighMask << 64) | topLowMask;
        UInt128 top = value[topIndex];
        UInt128 carryBits = top >> topBits;
        top &= mask;
        value[topIndex] = new GpuUInt128(top);
        if (carryBits != 0UL)
        {
            int j = 0;
            UInt128 carry = carryBits;
            while (j < limbCount && carry != 0UL)
            {
                UInt128 cur = value[j];
                UInt128 sum = cur + carry;
                bool overflow = sum < cur;
                value[j] = new GpuUInt128(sum);
                carry = overflow ? 1UL : 0UL;
                j++;
            }
        }

        bool geq = true;
        for (int i = topIndex; i >= 0; i--)
        {
            UInt128 limb = value[i];
            UInt128 modLimb = i == topIndex ? mask : UInt128.MaxValue;
            if (limb > modLimb)
            {
                geq = true;
                break;
            }

            if (limb < modLimb)
            {
                geq = false;
                break;
            }
        }

        if (geq)
        {
            ulong borrow = 0UL;
            for (int i = 0; i < limbCount - 1; i++)
            {
                ulong low = value[i].Low;
                ulong high = value[i].High;
                ulong newLow = low - ulong.MaxValue - borrow;
                borrow = low < ulong.MaxValue + borrow ? 1UL : 0UL;
                ulong newHigh = high - ulong.MaxValue - borrow;
                borrow = high < ulong.MaxValue + borrow ? 1UL : 0UL;
                value[i] = new GpuUInt128(newHigh, newLow);
            }

            ulong topLow = value[topIndex].Low;
            ulong topHigh = value[topIndex].High;
            ulong newTopLow = topLow - topLowMask - borrow;
            borrow = topLow < topLowMask + borrow ? 1UL : 0UL;
            ulong newTopHigh = topHigh - topHighMask - borrow;
            value[topIndex] = new GpuUInt128(newTopHigh, newTopLow);
        }

        for (int i = limbCount; i < value.Length; i++)
        {
            value[i] = new GpuUInt128(0UL, 0UL);
        }
    }

    private static void KernelBatch(Index1D index, ArrayView<ulong> exponents, ArrayView<GpuUInt128> moduli, ArrayView<GpuUInt128> states)
    {
        int idx = index.X;
        ulong exponent = exponents[idx];
        GpuUInt128 modulus = moduli[idx];
        var s = new GpuUInt128(0UL, 4UL);
        ulong limit = exponent - 2UL;
        for (ulong i = 0UL; i < limit; i++)
        {
            s = SquareModMersenne128(s, exponent);
            s.SubMod(2UL, modulus);
        }

        states[idx] = s;
    }

    private static void Kernel(Index1D index, ulong exponent, GpuUInt128 modulus, ArrayView<GpuUInt128> state)
    {
        // Lucas–Lehmer iteration in the field GF(2^p-1).
        // For p < 128, use a Mersenne-specific squaring + reduction to avoid
        // the generic 128-bit long-division-like reduction.
        var s = new GpuUInt128(0UL, 4UL);
        ulong i = 0UL, limit = exponent - 2UL;
        if (exponent < 128UL)
        {
            for (; i < limit; i++)
            {
                s = SquareModMersenne128(s, exponent);
                s.SubMod(2UL, modulus);
            }
        }
        else
        {
            for (; i < limit; i++)
            {
                s.SquareMod(modulus);
                s.SubMod(2UL, modulus);
            }
        }

        state[0] = s;
    }

    // Squares a 128-bit value and reduces modulo M_p = 2^p - 1 using
    // Mersenne folding. Valid for 0 < p < 128 and input s in [0, M_p].
    private static GpuUInt128 SquareModMersenne128(GpuUInt128 s, ulong p)
    {
        // Full square: (H*2^64 + L)^2 = H^2*2^128 + 2HL*2^64 + L^2
        SquareFull128(s, out var q3, out var q2, out var q1, out var q0);

        // v = low128 + high128 (fold by 128 bits once)
        var vHigh = new GpuUInt128(q3, q2);
        var vLow = new GpuUInt128(q1, q0);
        vLow.Add(vHigh);

        // Mask to keep only p low bits and fold the carry bits (v >> p)
        int topBits = (int)p.Mod128();
        if (topBits == 0)
        {
            topBits = 128;
        }

        ulong maskLow = topBits >= 64 ? ulong.MaxValue : (1UL << topBits) - 1UL;
        ulong maskHigh = topBits > 64 ? (1UL << (topBits - 64)) - 1UL : 0UL;

        var carryBits = vLow >> topBits;
		vLow = new(vLow.High & maskHigh, vLow.Low & maskLow);

		// Propagate carryBits back into the masked value (single-limb fold)
		GpuUInt128 before;
        while (!carryBits.IsZero)
		{
			before = vLow;
                    vLow.Add(carryBits);
			carryBits = vLow.CompareTo(before) < 0 ? GpuUInt128.One : GpuUInt128.Zero;
		}

        // Final correction if v >= modulus (mask)
        if (vLow.High > maskHigh || (vLow.High == maskHigh && vLow.Low >= maskLow))
        {
            ulong borrow = 0UL;
            ulong newLow = vLow.Low - maskLow;
            borrow = vLow.Low < maskLow ? 1UL : 0UL;
            ulong newHigh = vLow.High - maskHigh - borrow;
            vLow = new GpuUInt128(newHigh, newLow);
        }

        return vLow;
    }

    // Helper: 128-bit square -> 256-bit product as four 64-bit limbs (q3..q0)
    private static void SquareFull128(GpuUInt128 value, out ulong q3, out ulong q2, out ulong q1, out ulong q0)
    {
        ulong L = value.Low;
        ulong H = value.High;
        Mul64Parts(L, L, out var hLL, out var lLL);    // L^2
        Mul64Parts(H, H, out var hHH, out var lHH);    // H^2
        Mul64Parts(L, H, out var hLH, out var lLH);    // L*H

        // double LH
        ulong dLH_low = lLH << 1;
        ulong carry = (lLH >> 63) & 1UL;
        ulong dLH_high = (hLH << 1) | carry;

        q0 = lLL;
        ulong sum1 = hLL + dLH_low;
        ulong c1 = sum1 < hLL ? 1UL : 0UL;
        q1 = sum1;

        ulong sum2 = lHH + dLH_high;
        ulong c2 = sum2 < lHH ? 1UL : 0UL;
        sum2 += c1;
        if (sum2 < c1)
        {
            c2++;
        }
        q2 = sum2;
        q3 = hHH + c2;
    }

    // 64x64 -> 128 multiply (high,low)
    [MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static void Mul64Parts(ulong a, ulong b, out ulong high, out ulong low)
    {
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

    [MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static ulong ModPow64(ulong value, ulong exponent, ulong modulus)
    {
        ulong result = 1UL;
        ulong baseValue = value % modulus;
        ulong exp = exponent;

        while (exp != 0UL)
        {
            if ((exp & 1UL) != 0UL)
            {
                result = (ulong)(((UInt128)result * baseValue) % modulus);
            }

            baseValue = (ulong)(((UInt128)baseValue * baseValue) % modulus);
            exp >>= 1;
        }

        return result;
    }
}

