using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using static PerfectNumbers.Core.Gpu.GpuContextPool;

namespace PerfectNumbers.Core.Gpu;

public readonly struct ResidueAutomatonArgs
{
    public readonly ulong Q0M10;
    public readonly ulong Step10;
    public readonly ulong Q0M8;
    public readonly ulong Step8;
    public readonly ulong Q0M3;
    public readonly ulong Step3;
    public readonly ulong Q0M5;
    public readonly ulong Step5;

    public ResidueAutomatonArgs(ulong q0m10, ulong step10, ulong q0m8, ulong step8, ulong q0m3, ulong step3, ulong q0m5, ulong step5)
    {
        Q0M10 = q0m10; Step10 = step10; Q0M8 = q0m8; Step8 = step8; Q0M3 = q0m3; Step3 = step3; Q0M5 = q0m5; Step5 = step5;
    }
}

public sealed class KernelContainer
{
    // Serializes first-time initialization of kernels/buffers per accelerator.
    public Action<AcceleratorStream, Index1D, ulong, ulong, ArrayView<GpuUInt128>, ArrayView<ulong>>? Order;
    public Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong,
    ulong, ulong, ulong, ulong, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>>? Incremental;
    public Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong,
        ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>,
        ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>,
        ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>? Pow2Mod;
    public Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong,
            ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>>? IncrementalOrder;
    public Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong,
        ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>>? Pow2ModOrder;

    // Optional device buffer with small divisor cycles (<= 4M). Index = divisor, value = cycle length.
    public MemoryBuffer1D<ulong, Stride1D.Dense>? SmallCycles;
    public MemoryBuffer1D<uint, Stride1D.Dense>? SmallPrimesLastOne;
    public MemoryBuffer1D<ulong, Stride1D.Dense>? SmallPrimesPow2LastOne;
    public MemoryBuffer1D<uint, Stride1D.Dense>? SmallPrimesLastSeven;
    public MemoryBuffer1D<ulong, Stride1D.Dense>? SmallPrimesPow2LastSeven;

    public static T InitOnce<T>(ref T? slot, Func<T> factory) where T : class
    {
        var current = Volatile.Read(ref slot);
        if (current is not null)
        {
            return current;
        }

        current = Volatile.Read(ref slot);
        if (current is null)
        {
            current = factory();
            Volatile.Write(ref slot, current);
        }
        return current;
    }
}

public readonly struct ResiduePrimeViews(
    ArrayView1D<uint, Stride1D.Dense> lastOne,
    ArrayView1D<uint, Stride1D.Dense> lastSeven,
    ArrayView1D<ulong, Stride1D.Dense> lastOnePow2,
    ArrayView1D<ulong, Stride1D.Dense> lastSevenPow2)
{
    public readonly ArrayView1D<uint, Stride1D.Dense> LastOne = lastOne;
    public readonly ArrayView1D<uint, Stride1D.Dense> LastSeven = lastSeven;
    public readonly ArrayView1D<ulong, Stride1D.Dense> LastOnePow2 = lastOnePow2;
    public readonly ArrayView1D<ulong, Stride1D.Dense> LastSevenPow2 = lastSevenPow2;
}

public ref struct GpuKernelLease(IDisposable limiter, GpuContextLease gpu, KernelContainer kernels)
{
    private bool disposedValue;
    private IDisposable? _limiter = limiter;
    private GpuContextLease? _gpu = gpu;
    private KernelContainer _kernels = kernels;
    private AcceleratorStream? _stream;
    private object? _executionLock = gpu.ExecutionLock;

    public readonly Accelerator Accelerator => _gpu?.Accelerator ?? throw new NullReferenceException("GPU context is already released");

    public AcceleratorStream Stream
    {
        get
        {
            if (_stream is { } stream)
            {
                return stream;
            }

            // It feels that we can combine these into 1 return statement, but that increases the max stack frame size to 3 from 2
            stream = Accelerator.CreateStream();
            _stream = stream;
            return stream;
        }
    }

    public readonly ExecutionScope EnterExecutionScope() => new ExecutionScope(_executionLock);

    public readonly Action<AcceleratorStream, Index1D, ulong, ulong, ArrayView<GpuUInt128>, ArrayView<ulong>> OrderKernel
    {
        get
        {
            var accel = Accelerator; // avoid capturing 'this' in lambda
            lock (_gpu!.Value.KernelInitLock)
            {
                return KernelContainer.InitOnce(ref _kernels.Order, () =>
                {
                    var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ulong, ulong, ArrayView<GpuUInt128>, ArrayView<ulong>>(OrderKernelScan);
                    var kernel = KernelUtil.GetKernel(loaded);
                    return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, ulong, ArrayView<GpuUInt128>, ArrayView<ulong>>>();
                });
            }
        }
    }

    public readonly Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>> Pow2ModKernel
    {
        get
        {
            var accel = Accelerator;
            lock (_gpu!.Value.KernelInitLock)
            {
                return KernelContainer.InitOnce(ref _kernels.Pow2Mod, () =>
                {
                    var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>(Pow2ModKernelScan);
                    var kernel = KernelUtil.GetKernel(loaded);
                    return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>>();
                });
            }
        }
    }

    public readonly Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ulong, ulong, ulong, ulong, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>> IncrementalKernel
    {
        get
        {
            var accel = Accelerator;
            lock (_gpu!.Value.KernelInitLock)
            {
                return KernelContainer.InitOnce(ref _kernels.Incremental, () =>
                {
                    var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ulong, ulong, ulong, ulong, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>>(IncrementalKernelScan);
                    var kernel = KernelUtil.GetKernel(loaded);
                    return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ulong, ulong, ulong, ulong, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>>>();
                });
            }
        }
    }

    public readonly Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>> IncrementalOrderKernel
    {
        get
        {
            var accel = Accelerator;
            lock (_gpu!.Value.KernelInitLock)
            {
                return KernelContainer.InitOnce(ref _kernels.IncrementalOrder, () =>
                {
                    var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>>(IncrementalOrderKernelScan);
                    var kernel = KernelUtil.GetKernel(loaded);
                    return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>>>();
                });
            }
        }
    }

    public readonly Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>> Pow2ModOrderKernel
    {
        get
        {
            var accel = Accelerator;
            lock (_gpu!.Value.KernelInitLock)
            {
                return KernelContainer.InitOnce(ref _kernels.Pow2ModOrder, () =>
                {
                    var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>>(Pow2ModOrderKernelScan);
                    var kernel = KernelUtil.GetKernel(loaded);
                    return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>>>();
                });
            }
        }
    }

    // TODO: Plumb the small-cycles device buffer into all kernels that can benefit
    // (some already accept it). Consider a compact type (byte/ushort) for memory footprint.

    private static void IncrementalKernelScan(Index1D index, ulong exponent, GpuUInt128 twoP, GpuUInt128 kStart, byte lastIsSeven, ulong divMul,
        ulong q0m10, ulong q0m8, ulong q0m3, ulong q0m5, ArrayView<ulong> orders,
        ArrayView1D<ulong, Stride1D.Dense> smallCycles)
    {
        ulong idx = (ulong)index.X;
        // TODO: Replace these `%` computations with the precomputed Mod3/Mod5 tables so GPU kernels reuse cached residues
        // instead of performing modulo operations that the benchmarks showed slower on wide sweeps.
        ulong idxMod3 = idx % 3UL;
        ulong idxMod5 = idx % 5UL;
        // residue automaton
        ulong step10 = (exponent.Mod10() << 1).Mod10();
        ulong r10 = q0m10 + (step10 * idx).Mod10();
        r10 -= (r10 >= 10UL) ? 10UL : 0UL;
        bool shouldCheck = r10 != 5UL; // skip q ≡ 5 (mod 10); other residues handled by r8/r3/r5
        if (shouldCheck)
        {
            ulong step8 = ((exponent & 7UL) << 1) & 7UL;
            ulong r8 = (q0m8 + ((step8 * idx) & 7UL)) & 7UL;
            if (r8 != 1UL && r8 != 7UL)
            {
                shouldCheck = false;
            }
            else
            {
                // TODO: Swap these `%` filters to the shared Mod helpers so the residue automaton matches the benchmarked
                // bitmask-based implementation for GPU workloads.
                ulong step3 = ((exponent % 3UL) << 1) % 3UL;
                ulong r3 = q0m3 + (step3 * idxMod3); r3 -= (r3 >= 3UL) ? 3UL : 0UL;
                if (r3 == 0UL)
                {
                    shouldCheck = false;
                }
                else
                {
                    ulong step5Loc = ((exponent % 5UL) << 1) % 5UL;
                    ulong r5 = q0m5 + (step5Loc * idxMod5); if (r5 >= 5UL) r5 -= 5UL;
                    if (r5 == 0UL)
                    {
                        shouldCheck = false;
                    }
                }
            }
        }

        if (!shouldCheck)
        {
            orders[index] = 0UL;
            return;
        }

        // TODO: Is this expected that the value of kStart is modified?
        GpuUInt128 k = kStart + (GpuUInt128)idx;
        GpuUInt128 q = twoP;
        q.Mul64(k);
        q.Add(GpuUInt128.One);
        ReadOnlyGpuUInt128 readOnlyQ = q.AsReadOnly();

        // Small-cycles in-kernel early rejection from device table
        if (q.High == 0UL && q.Low < (ulong)smallCycles.Length)
        {
            ulong cycle = smallCycles[(int)q.Low];
            if (cycle != 0UL && cycle <= exponent && (exponent % cycle) != 0UL)
            {
                orders[index] = 0UL;
                return;
            }
        }

        // TODO: q will be lowered by GpuUInt128.One here. Is this expected? Maybe that's why we have issues with the incremental kernel? Review other places where we use GpuUInt128 operators.
        GpuUInt128 phi = q - GpuUInt128.One;
        // TODO: Do we have the check here because we don't support high values or this is expected result?
        if (phi.High != 0UL)
        {
            orders[index] = 0UL;
            return;
        }

        ulong phi64 = phi.Low;
        // TODO: Replace these Pow2Mod calls with the ProcessEightBitWindows helper when the shared windowed
        // scalar implementation lands; benchmarks showed the windowed kernel trimming per-divisor runtime by ~2.4×.
        if (GpuUInt128.Pow2Mod(phi64, in readOnlyQ) != GpuUInt128.One)
        {
            orders[index] = 0UL;
            return;
        }

        GpuUInt128 halfPow = GpuUInt128.Pow2Mod(phi64 >> 1, in readOnlyQ) - GpuUInt128.One;
        if (GpuUInt128.BinaryGcd(halfPow, q) != GpuUInt128.One)
        {
            orders[index] = 0UL;
            return;
        }

        ulong div = FastDiv64Gpu(phi64, exponent, divMul);
        GpuUInt128 divPow = GpuUInt128.Pow2Mod(div, in readOnlyQ) - GpuUInt128.One;
        if (GpuUInt128.BinaryGcd(divPow, q) != GpuUInt128.One)
        {
            orders[index] = 0UL;
            return;
        }

        orders[index] = 1UL;
    }

    private static void OrderKernelScan(Index1D index, ulong exponent, ulong divMul, ArrayView<GpuUInt128> qs, ArrayView<ulong> orders)
    {
        GpuUInt128 q = qs[index];
        ReadOnlyGpuUInt128 readOnlyQ = q.AsReadOnly();
        GpuUInt128 phi = q - GpuUInt128.One;
        if (phi.High != 0UL)
        {
            orders[index] = 0UL;
            return;
        }

        ulong phi64 = phi.Low;
        // TODO: Once the ProcessEightBitWindows helper is available, switch this order kernel to that faster
        // Pow2Mod variant so cycle checks inherit the same gains observed in GpuPow2ModBenchmarks.
        GpuUInt128 pow = GpuUInt128.Pow2Mod(phi64, in readOnlyQ);
        if (pow != GpuUInt128.One)
        {
            orders[index] = 0UL;
            return;
        }

        GpuUInt128 halfPow = GpuUInt128.Pow2Mod(phi64 >> 1, in readOnlyQ) - GpuUInt128.One;
        if (GpuUInt128.BinaryGcd(halfPow, q) != GpuUInt128.One)
        {
            orders[index] = 0UL;
            return;
        }

        ulong div = FastDiv64Gpu(phi64, exponent, divMul);
        GpuUInt128 divPow = GpuUInt128.Pow2Mod(div, in readOnlyQ) - GpuUInt128.One;
        if (GpuUInt128.BinaryGcd(divPow, q) != GpuUInt128.One)
        {
            orders[index] = 0UL;
            return;
        }

        orders[index] = exponent;
    }

    private void Dispose(bool disposing)
    {
        if (!disposedValue)
        {
            if (disposing)
            {
                _stream?.Dispose();
                _stream = null;

                _gpu?.Dispose();
                _gpu = null;

                _executionLock = null;

                _limiter?.Dispose();
                _limiter = null;
            }

            // free unmanaged resources (unmanaged objects) and override finalizer
            // set large fields to null
            disposedValue = true;
        }
    }

    private static void Pow2ModKernelScan(Index1D index, ulong exponent, GpuUInt128 twoP, GpuUInt128 kStart, byte lastIsSeven, ulong _,
        ResidueAutomatonArgs ra, ArrayView<ulong> orders,
        ArrayView1D<ulong, Stride1D.Dense> smallCycles,
        ArrayView1D<uint, Stride1D.Dense> smallPrimesLastOne,
        ArrayView1D<uint, Stride1D.Dense> smallPrimesLastSeven,
        ArrayView1D<ulong, Stride1D.Dense> smallPrimesPow2LastOne,
        ArrayView1D<ulong, Stride1D.Dense> smallPrimesPow2LastSeven)
    {
        ulong idx = (ulong)index.X;
        ulong idxMod3 = idx % 3UL;
        ulong idxMod5 = idx % 5UL;
        ulong r10 = ra.Q0M10 + (ra.Step10 * idx).Mod10(); r10 -= (r10 >= 10UL) ? 10UL : 0UL;
        bool shouldCheck = r10 != 5UL;
        if (shouldCheck)
        {
            ulong r8 = (ra.Q0M8 + ((ra.Step8 * idx) & 7UL)) & 7UL;
            if (r8 != 1UL && r8 != 7UL)
            {
                shouldCheck = false;
            }
            else
            {
                ulong r3 = ra.Q0M3 + (ra.Step3 * idxMod3);
                if (r3 >= 6UL) r3 -= 6UL;
                if (r3 >= 3UL) r3 -= 3UL;
                if (r3 == 0UL)
                {
                    shouldCheck = false;
                }
                else
                {
                    ulong r5 = ra.Q0M5 + (ra.Step5 * idxMod5);
                    if (r5 >= 15UL) r5 -= 15UL;
                    if (r5 >= 10UL) r5 -= 10UL;
                    if (r5 >= 5UL) r5 -= 5UL;
                    if (r5 == 0UL)
                    {
                        shouldCheck = false;
                    }
                }
            }
        }
        if (!shouldCheck)
        {
            orders[index] = 0UL;
            return;
        }

        // TODO: kStart is modified after this. Is this expected?
        kStart.Add(idx);
        GpuUInt128 q = twoP;
        q.Mul64(kStart);
        q.Add(GpuUInt128.One);
        ReadOnlyGpuUInt128 readOnlyQ = q.AsReadOnly();
        if (q.High == 0UL && q.Low < (ulong)smallCycles.Length)
        {
            ulong cycle = smallCycles[(int)q.Low];
            // cycle should always be initialized if we're within array limit in production code
            if (cycle != 0UL && cycle <= exponent && (exponent % cycle) != 0UL)
            {
                orders[index] = 0UL;
                return;
            }
        }
        // TODO: Swap Pow2Minus1Mod for the eight-bit window helper once the scalar version switches;
        // benchmarks show the windowed variant cuts large-divisor scans from ~51 µs to ~21 µs.
        if (GpuUInt128.Pow2Minus1Mod(exponent, in readOnlyQ) != GpuUInt128.Zero)
        {
            orders[index] = 0UL;
            return;
        }

        ArrayView1D<uint, Stride1D.Dense> primes = lastIsSeven != 0 ? smallPrimesLastSeven : smallPrimesLastOne;
        ArrayView1D<ulong, Stride1D.Dense> primesPow2 = lastIsSeven != 0 ? smallPrimesPow2LastSeven : smallPrimesPow2LastOne;
        int primesLen = (int)primes.Length;
        for (int i = 0; i < primesLen; i++)
        {
            ulong square = primesPow2[i];
            if (new GpuUInt128(0UL, square) > q)
            {
                break;
            }
            ulong prime = primes[i];
            if (Mod128By64(q, prime) == 0UL)
            {
                orders[index] = 0UL;
                return;
            }
        }

        orders[index] = exponent;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong FastDiv64Gpu(ulong value, ulong divisor, ulong mul)
    {
        ulong quotient = GpuUInt128.MulHigh(value, mul);
        GpuUInt128 remainder = new(0UL, value);
        GpuUInt128 product = new(GpuUInt128.MulHigh(quotient, divisor), quotient * divisor);
        remainder.Sub(product);

        if (remainder.High != 0UL || remainder.Low >= divisor)
        {
            quotient++;
        }

        return quotient;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Mod128By64(GpuUInt128 value, ulong modulus)
    {
        ulong rem = 0UL;
        ulong part = value.High;
        for (int i = 0; i < 64; i++)
        {
            rem = (rem << 1) | (part >> 63);
            if (rem >= modulus)
            {
                rem -= modulus;
            }
            part <<= 1;
        }

        part = value.Low;
        for (int i = 0; i < 64; i++)
        {
            rem = (rem << 1) | (part >> 63);
            if (rem >= modulus)
            {
                rem -= modulus;
            }
            part <<= 1;
        }

        return rem;
    }

    // Device-friendly small cycle length for 2 mod divisor
    private static ulong CalculateCycleLengthSmall(ulong divisor)
    {
        if ((divisor & (divisor - 1UL)) == 0UL)
            return 1UL;

        ulong order = 1UL, pow = 2UL;
        while (pow != 1UL)
        {
            pow <<= 1;
            if (pow > divisor)
                pow -= divisor;

            order++;
        }

        return order;
    }

    private static void IncrementalOrderKernelScan(Index1D index, ulong exponent, GpuUInt128 twoP, GpuUInt128 kStart, byte lastIsSeven, ulong divMul,
                ResidueAutomatonArgs ra, ArrayView<int> found, ArrayView1D<ulong, Stride1D.Dense> smallCycles)
    {
        ulong idx = (ulong)index.X;
        ulong idxMod3 = idx % 3UL;
        ulong idxMod5 = idx % 5UL;
        ulong r10 = ra.Q0M10 + (ra.Step10 * idx).Mod10(); r10 -= (r10 >= 10UL) ? 10UL : 0UL;
        bool shouldCheck = r10 != 5UL;
        if (shouldCheck)
        {
            ulong r8 = (ra.Q0M8 + ((ra.Step8 * idx) & 7UL)) & 7UL;
            if (r8 != 1UL && r8 != 7UL)
            {
                shouldCheck = false;
            }
            else
            {
                ulong r3 = ra.Q0M3 + (ra.Step3 * idxMod3);
                if (r3 >= 6UL) r3 -= 6UL;
                if (r3 >= 3UL) r3 -= 3UL;
                if (r3 == 0UL)
                {
                    shouldCheck = false;
                }
                else
                {
                    ulong r5 = ra.Q0M5 + (ra.Step5 * idxMod5);
                    if (r5 >= 15UL) r5 -= 15UL;
                    if (r5 >= 10UL) r5 -= 10UL;
                    if (r5 >= 5UL) r5 -= 5UL;
                    if (r5 == 0UL)
                    {
                        shouldCheck = false;
                    }
                }
            }
        }
        if (!shouldCheck)
        {
            return;
        }

        GpuUInt128 k = kStart + (GpuUInt128)idx;
        GpuUInt128 q = twoP;
        q.Mul64(k);
        q.Add(GpuUInt128.One);
        ReadOnlyGpuUInt128 readOnlyQ = q.AsReadOnly();

        // Small-cycles in-kernel early rejection from device table
        if (q.High == 0UL && q.Low < (ulong)smallCycles.Length)
        {
            ulong cycle = smallCycles[(int)q.Low];
            if (cycle != 0UL && cycle <= exponent && (exponent % cycle) != 0UL)
            {
                return;
            }
        }
        GpuUInt128 phi = q - GpuUInt128.One;
        if (phi.High != 0UL)
        {
            return;
        }

        ulong phi64 = phi.Low;
        // TODO: Upgrade this pow2mod order kernel to the ProcessEightBitWindows helper once available so GPU residue
        // scans avoid the single-bit ladder that benchmarks found to be 2.3× slower on large exponents.
        if (GpuUInt128.Pow2Mod(phi64, in readOnlyQ) != GpuUInt128.One)
        {
            return;
        }

        GpuUInt128 halfPow = GpuUInt128.Pow2Mod(phi64 >> 1, in readOnlyQ) - GpuUInt128.One;
        if (GpuUInt128.BinaryGcd(halfPow, q) != GpuUInt128.One)
        {
            return;
        }

        ulong div = FastDiv64Gpu(phi64, exponent, divMul);
        GpuUInt128 divPow = GpuUInt128.Pow2Mod(div, in readOnlyQ) - GpuUInt128.One;
        if (GpuUInt128.BinaryGcd(divPow, q) != GpuUInt128.One)
        {
            return;
        }

        Atomic.Or(ref found[0], 1);
    }

    private static void Pow2ModOrderKernelScan(Index1D index, ulong exponent, GpuUInt128 twoP, GpuUInt128 kStart, byte lastIsSeven, ulong _,
                ResidueAutomatonArgs ra, ArrayView<int> found, ArrayView1D<ulong, Stride1D.Dense> smallCycles)
    {
        ulong idx = (ulong)index.X;
        ulong idxMod3 = idx % 3UL;
        ulong idxMod5 = idx % 5UL;
        ulong r10 = ra.Q0M10 + (ra.Step10 * idx).Mod10(); r10 -= (r10 >= 10UL) ? 10UL : 0UL;
        bool shouldCheck = r10 != 5UL;
        if (shouldCheck)
        {
            ulong r8 = (ra.Q0M8 + ((ra.Step8 * idx) & 7UL)) & 7UL;
            if (r8 != 1UL && r8 != 7UL)
            {
                shouldCheck = false;
            }
            else
            {
                ulong r3 = ra.Q0M3 + (ra.Step3 * idxMod3);
                if (r3 >= 6UL) r3 -= 6UL;
                if (r3 >= 3UL) r3 -= 3UL;
                if (r3 == 0UL)
                {
                    shouldCheck = false;
                }
                else
                {
                    ulong r5 = ra.Q0M5 + (ra.Step5 * idxMod5);
                    if (r5 >= 15UL) r5 -= 15UL;
                    if (r5 >= 10UL) r5 -= 10UL;
                    if (r5 >= 5UL) r5 -= 5UL;
                    if (r5 == 0UL)
                    {
                        shouldCheck = false;
                    }
                }
            }
        }
        if (!shouldCheck)
        {
            return;
        }

        GpuUInt128 k = kStart + (GpuUInt128)idx;
        GpuUInt128 q = twoP;
        q.Mul64(k);
        q.Add(GpuUInt128.One);
        ReadOnlyGpuUInt128 readOnlyQ = q.AsReadOnly();
        // Small-cycles in-kernel early rejection from device table
        if (q.High == 0UL && q.Low < (ulong)smallCycles.Length)
        {
            ulong cycle = smallCycles[(int)q.Low];
            if (cycle != 0UL && cycle <= exponent && (exponent % cycle) != 0UL)
            {
                return;
            }
        }
        // TODO: Replace this Pow2Mod check with the ProcessEightBitWindows helper once Pow2Minus1Mod adopts it;
        // residue order scans will then benefit from the same 2× speedup the benchmarked windowed kernel delivered.
        if (GpuUInt128.Pow2Mod(exponent, in readOnlyQ) != GpuUInt128.One)
        {
            return;
        }

        Atomic.Or(ref found[0], 1);
    }

    public void Dispose()
    {
        // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        Dispose(disposing: true);
        // Enabled this when the struct is converted to a class
        // GC.SuppressFinalize(this);
    }

    public readonly struct ExecutionScope : IDisposable
    {
        private readonly object? _lock;

        public ExecutionScope(object? sync)
        {
            _lock = sync;
            if (_lock is not null)
            {
                Monitor.Enter(_lock);
            }
        }

        public void Dispose()
        {
            if (_lock is not null)
            {
                Monitor.Exit(_lock);
            }
        }
    }
}

public class GpuKernelPool
{
    private static readonly ConcurrentDictionary<Accelerator, KernelContainer> KernelCache = new(); // TODO: Replace this concurrent map with a simple accelerator-indexed lookup once kernel launchers are prewarmed during startup so we can drop the thread-safe wrapper entirely.

    private static KernelContainer GetKernels(Accelerator accelerator)
    {
        return KernelCache.GetOrAdd(accelerator, _ => new KernelContainer());
    }

    // Ensures the small cycles table is uploaded to the device for the given accelerator.
    // Returns the ArrayView to pass into kernels (when kernels are extended to accept it).
    public static ArrayView1D<ulong, Stride1D.Dense> EnsureSmallCyclesOnDevice(Accelerator accelerator)
    {
        var kernels = GetKernels(accelerator);
        if (kernels.SmallCycles is { } buffer)
        {
            return buffer.View;
        }

        // Ensure single upload per accelerator even if multiple threads race here.
        lock (kernels) // TODO: Remove this lock by pre-uploading the immutable small-cycle snapshot during initialization; once no mutation happens at runtime, the pool must expose a simple reference without synchronization.
        {
            if (kernels.SmallCycles is { } existing)
            {
                return existing.View;
            }

            var host = MersenneDivisorCycles.Shared.ExportSmallCyclesSnapshot(); // TODO: Preload this device buffer during startup and keep it immutable so we can delete the lock above in favor of the preloaded snapshot.
            var device = accelerator.Allocate1D<ulong>(host.Length);
            device.View.CopyFromCPU(host);
            kernels.SmallCycles = device;
            return device.View;
        }
    }

    public static ResiduePrimeViews EnsureSmallPrimesOnDevice(Accelerator accelerator)
    {
        var kernels = GetKernels(accelerator);
        if (kernels.SmallPrimesLastOne is { } lastOne &&
            kernels.SmallPrimesLastSeven is { } lastSeven &&
            kernels.SmallPrimesPow2LastOne is { } lastOnePow2 &&
            kernels.SmallPrimesPow2LastSeven is { } lastSevenPow2)
        {
            return new ResiduePrimeViews(lastOne.View, lastSeven.View, lastOnePow2.View, lastSevenPow2.View);
        }

        lock (kernels) // TODO: Inline these small-prime uploads into startup initialization alongside the small-cycle snapshot so we can drop runtime locking and keep the GPU pool free of synchronization.
        {
            if (kernels.SmallPrimesLastOne is { } existingLastOne &&
                kernels.SmallPrimesLastSeven is { } existingLastSeven &&
                kernels.SmallPrimesPow2LastOne is { } existingLastOnePow2 &&
                kernels.SmallPrimesPow2LastSeven is { } existingLastSevenPow2)
            {
                return new ResiduePrimeViews(existingLastOne.View, existingLastSeven.View, existingLastOnePow2.View, existingLastSevenPow2.View);
            }

            var hostLastOne = PrimesGenerator.SmallPrimesLastOne;
            var hostLastSeven = PrimesGenerator.SmallPrimesLastSeven;
            var hostLastOnePow2 = PrimesGenerator.SmallPrimesPow2LastOne;
            var hostLastSevenPow2 = PrimesGenerator.SmallPrimesPow2LastSeven;

            var deviceLastOne = accelerator.Allocate1D<uint>(hostLastOne.Length);
            deviceLastOne.View.CopyFromCPU(hostLastOne);
            var deviceLastSeven = accelerator.Allocate1D<uint>(hostLastSeven.Length);
            deviceLastSeven.View.CopyFromCPU(hostLastSeven);
            var deviceLastOnePow2 = accelerator.Allocate1D<ulong>(hostLastOnePow2.Length);
            deviceLastOnePow2.View.CopyFromCPU(hostLastOnePow2);
            var deviceLastSevenPow2 = accelerator.Allocate1D<ulong>(hostLastSevenPow2.Length);
            deviceLastSevenPow2.View.CopyFromCPU(hostLastSevenPow2);

            kernels.SmallPrimesLastOne = deviceLastOne;
            kernels.SmallPrimesLastSeven = deviceLastSeven;
            kernels.SmallPrimesPow2LastOne = deviceLastOnePow2;
            kernels.SmallPrimesPow2LastSeven = deviceLastSevenPow2;

            return new ResiduePrimeViews(deviceLastOne.View, deviceLastSeven.View, deviceLastOnePow2.View, deviceLastSevenPow2.View);
        }
    }

    public static GpuKernelLease GetKernel(bool useGpuOrder)
    {
        var limiter = GpuPrimeWorkLimiter.Acquire();
        var gpu = RentPreferred(preferCpu: !useGpuOrder);
        var accelerator = gpu.Accelerator;
        var kernels = GetKernels(accelerator);
        return new(limiter, gpu, kernels);
    }

    /// <summary>
    /// Runs a GPU action with an acquired accelerator and stream.
    /// </summary>
    /// <param name="action">Action to run with (Accelerator, Stream).</param>
    public static void Run(Action<Accelerator, AcceleratorStream> action)
    {
        var lease = GetKernel(useGpuOrder: true);
        var execution = lease.EnterExecutionScope();
        var accelerator = lease.Accelerator;
        var stream = accelerator.CreateStream();
        action(accelerator, stream);
        stream.Dispose();
        execution.Dispose();
        lease.Dispose();

    }
}
