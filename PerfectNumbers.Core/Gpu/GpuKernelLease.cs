using System;
using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core;
using static PerfectNumbers.Core.Gpu.GpuContextPool;

namespace PerfectNumbers.Core.Gpu;

public sealed class GpuKernelLease
{
    private static readonly ConcurrentQueue<GpuKernelLease> Pool = new();

    private GpuContextLease? _gpu;
    public KernelContainer Kernels = null!;
    private AcceleratorStream? _stream;

    private GpuKernelLease()
	{
	}

    internal static GpuKernelLease Rent(GpuContextLease gpu)
    {
        if (!Pool.TryDequeue(out var lease))
        {
            lease = new GpuKernelLease();
        }

        lease._gpu = gpu;
        lease.Kernels = gpu.Kernels;
        lease._stream = null;
        return lease;
    }

    public Accelerator Accelerator => _gpu?.Accelerator ?? throw new NullReferenceException("GPU context is already released");

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

    public Action<AcceleratorStream, Index1D, ulong, ulong, ArrayView<GpuUInt128>, ArrayView<ulong>> OrderKernel
    {
        get
        {
            var accel = Accelerator; // avoid capturing 'this' in lambda
            return KernelContainer.InitOnce(ref Kernels!.Order, () =>
            {
                var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ulong, ulong, ArrayView<GpuUInt128>, ArrayView<ulong>>(OrderKernels.OrderKernelScan);
                var kernel = KernelUtil.GetKernel(loaded);
                return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, ulong, ArrayView<GpuUInt128>, ArrayView<ulong>>>();
            });
        }
    }

    public Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>> Pow2ModKernel
    {
        get
        {
            var accel = Accelerator;
            return KernelContainer.InitOnce(ref Kernels!.Pow2Mod, () =>
            {
                var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>(Pow2ModKernels.Pow2ModKernelScan);
                var kernel = KernelUtil.GetKernel(loaded);
                return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>>();
            });
        }
    }

    public Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ulong, ulong, ulong, ulong, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>> IncrementalKernel
    {
        get
        {
            var accel = Accelerator;
            return KernelContainer.InitOnce(ref Kernels!.Incremental, () =>
            {
                var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ulong, ulong, ulong, ulong, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>>(IncrementalKernels.IncrementalKernelScan);
                var kernel = KernelUtil.GetKernel(loaded);
                return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ulong, ulong, ulong, ulong, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>>>();
            });
        }
    }

    public Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>> IncrementalOrderKernel
    {
        get
        {
            var accel = Accelerator;
            return KernelContainer.InitOnce(ref Kernels!.IncrementalOrder, () =>
            {
                var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>>(IncrementalKernels.IncrementalOrderKernelScan);
                var kernel = KernelUtil.GetKernel(loaded);
                return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>>>();
            });
        }
    }

    public Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>> Pow2ModOrderKernel
    {
        get
        {
            var accel = Accelerator;
            return KernelContainer.InitOnce(ref Kernels!.Pow2ModOrder, () =>
            {
                var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>>(Pow2ModKernels.Pow2ModOrderKernelScan);
                var kernel = KernelUtil.GetKernel(loaded);
                return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>>>();
            });
        }
    }

    public Action<AcceleratorStream, Index1D, ulong, ArrayView1D<ulong, Stride1D.Dense>, int, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ushort, Stride1D.Dense>> SpecialMaxKernel
    {
        get
        {
            var accel = Accelerator;
            return KernelContainer.InitOnce(ref Kernels!.SpecialMax, () =>
            {
                var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ulong, ArrayView1D<ulong, Stride1D.Dense>, int, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ushort, Stride1D.Dense>>(PrimeOrderGpuHeuristics.EvaluateSpecialMaxCandidatesKernel);
                var kernel = KernelUtil.GetKernel(loaded);
                return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, ArrayView1D<ulong, Stride1D.Dense>, int, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ushort, Stride1D.Dense>>>();
            });
        }
    }

    public Action<AcceleratorStream, Index1D, ulong, uint, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>> SmallPrimeFactorKernel
    {
        get
        {
            var accel = Accelerator;
            return KernelContainer.InitOnce(ref Kernels!.SmallPrimeFactor, () =>
            {
                var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ulong, uint, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>(SmallPrimeFactorKernels.SmallPrimeFactorKernelScan);
                var kernel = KernelUtil.GetKernel(loaded);
                return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, uint, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>>();
            });
        }
    }

    public void Dispose()
    {
        _stream?.Dispose();
        _stream = null;

        _gpu?.Dispose();
        _gpu = null;
        Kernels = null!;

        GpuPrimeWorkLimiter.Release();
        Pool.Enqueue(this);
    }

}
