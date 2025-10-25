using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;
using static PerfectNumbers.Core.Gpu.GpuContextPool;

namespace PerfectNumbers.Core.Gpu;

public sealed class GpuKernelLease
{
    private static readonly ConcurrentQueue<GpuKernelLease> Pool = new();

    private IDisposable? _limiter;
    private GpuContextLease? _gpu;
    private KernelContainer? _kernels;
    private AcceleratorStream? _stream;
    private object? _executionLock;
    private bool _disposed;

    private GpuKernelLease()
    {
    }

    internal static GpuKernelLease Rent(IDisposable limiter, GpuContextLease gpu, KernelContainer kernels)
    {
        if (!Pool.TryDequeue(out var lease))
        {
            lease = new GpuKernelLease();
        }

        lease._limiter = limiter;
        lease._gpu = gpu;
        lease._kernels = kernels;
        lease._stream = null;
        lease._executionLock = gpu.ExecutionLock;
        lease._disposed = false;
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

    public ExecutionScope EnterExecutionScope() => new ExecutionScope(_executionLock);

    public Action<AcceleratorStream, Index1D, ulong, ulong, ArrayView<GpuUInt128>, ArrayView<ulong>> OrderKernel
    {
        get
        {
            var accel = Accelerator; // avoid capturing 'this' in lambda
            lock (_gpu!.Value.KernelInitLock)
            {
                return KernelContainer.InitOnce(ref _kernels!.Order, () =>
                {
                    var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ulong, ulong, ArrayView<GpuUInt128>, ArrayView<ulong>>(OrderKernels.OrderKernelScan);
                    var kernel = KernelUtil.GetKernel(loaded);
                    return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, ulong, ArrayView<GpuUInt128>, ArrayView<ulong>>>();
                });
            }
        }
    }

    public Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>> Pow2ModKernel
    {
        get
        {
            var accel = Accelerator;
            lock (_gpu!.Value.KernelInitLock)
            {
                return KernelContainer.InitOnce(ref _kernels!.Pow2Mod, () =>
                {
                    var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>(Pow2ModKernels.Pow2ModKernelScan);
                    var kernel = KernelUtil.GetKernel(loaded);
                    return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>>();
                });
            }
        }
    }

    public Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ulong, ulong, ulong, ulong, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>> IncrementalKernel
    {
        get
        {
            var accel = Accelerator;
            lock (_gpu!.Value.KernelInitLock)
            {
                return KernelContainer.InitOnce(ref _kernels!.Incremental, () =>
                {
                    var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ulong, ulong, ulong, ulong, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>>(IncrementalKernels.IncrementalKernelScan);
                    var kernel = KernelUtil.GetKernel(loaded);
                    return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ulong, ulong, ulong, ulong, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>>>();
                });
            }
        }
    }

    public Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>> IncrementalOrderKernel
    {
        get
        {
            var accel = Accelerator;
            lock (_gpu!.Value.KernelInitLock)
            {
                return KernelContainer.InitOnce(ref _kernels!.IncrementalOrder, () =>
                {
                    var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>>(IncrementalKernels.IncrementalOrderKernelScan);
                    var kernel = KernelUtil.GetKernel(loaded);
                    return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>>>();
                });
            }
        }
    }

    public Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>> Pow2ModOrderKernel
    {
        get
        {
            var accel = Accelerator;
            lock (_gpu!.Value.KernelInitLock)
            {
                return KernelContainer.InitOnce(ref _kernels!.Pow2ModOrder, () =>
                {
                    var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>>(Pow2ModKernels.Pow2ModOrderKernelScan);
                    var kernel = KernelUtil.GetKernel(loaded);
                    return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>>>();
                });
            }
        }
    }

    public Action<AcceleratorStream, Index1D, ulong, uint, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>> SmallPrimeFactorKernel
    {
        get
        {
            var accel = Accelerator;
            lock (_gpu!.Value.KernelInitLock)
            {
                return KernelContainer.InitOnce(ref _kernels!.SmallPrimeFactor, () =>
                {
                    var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ulong, uint, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>(SmallPrimeFactorKernels.SmallPrimeFactorKernelScan);
                    var kernel = KernelUtil.GetKernel(loaded);
                    return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, uint, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>>();
                });
            }
        }
    }

    private void Dispose(bool disposing)
    {
        if (_disposed)
        {
            return;
        }

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

        _kernels = null;
        _disposed = true;
        Pool.Enqueue(this);
    }

    public void Dispose()
    {
        Dispose(disposing: false);
    }

    public readonly struct ExecutionScope
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
