using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public static class GpuContextPool
{
        private static readonly bool PoolingEnabled = true;
        // TODO: Introduce accelerator-specific warmup so pooled contexts precompile the ProcessEightBitWindows kernels and load
        // divisor-cycle data before the first scan begins.
        // Default device preference for generic GPU kernels (prime scans, NTT, etc.)
        public static bool ForceCpu { get; set; } = false;

        internal sealed class PooledContext
        {
                private bool disposed;

                public Context Context { get; }
                public Accelerator Accelerator { get; }
                public bool IsCpu { get; }
                public object ExecutionLock { get; } = new();

		public PooledContext(bool preferCpu)
		{
			Context = Context.CreateDefault();
			Accelerator = Context.GetPreferredDevice(preferCpu).CreateAccelerator(Context);
			IsCpu = Accelerator.AcceleratorType == AcceleratorType.CPU;
			// NOTE: Avoid loading/compiling any kernel here to prevent implicit
			// CL stream/queue creation during accelerator construction.
			// Some OpenCL drivers are fragile when a queue is created immediately
			// after device init. Real kernels will be JIT-loaded at first use.
		}

		public void Dispose()
		{
			if (disposed)
			{
				return;
			}

			// Ensure all cached GPU buffers for this accelerator are released.
			NttGpuMath.ClearCaches(Accelerator);
			// Release PrimeTester GPU state for this accelerator (kernel/device primes)
			// Clear any per-accelerator cached resources to avoid releasing
			// after the accelerator is destroyed.
			PrimeTester.ClearGpuCaches(Accelerator);

			Accelerator.Dispose();
			Context.Dispose();
			disposed = true;
		}
	}

	private static readonly ConcurrentQueue<PooledContext> CpuPool = new();
	private static readonly ConcurrentQueue<PooledContext> GpuPool = new();
	private static readonly object CreationLock = new();

	public static GpuContextLease Rent()
	{
		return RentPreferred(ForceCpu);
	}

	// Allows callers to choose CPU/GPU per use-case, decoupled from ForceCpu.
	public static GpuContextLease RentPreferred(bool preferCpu)
	{
		if (PoolingEnabled)
		{
			if (preferCpu && CpuPool.TryDequeue(out var cpu))
			{
				return new GpuContextLease(cpu, CreationLock);
			}

			if (!preferCpu && GpuPool.TryDequeue(out var gpu))
			{
				return new GpuContextLease(gpu, CreationLock);
			}
		}

		// Serialize accelerator creation to avoid concurrent CL device/stream setup across threads.
		lock (CreationLock)
		{
			return new GpuContextLease(new PooledContext(preferCpu), CreationLock);
		}
	}

	public static void DisposeAll()
	{
		if (!PoolingEnabled)
		{
			return;
		}

		while (CpuPool.TryDequeue(out var cpu))
		{
			cpu.Dispose();
		}

		while (GpuPool.TryDequeue(out var gpu))
		{
			gpu.Dispose();
		}
	}

	private static void Return(PooledContext ctx)
	{
		if (PoolingEnabled)
		{
			ctx.Accelerator.Synchronize();
			if (ctx.IsCpu)
			{
				CpuPool.Enqueue(ctx);
			}
			else
			{
				GpuPool.Enqueue(ctx);
			}
		}

                else
                {
                        lock (CreationLock)
                        {
                                ctx.Dispose();
                        }
                }
        }

    public struct GpuContextLease : IDisposable
    {
        private readonly PooledContext _ctx;
                private bool disposed;

                internal GpuContextLease(PooledContext ctx, object kernelInitLock)
        {
            _ctx = ctx;
                        KernelInitLock = kernelInitLock;
                }

        public Context Context => _ctx.Context;

        public Accelerator Accelerator => _ctx.Accelerator;

                public object KernelInitLock { get; }

                public object ExecutionLock => _ctx.ExecutionLock;


                public void Dispose()
                {
                        if (disposed)
			{
				return;
			}

			Return(_ctx);
			disposed = true;
        }
    }
}

