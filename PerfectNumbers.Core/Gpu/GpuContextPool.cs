using System;
using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public static class GpuContextPool
{
        private static readonly bool PoolingEnabled = true;
        // The pool intentionally skips pre-loading ProcessEightBitWindows kernels so accelerator initialization
        // stays stable; GpuKernelPool JIT-compiles them on first use alongside the small-cycle uploads.
        // Default device preference for generic GPU kernels (prime scans, NTT, etc.)
        public static bool ForceCpu { get; set; } = false;

        internal sealed class PooledContext
        {
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

			// Ensure all cached GPU buffers for this accelerator are released.
			NttGpuMath.ClearCaches(Accelerator);
			// Release PrimeTester GPU state for this accelerator (kernel/device primes)
			// Clear any per-accelerator cached resources to avoid releasing
			// after the accelerator is destroyed.
			PrimeTester.ClearGpuCaches(Accelerator);

			Accelerator.Dispose();
			Context.Dispose();
		}
	}

	private static readonly ConcurrentQueue<PooledContext> CpuPool = new();
	private static readonly ConcurrentQueue<PooledContext> GpuPool = new();
        private static readonly object WarmUpLock = new();
        private static int WarmedGpuContextCount;

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
				return new GpuContextLease(cpu);
			}

			if (!preferCpu && GpuPool.TryDequeue(out var gpu))
			{
				return new GpuContextLease(gpu);
			}
		}

		// Create a new accelerator when the pool does not have one available.
		return new GpuContextLease(new PooledContext(preferCpu));
	}

        public static void WarmUpPool(int threadCount)
        {
                if (!PoolingEnabled || ForceCpu)
                {
                        return;
                }

                if (threadCount <= 0)
                {
                        return;
                }

                int target = threadCount / 4;
                if (target == 0)
                {
                        target = 1;
                }

                lock (WarmUpLock)
                {
                        if (target <= WarmedGpuContextCount)
                        {
                                return;
                        }

                        int toCreate = target - WarmedGpuContextCount;
                        var contexts = new PooledContext[toCreate];
                        for (int i = 0; i < toCreate; i++)
                        {
                                contexts[i] = new PooledContext(preferCpu: false);
                        }

                        for (int i = 0; i < toCreate; i++)
                        {
                                GpuPool.Enqueue(contexts[i]);
                        }

                        WarmedGpuContextCount = target;
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

		PrimeTester.DisposeGpuContexts();
                WarmedGpuContextCount = 0;
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
                        ctx.Dispose();
                }
        }

    public struct GpuContextLease
    {
        private readonly PooledContext _ctx;

                internal GpuContextLease(PooledContext ctx)
        {
            _ctx = ctx;
                }

        public Context Context => _ctx.Context;

        public Accelerator Accelerator => _ctx.Accelerator;

                public object ExecutionLock => _ctx.ExecutionLock;


                public void Dispose()
                {

			Return(_ctx);
        }
    }
}

