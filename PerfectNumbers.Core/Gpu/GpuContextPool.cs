using System;
using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public static class GpuContextPool
{
	// The pool intentionally skips pre-loading ProcessEightBitWindows kernels so accelerator initialization
	// stays stable; GpuKernelPool JIT-compiles them on first use alongside the small-cycle uploads.
	// Default device preference for generic GPU kernels (prime scans, NTT, etc.)
	internal sealed class PooledContext
	{
		public readonly Context Context;
		public readonly Accelerator Accelerator;

		private static readonly Context _sharedContext = Context.CreateDefault();
		private static readonly Device _sharedDevice = _sharedContext.GetPreferredDevice(false);
		private static readonly Accelerator _sharedAccelerator = _sharedDevice.CreateAccelerator(_sharedContext);

		public PooledContext()
		{
			Context = _sharedContext;
			Accelerator = _sharedAccelerator;
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

			// These resources are shared between GPU leases
			//  Accelerator.Dispose();
			// 	Context.Dispose();
		}
	}

	private static readonly ConcurrentQueue<PooledContext> GpuPool = new();
	private static int WarmedGpuContextCount;

	// Allows callers to choose CPU/GPU per use-case, decoupled from ForceCpu.
        	public static GpuContextLease Rent()
	{
		if (GpuPool.TryDequeue(out var gpu))
		{
			return new GpuContextLease(gpu);
		}

		// Create a new accelerator when the pool does not have one available.
		return new GpuContextLease(new PooledContext());
	}


	public static void WarmUpPool(int threadCount)
	{
		if (threadCount <= WarmedGpuContextCount)
		{
			return;
		}

		int toCreate = threadCount - WarmedGpuContextCount;
		Console.WriteLine($"Warming up GPU context pool to {toCreate} contexts...");
		var contexts = new PooledContext[toCreate];
		var pool = GpuPool;
		PooledContext context;
		for (int i = 0; i < toCreate; i++)
		{
			context = new PooledContext();
			contexts[i] = context;
			pool.Enqueue(context);
		}

		WarmedGpuContextCount = threadCount;
	}

	public static void DisposeAll()
	{
		while (GpuPool.TryDequeue(out var gpu))
		{
			gpu.Dispose();
		}

		PrimeTester.DisposeGpuContexts();
		WarmedGpuContextCount = 0;
	}

	private static void Return(PooledContext ctx)
	{
		ctx.Accelerator.Synchronize();
		GpuPool.Enqueue(ctx);
	}

	public struct GpuContextLease
	{
		private PooledContext? _ctx;
		private bool _disposed;

		internal GpuContextLease(PooledContext ctx)
		{
			_ctx = ctx;
			_disposed = false;
		}

		public Context Context => _ctx!.Context;
		public Accelerator Accelerator => _ctx!.Accelerator;

		public void Dispose()
		{
			if (_disposed)
			{
				return;
			}

			_disposed = true;
			Return(_ctx!);
			_ctx = null;
		}
	}
}

