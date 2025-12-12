using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core.Gpu;

public static class GpuWorkLimiter
{
	private static readonly TimeSpan _pauseBetweenChecks = TimeSpan.FromSeconds(30);
	private static PollingSemaphore _semaphore = new(int.MaxValue, _pauseBetweenChecks, 1);
	private static readonly object _lock = new();
	private static int _currentLimit = 1;

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static Lease Acquire()
	{
		// TODO: Replace the per-call Lease allocation with the pooled struct-based guard from the
		// GpuLimiterThroughputBenchmarks so limiter acquisition does not allocate when we enter the
		// GPU scanning hot path.
		var sem = _semaphore;
		sem.Wait();
		return new Lease(sem);
	}

	public static void SetLimit(int value)
	{
		if (value < 1)
		{
			value = 1;
		}

		lock (_lock)
		{
			if (value == _currentLimit)
			{
				return;
			}

			// Swap to a new semaphore for the new limit. We intentionally
			// do not dispose the old semaphore to avoid releasing handles
			// that may still be in use by existing Lease instances.
			// TODO: Consolidate with GpuPrimeWorkLimiter so both limiters share a pooled SemaphoreSlim and avoid rebuilding the
			// limiter state whenever limits are adjusted from the CLI.
			_semaphore = new PollingSemaphore(value, _pauseBetweenChecks, 1);
			_currentLimit = value;
		}
	}

	[method: MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public readonly struct Lease(PollingSemaphore sem)
	{
		private readonly PollingSemaphore _sem = sem;

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public void Dispose()
		{
			_sem.Release();
		}
	}
}

