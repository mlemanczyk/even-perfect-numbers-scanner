using System.Collections.Concurrent;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu.Accelerators;

public static class AcceleratorStreamPool
{
	private static readonly Accelerator[] _accelerators = AcceleratorPool.Shared.Accelerators;
	private static readonly SemaphoreSlim[] _locks = new SemaphoreSlim[PerfectNumberConstants.RollingAccelerators];
	private static readonly AcceleratorStream[] _streams = new AcceleratorStream[PerfectNumberConstants.RollingAccelerators];
	private static SemaphoreSlim CreateLock() => new(PerfectNumberConstants.ThreadsByAccelerator);


	// private static int _rented;

	public static AcceleratorStream Rent(int acceleratorIndex)
	{
		var streamLock = _locks[acceleratorIndex];

		streamLock.Wait();
		var queue = _streams[acceleratorIndex];
		if (queue != null)
		{
			return queue;
		}

		queue = _accelerators[acceleratorIndex].CreateStream();
		_streams[acceleratorIndex] = queue;
		return queue;
	}

	public static void Return(int acceleratorIndex)
	{
		var streamLock = _locks[acceleratorIndex];
		streamLock.Release();
	}

	public static void WarmUp(int acceleratorIndex)
	{
		_locks[acceleratorIndex] = CreateLock();
	}
}
