using System.Collections.Concurrent;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu.Accelerators;

public static class AcceleratorStreamPool
{
	private static readonly Accelerator[] _accelerators = AcceleratorPool.Shared.Accelerators;
	private static readonly SemaphoreSlim[] _locks = new SemaphoreSlim[PerfectNumberConstants.RollingAccelerators];
	private static readonly ConcurrentQueue<AcceleratorStream>[] _streams = new ConcurrentQueue<AcceleratorStream> [PerfectNumberConstants.RollingAccelerators];

	private static SemaphoreSlim CreateLock() => new(PerfectNumberConstants.ThreadsByAccelerator);


	// private static int _rented;

	public static AcceleratorStream Rent(int acceleratorIndex)
	{
		// var streamLock = _locks[acceleratorIndex];
		var queue = _streams[acceleratorIndex];

		// streamLock.Wait();

		return queue.TryDequeue(out var stream)
			? stream
			: _accelerators[acceleratorIndex].CreateStream();
	}

	public static void Return(int acceleratorIndex, AcceleratorStream stream)
	{
		// var streamLock = _locks[acceleratorIndex];
		var queue = _streams[acceleratorIndex];
		queue.Enqueue(stream);
		// streamLock.Release();
	}

	public static void WarmUp(int acceleratorIndex)
	{
		_streams[acceleratorIndex] = new();
		_locks[acceleratorIndex] = CreateLock();
	}
}
