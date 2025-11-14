using System.Collections.Concurrent;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu.Accelerators;

public static class AcceleratorStreamPool
{
	private static readonly Dictionary<Accelerator, ConcurrentQueue<AcceleratorStream>> _streams = new(PerfectNumberConstants .RollingAccelerators << 1);

	private static SemaphoreSlim CreateLock() => new(PerfectNumberConstants.ThreadsByStream);

	private static readonly Dictionary<Accelerator, SemaphoreSlim> _locks = new(PerfectNumberConstants.RollingAccelerators << 1);

	// private static int _rented;

	public static AcceleratorStream Rent(Accelerator accelerator)
	{
		var streamLock = _locks[accelerator];
		var queue = _streams[accelerator];

		streamLock.Wait();

		return queue.TryDequeue(out var stream)
			? stream
			: accelerator.CreateStream();
	}

	public static void Return(AcceleratorStream stream)
	{
		Accelerator accelerator = stream.Accelerator;
		var streamLock = _locks[accelerator];
		var queue = _streams[accelerator];
		queue.Enqueue(stream);
		streamLock.Release();
	}

	public static void WarmUp(Accelerator accelerator)
	{
		_streams.Add(accelerator, new());
		_ = _locks.TryAdd(accelerator, CreateLock());
	}
}
