using System.Collections.Concurrent;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu.Accelerators;

public static class AcceleratorStreamPool
{
	private static readonly ConcurrentDictionary<Accelerator, ConcurrentQueue<AcceleratorStream>> _streams = new(20_480, PerfectNumberConstants.RollingAccelerators);
	// private static readonly ConcurrentDictionary<Accelerator, AcceleratorStream> _streams = new(AcceleratorPool.Shared.Accelerators.ToDictionary(static x => x, CreateStream));

	private static SemaphoreSlim CreateLock() => new(PerfectNumberConstants.ThreadsByStream);
	private static AcceleratorStream CreateStream(Accelerator accelerator) => accelerator.CreateStream();

	private static readonly ConcurrentDictionary<Accelerator, SemaphoreSlim> _locks = new(AcceleratorPool.Shared.Accelerators.ToDictionary(x => x, static x => CreateLock()));

	public static AcceleratorStream Rent(Accelerator accelerator)
	{
		var streamLock = _locks.GetOrAdd(accelerator, static _ => new SemaphoreSlim(PerfectNumberConstants.ThreadsByStream));

		streamLock.Wait();
		var queue = _streams.GetOrAdd(accelerator, static _ => new());

		return queue.TryDequeue(out var stream)
			? stream
			: accelerator.CreateStream();
	}

	public static void Return(AcceleratorStream stream)
	{
		Accelerator accelerator = stream.Accelerator;
		var streamLock = _locks.GetOrAdd(accelerator, static _ => CreateLock());
		var queue = _streams.GetOrAdd(accelerator, static _ => new());
		queue.Enqueue(stream);
		streamLock.Release();
	}
}
