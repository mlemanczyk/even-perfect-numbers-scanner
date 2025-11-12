using System.Collections.Concurrent;
using ILGPU.Runtime;
using Open.Collections;

namespace PerfectNumbers.Core.Gpu.Accelerators;

public static class AcceleratorStreamPool
{
	private static readonly Dictionary<Accelerator, ConcurrentQueue<AcceleratorStream>> _streams = new(PerfectNumberConstants .RollingAccelerators);
	// private static readonly ConcurrentDictionary<Accelerator, AcceleratorStream> _streams = new(AcceleratorPool.Shared.Accelerators.ToDictionary(static x => x, CreateStream));

	private static SemaphoreSlim CreateLock() => new(PerfectNumberConstants.ThreadsByStream);
	private static AcceleratorStream CreateStream(Accelerator accelerator) => accelerator.CreateStream();

	private static readonly Dictionary<Accelerator, SemaphoreSlim> _locks = new(AcceleratorPool.Shared.Accelerators.ToDictionary(x => x, static x => CreateLock()));

	public static AcceleratorStream Rent(Accelerator accelerator)
	{
		var streamLock = _locks[accelerator];
		// var streamLock = _locks.GetOrAdd(accelerator, static _ => CreateLock());

		streamLock.Wait();
		var queue = _streams[accelerator];

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

		// ConcurrentQueue<AcceleratorStream> streamQueue = new();
		// var stream = accelerator.CreateStream();
		// // var stream = accelerator.CreateStream();
		// streamQueue.Enqueue(stream);
		// _streams[accelerator] = streamQueue;
		// _locks[accelerator] = CreateLock();
		// stream.Synchronize();
	}
}
