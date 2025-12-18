using System.Runtime.CompilerServices;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu.Accelerators;

public static class AcceleratorStreamPool
{
	private static readonly Accelerator[] _accelerators = AcceleratorPool.Shared.Accelerators;
	private static readonly PollingSemaphore[] _locks = new PollingSemaphore[EnvironmentConfiguration.RollingAccelerators];
	private static readonly ConcurrentFixedCapacityStack<AcceleratorStream>[] _streams = new ConcurrentFixedCapacityStack<AcceleratorStream>[EnvironmentConfiguration.RollingAccelerators];

	private static PollingSemaphore CreateLock() => new(PerfectNumberConstants.ThreadsByAccelerator, TimeSpan.FromMilliseconds(1000), 1);


	// private static int _rented;

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static AcceleratorStream Rent(int acceleratorIndex)
	{
		var streamLock = _locks[acceleratorIndex];
		var queue = _streams[acceleratorIndex];

		streamLock.Wait();

		return queue.Pop() is { } stream
			? stream
			: _accelerators[acceleratorIndex].CreateStream();
	}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static void Return(int acceleratorIndex, AcceleratorStream stream)
	{
		var streamLock = _locks[acceleratorIndex];
		var queue = _streams[acceleratorIndex];
		queue.Push(stream);
		streamLock.Release();
	}

	public static void WarmUp(int acceleratorIndex)
	{
		_streams[acceleratorIndex] = new(PerfectNumberConstants.DefaultPoolCapacity);
		_locks[acceleratorIndex] = CreateLock();
	}
}
