using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public sealed class GpuScratchBuffer
{
	internal readonly Accelerator _accelerator;
	public readonly MemoryBuffer1D<int, Stride1D.Dense> SmallPrimeFactorCountSlot;
	public MemoryBuffer1D<int, Stride1D.Dense> SmallPrimeFactorExponentSlots;
	public MemoryBuffer1D<ulong, Stride1D.Dense> SmallPrimeFactorPrimeSlots;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> SmallPrimeFactorRemainingSlot;
	public MemoryBuffer1D<ulong, Stride1D.Dense> SpecialMaxCandidates;
	public MemoryBuffer1D<ulong, Stride1D.Dense> SpecialMaxFactors;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> SpecialMaxResult;

	public GpuScratchBuffer(Accelerator accelerator, int smallPrimeFactorSlotCount, int specialMaxFactorCapacity)
	{
		_accelerator = accelerator;

		// lock (accelerator)
		{
			SmallPrimeFactorPrimeSlots = accelerator.Allocate1D<ulong>(smallPrimeFactorSlotCount);
			SmallPrimeFactorExponentSlots = accelerator.Allocate1D<int>(smallPrimeFactorSlotCount);
			SmallPrimeFactorCountSlot = accelerator.Allocate1D<int>(1);
			SmallPrimeFactorRemainingSlot = accelerator.Allocate1D<ulong>(1);

			SpecialMaxFactors = accelerator.Allocate1D<ulong>(specialMaxFactorCapacity);
			SpecialMaxCandidates = accelerator.Allocate1D<ulong>(specialMaxFactorCapacity);
			SpecialMaxResult = accelerator.Allocate1D<ulong>(1);
		}
	}

	public void ResizeSmallPrimeFactorSlots(int newSize)
	{
		if (SmallPrimeFactorPrimeSlots.Length < newSize)
		{
			SmallPrimeFactorPrimeSlots.Dispose();
			SmallPrimeFactorExponentSlots.Dispose();

			// lock(_accelerator)
			{
				SmallPrimeFactorPrimeSlots = _accelerator.Allocate1D<ulong>(newSize);
				SmallPrimeFactorExponentSlots = _accelerator.Allocate1D<int>(newSize);				
			}
		}
	}

	public void ResizeSpecialMaxFactors(int newSize)
	{
		if (SpecialMaxFactors.Length < newSize)
		{
			SpecialMaxFactors.Dispose();
			SpecialMaxCandidates.Dispose();

			// lock(_accelerator)
			{
				SpecialMaxFactors = _accelerator.Allocate1D<ulong>(newSize);
				SpecialMaxCandidates = _accelerator.Allocate1D<ulong>(newSize);				
			}
		}
	}

	internal void Dispose()
	{
		SmallPrimeFactorPrimeSlots.Dispose();
		SmallPrimeFactorExponentSlots.Dispose();
		SmallPrimeFactorCountSlot.Dispose();
		SmallPrimeFactorRemainingSlot.Dispose();
		SpecialMaxFactors.Dispose();
		SpecialMaxCandidates.Dispose();
		SpecialMaxResult.Dispose();
	}
}

public static class GpuScratchBufferPool
{
	[ThreadStatic]
	private static Dictionary<Accelerator, Queue<GpuScratchBuffer>>? _pools;

	// private static readonly ConcurrentDictionary<Accelerator, ConcurrentQueue<ScratchBuffer>> _pools = new();
	private const int DefaultSmallPrimeFactorSlotCount = 64; // From PrimeOrderCalculator.Gpu.cs
	private const int DefaultSpecialMaxFactorCapacity = 1024; // A reasonable default, will be resized if needed

	public static GpuScratchBuffer Rent(Accelerator accelerator, int smallPrimeFactorSlotCount, int specialMaxFactorCapacity)
	{
		const int WarmUpCapacity = 100;
		// var pool = _pools.GetOrAdd(accelerator, _ => new ConcurrentQueue<ScratchBuffer>());
		if (_pools is not { })
		{
			_pools = new(WarmUpCapacity);
			WarmUp(WarmUpCapacity, DefaultSmallPrimeFactorSlotCount, DefaultSpecialMaxFactorCapacity);
		}
		
		var pools = _pools;
		if (!pools.TryGetValue(accelerator, out var pool))
		{
			pool = new();
			pools[accelerator] = pool;
		}
		
		// var pool = _pools.GetOrAdd(accelerator, _ => new ConcurrentQueue<ScratchBuffer>());
		if (pool.TryDequeue(out var buffer))
		{
			if (buffer.SmallPrimeFactorPrimeSlots.Length < smallPrimeFactorSlotCount || buffer.SpecialMaxFactors.Length < specialMaxFactorCapacity)
			{
				// throw new InvalidOperationException();
				// lock (accelerator)
				{
					// Console.WriteLine($"Resizing GPU scratch buffer from pool ({buffer.SmallPrimeFactorPrimeSlots.Length} / {smallPrimeFactorSlotCount}), ({buffer.SpecialMaxFactors.Length}/{specialMaxFactorCapacity})");
					buffer.ResizeSmallPrimeFactorSlots(smallPrimeFactorSlotCount);
					buffer.ResizeSpecialMaxFactors(specialMaxFactorCapacity);
				}
			}
			return buffer;
		}

		return new GpuScratchBuffer(accelerator, smallPrimeFactorSlotCount, specialMaxFactorCapacity);
	}

	public static void Return(GpuScratchBuffer buffer)
	{
		if (_pools!.TryGetValue(buffer._accelerator, out var pool))
		{
			pool.Enqueue(buffer);
		}
	}

	public static void WarmUp(int count, int smallPrimeFactorSlots, int specialMaxFactorCapacity)
	{
		// Accelerator accelerator = SharedGpuContext.Accelerator;

		// for (int i = 0; i < count; i++)
		// {
		// 	var buffer = Rent(accelerator, smallPrimeFactorSlots, specialMaxFactorCapacity);
		// 	Return(buffer);
		// }
	}

	public static void DisposeAll()
	{
		var pool = _pools;
		if (pool is not null)
		{
			foreach (var queue in pool.Values)
			{
				while (queue.TryDequeue(out var buffer))
				{
					buffer.Dispose();
				}
			}

			pool.Clear();
			_pools = null;
		}
	}
}