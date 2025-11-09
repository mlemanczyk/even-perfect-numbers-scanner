using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public sealed class GpuInputOutputScratchBuffer<TInput, TOutput>
	where TInput : unmanaged
	where TOutput: unmanaged
{
	private readonly Accelerator _accelerator;
	public MemoryBuffer1D<TInput, Stride1D.Dense> Input { get; private set; }
	public MemoryBuffer1D<TOutput, Stride1D.Dense> Output { get; private set; }

	public GpuInputOutputScratchBuffer(Accelerator accelerator,int inputCapacity, int outputCapacity)
	{
		_accelerator = accelerator;
		// lock (accelerator)
		{
			Input = accelerator.Allocate1D<TInput>(inputCapacity);
			Output = accelerator.Allocate1D<TOutput>(outputCapacity);
		}
	}

	public void EnsureCapacity(int inputCapacity, int outputCapacity)
	{
		if (Input.Length >= inputCapacity && Output.Length >= outputCapacity)
		{
			return;
		}

		Accelerator accelerator = _accelerator;
		if (Input.Length >= inputCapacity && Output.Length >= outputCapacity)
		{
			return;
		}

		// lock (accelerator)
		{
			if (Input.Length < inputCapacity)
			{
				Input.Dispose();
				Input = accelerator.Allocate1D<TInput>(inputCapacity);
			}

			if (Output.Length < outputCapacity)
			{
				Output.Dispose();
				Output = accelerator.Allocate1D<TOutput>(outputCapacity);
			}
		}
	}
	
	public void Dispose()
	{
		Input.Dispose();
		Output.Dispose();
	}
}

public static class GpuInputOutputScratchBufferPool<TInput, TOutput>
	where TInput : unmanaged
	where TOutput : unmanaged
{
	private const int WarmUpCapacity = 20480;

	[ThreadStatic]
	private static Dictionary<Accelerator, Queue<GpuInputOutputScratchBuffer<TInput, TOutput>>>? _pool;

	private static Queue<GpuInputOutputScratchBuffer<TInput, TOutput>> GetPools(Accelerator accelerator)
	{
		var pool = _pool ??= [];

		if (!pool.TryGetValue(accelerator, out var bufferQueue))
		{
			bufferQueue = [];
			pool[accelerator] = bufferQueue;
		}
		
		return bufferQueue;
	}

	// private static readonly ConcurrentDictionary<Accelerator, ConcurrentQueue<InputOutputScratchBuffer>> _pools = new();

	private const int DefaultSmallPrimeFactorSlotCount = 64; // From PrimeOrderCalculator.Gpu.cs
	private const int DefaultSpecialMaxFactorCapacity = 1024; // A reasonable default, will be resized if needed

	public static GpuInputOutputScratchBuffer<TInput, TOutput> Rent(Accelerator accelerator, int inputCapacity, int outputCapacity)
	{
		var bufferQueue = GetPools(accelerator);
		if (bufferQueue.TryDequeue(out var buffer))
		{
			if (buffer.Input.Length < inputCapacity || buffer.Output.Length < outputCapacity)
			{
				// lock (accelerator)
				// {
				// Console.WriteLine($"Resizing GPU scratch buffer from pool ({buffer.SmallPrimeFactorPrimeSlots.Length} / {smallPrimeFactorSlotCount}), ({buffer.SpecialMaxFactors.Length}/{specialMaxFactorCapacity})");
				buffer.EnsureCapacity(inputCapacity, outputCapacity);
				// }
			}

			return buffer;
		}

		return new GpuInputOutputScratchBuffer<TInput, TOutput>(accelerator, inputCapacity, outputCapacity);
	}

	public static void Return(Accelerator accelerator, GpuInputOutputScratchBuffer<TInput, TOutput> buffer) => GetPools(accelerator).Enqueue(buffer);

	public static void WarmUp(int count, int inputCapacity, int outputCapacity)
	{
		// List<GpuInputOutputScratchBuffer<TInput, TOutput>> rented = new(count);
		// for (int i = 0; i < count; i++)
		// {
		// 	rented.Add(Rent(inputCapacity, outputCapacity));
		// }

		// for (int i = 0; i < count; i++)
		// {
		// 	Return(rented[i]);
		// }

		// rented.Clear();
		// rented.Capacity = 1;
	}
}