using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu
{
	public struct AcceleratorPool : IDisposable
	{
		internal readonly Context Context;
		internal readonly Device Device;
		internal readonly Accelerator[] Accelerators;

		private int _index;

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public int Rent()
		{
			_ = Interlocked.CompareExchange(ref _index, Capacity, 0);
			var index = Interlocked.Increment(ref _index);
			if (index >= Capacity)
			{
				index -= Capacity;
			}

			return index;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public void Return(int acceleratorIndex)
		{
			// Intentionally left empty - there's nothing to do here. We're not really renting anything.
		}

		public readonly void Dispose()
		{
			Accelerator[] accelerators = Accelerators;
			HashSet<Context> contexts = new(accelerators.Length);
			foreach (var accelerator in accelerators)
			{
				contexts.Add(accelerator.Context);
				accelerator.Synchronize();
				accelerator.Dispose();
			}

			foreach (var context in contexts)
			{
				context.Dispose();
			}
		}

		public static readonly AcceleratorPool Shared = new(PerfectNumberConstants.RollingAccelerators);

		private readonly int Capacity;

		public AcceleratorPool(int capacity)
		{
			Console.WriteLine("Creating accelerators");
			Capacity = capacity;
			var context = Context.CreateDefault();
			var device = context.GetPreferredDevice(false);
			Context = context;
			Device = device;
			int acceleratorIndex = 0;
			Accelerators = [.. Enumerable.Range(0, capacity).Select(_ =>
			{
				Console.WriteLine($"Creating accelerator {acceleratorIndex++}");
				Accelerator accelerator = device.CreateAccelerator(context);

				return accelerator;
			})];
		}
	}
}