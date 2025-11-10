using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu
{
	public struct AcceleratorPool
	{
		internal readonly Context Context;
		internal readonly Device Device;
		internal readonly Accelerator[] Accelerators;

		private int _index;

		public Accelerator Rent()
		{
			Atomic.CompareExchange(ref _index, Capacity, 0);
			var index = Atomic.Add(ref _index, 1);
			if (index >= Capacity)
			{
				index -= Capacity;
			}

			return Accelerators[index];
		}

		public void Return(Accelerator accelerator)
		{
			// Intentionally left empty - there's nothing to do here. We're not really renting anything.
		}

		public static readonly AcceleratorPool Shared = new(PerfectNumberConstants.RollingAccelerators);
		private readonly int Capacity;

		public AcceleratorPool(int capacity)
		{
			Capacity = capacity;
			var context = Context.CreateDefault();
			var device = context.GetPreferredDevice(false);
			Context = context;
			Device = device;
			Accelerators = [.. Enumerable.Range(0, capacity).Select(_ => device.CreateAccelerator(context))];
		}
	}
}