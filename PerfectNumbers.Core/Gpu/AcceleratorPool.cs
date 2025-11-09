using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu
{
    public struct AcceleratorPool(int capacity)
	{
		internal readonly Accelerator[] Accelerators = [.. Enumerable.Range(1, capacity).Select(_ => {
			return SharedGpuContext.CreateAccelerator();
		})];

		private int _index;

		public Accelerator Rent()
		{
			Atomic.CompareExchange(ref _index, capacity, 0);
			var index = Atomic.Add(ref _index, 1);
			if (index >= capacity)
			{
				index -= capacity;
			}

			return Accelerators[index];
		}

		public void Return(Accelerator accelerator)
		{
			// Intentionally left empty - there's nothing to do here. We're not really renting anything.
		}

		public static readonly AcceleratorPool Shared = new(PerfectNumberConstants.RollingAccelerators);
	}
}