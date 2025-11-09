using System.Runtime.CompilerServices;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

internal static class GpuStaticTableInitializer
{
    internal static void EnsureStaticTables(Accelerator accelerator, KernelContainer kernels, AcceleratorStream stream)
	{
		// PrimeTester.PrimeTesterGpuContextPool.EnsureStaticTables(accelerator);
		GpuKernelPool.PreloadStaticTables(accelerator, kernels, stream);
		// PrimeOrderGpuHeuristics.PreloadStaticTables(accelerator, stream);
    }

    private sealed class AcceleratorReferenceComparer : IEqualityComparer<Accelerator>
    {
        internal static AcceleratorReferenceComparer Instance { get; } = new();

        public bool Equals(Accelerator? x, Accelerator? y)
        {
            return ReferenceEquals(x, y);
        }

        public int GetHashCode(Accelerator obj)
        {
            return RuntimeHelpers.GetHashCode(obj);
        }
    }
}
