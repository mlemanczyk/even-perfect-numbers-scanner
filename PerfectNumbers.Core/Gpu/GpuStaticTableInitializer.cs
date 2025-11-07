using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core;

namespace PerfectNumbers.Core.Gpu;

internal static class GpuStaticTableInitializer
{
    private static readonly ConcurrentDictionary<Accelerator, byte> Initialized = new(AcceleratorReferenceComparer.Instance);

    internal static void EnsureStaticTables(KernelContainer kernels, Accelerator accelerator, AcceleratorStream stream)
    {
        Initialized.GetOrAdd(accelerator, acc =>
        {
            PrimeTester.PrimeTesterGpuContextPool.EnsureStaticTables(kernels, acc, stream);
            GpuKernelPool.PreloadStaticTables(kernels, stream);
            PrimeOrderGpuHeuristics.PreloadStaticTables( acc, stream);
            return 0;
        });
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
