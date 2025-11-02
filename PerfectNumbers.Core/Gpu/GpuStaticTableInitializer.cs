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

    internal static void EnsureStaticTables(Accelerator accelerator)
    {
        if (accelerator is null)
        {
            return;
        }

        Initialized.GetOrAdd(accelerator, static acc =>
        {
            PrimeTester.PrimeTesterGpuContextPool.EnsureStaticTables(acc);
            GpuKernelPool.PreloadStaticTables(acc);
            PrimeOrderGpuHeuristics.PreloadStaticTables(acc);
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
