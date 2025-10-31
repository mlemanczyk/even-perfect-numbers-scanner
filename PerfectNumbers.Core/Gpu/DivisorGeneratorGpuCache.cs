using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

internal static class DivisorGeneratorGpuCache
{
        private sealed class AcceleratorReferenceComparer : IEqualityComparer<Accelerator>
        {
                public static AcceleratorReferenceComparer Instance { get; } = new();

                public bool Equals(Accelerator? x, Accelerator? y)
                {
                        return ReferenceEquals(x, y);
                }

                public int GetHashCode(Accelerator obj)
                {
                        return RuntimeHelpers.GetHashCode(obj);
                }
        }

        private sealed class DivisorGeneratorGpuTables : IDisposable
        {
                internal DivisorGeneratorGpuTables(Accelerator accelerator)
                {
                        SmallPrimes = AllocateAndCopyUInt(accelerator, DivisorGenerator.SmallPrimes);
                        SmallPrimesPow2 = AllocateAndCopyULong(accelerator, DivisorGenerator.SmallPrimesPow2);
                        SmallPrimesLastOne = AllocateAndCopyUInt(accelerator, DivisorGenerator.SmallPrimesLastOne);
                        SmallPrimesPow2LastOne = AllocateAndCopyULong(accelerator, DivisorGenerator.SmallPrimesPow2LastOne);
                        SmallPrimesLastSeven = AllocateAndCopyUInt(accelerator, DivisorGenerator.SmallPrimesLastSeven);
                        SmallPrimesPow2LastSeven = AllocateAndCopyULong(accelerator, DivisorGenerator.SmallPrimesPow2LastSeven);
                        SmallPrimesLastThree = AllocateAndCopyUInt(accelerator, DivisorGenerator.SmallPrimesLastThree);
                        SmallPrimesPow2LastThree = AllocateAndCopyULong(accelerator, DivisorGenerator.SmallPrimesPow2LastThree);
                        SmallPrimesLastNine = AllocateAndCopyUInt(accelerator, DivisorGenerator.SmallPrimesLastNine);
                        SmallPrimesPow2LastNine = AllocateAndCopyULong(accelerator, DivisorGenerator.SmallPrimesPow2LastNine);
                        SmallPrimesLastOneWithoutLastThree = AllocateAndCopyUInt(accelerator, DivisorGenerator.SmallPrimesLastOneWithoutLastThree);
                        SmallPrimesPow2LastOneWithoutLastThree = AllocateAndCopyULong(accelerator, DivisorGenerator.SmallPrimesPow2LastOneWithoutLastThree);
                        SmallPrimesLastSevenWithoutLastThree = AllocateAndCopyUInt(accelerator, DivisorGenerator.SmallPrimesLastSevenWithoutLastThree);
                        SmallPrimesPow2LastSevenWithoutLastThree = AllocateAndCopyULong(accelerator, DivisorGenerator.SmallPrimesPow2LastSevenWithoutLastThree);
                        SmallPrimesLastNineWithoutLastThree = AllocateAndCopyUInt(accelerator, DivisorGenerator.SmallPrimesLastNineWithoutLastThree);
                        SmallPrimesPow2LastNineWithoutLastThree = AllocateAndCopyULong(accelerator, DivisorGenerator.SmallPrimesPow2LastNineWithoutLastThree);
                }

                internal MemoryBuffer1D<uint, Stride1D.Dense> SmallPrimes { get; }
                internal MemoryBuffer1D<ulong, Stride1D.Dense> SmallPrimesPow2 { get; }
                internal MemoryBuffer1D<uint, Stride1D.Dense> SmallPrimesLastOne { get; }
                internal MemoryBuffer1D<ulong, Stride1D.Dense> SmallPrimesPow2LastOne { get; }
                internal MemoryBuffer1D<uint, Stride1D.Dense> SmallPrimesLastSeven { get; }
                internal MemoryBuffer1D<ulong, Stride1D.Dense> SmallPrimesPow2LastSeven { get; }
                internal MemoryBuffer1D<uint, Stride1D.Dense> SmallPrimesLastThree { get; }
                internal MemoryBuffer1D<ulong, Stride1D.Dense> SmallPrimesPow2LastThree { get; }
                internal MemoryBuffer1D<uint, Stride1D.Dense> SmallPrimesLastNine { get; }
                internal MemoryBuffer1D<ulong, Stride1D.Dense> SmallPrimesPow2LastNine { get; }
                internal MemoryBuffer1D<uint, Stride1D.Dense> SmallPrimesLastOneWithoutLastThree { get; }
                internal MemoryBuffer1D<ulong, Stride1D.Dense> SmallPrimesPow2LastOneWithoutLastThree { get; }
                internal MemoryBuffer1D<uint, Stride1D.Dense> SmallPrimesLastSevenWithoutLastThree { get; }
                internal MemoryBuffer1D<ulong, Stride1D.Dense> SmallPrimesPow2LastSevenWithoutLastThree { get; }
                internal MemoryBuffer1D<uint, Stride1D.Dense> SmallPrimesLastNineWithoutLastThree { get; }
                internal MemoryBuffer1D<ulong, Stride1D.Dense> SmallPrimesPow2LastNineWithoutLastThree { get; }

                public void Dispose()
                {
                        SmallPrimes.Dispose();
                        SmallPrimesPow2.Dispose();
                        SmallPrimesLastOne.Dispose();
                        SmallPrimesPow2LastOne.Dispose();
                        SmallPrimesLastSeven.Dispose();
                        SmallPrimesPow2LastSeven.Dispose();
                        SmallPrimesLastThree.Dispose();
                        SmallPrimesPow2LastThree.Dispose();
                        SmallPrimesLastNine.Dispose();
                        SmallPrimesPow2LastNine.Dispose();
                        SmallPrimesLastOneWithoutLastThree.Dispose();
                        SmallPrimesPow2LastOneWithoutLastThree.Dispose();
                        SmallPrimesLastSevenWithoutLastThree.Dispose();
                        SmallPrimesPow2LastSevenWithoutLastThree.Dispose();
                        SmallPrimesLastNineWithoutLastThree.Dispose();
                        SmallPrimesPow2LastNineWithoutLastThree.Dispose();
                }

                private static MemoryBuffer1D<uint, Stride1D.Dense> AllocateAndCopyUInt(Accelerator accelerator, uint[] source)
                {
                        var buffer = accelerator.Allocate1D<uint>(source.Length);
                        if (source.Length > 0)
                        {
                                buffer.View.CopyFromCPU(source);
                        }

                        return buffer;
                }

                private static MemoryBuffer1D<ulong, Stride1D.Dense> AllocateAndCopyULong(Accelerator accelerator, ulong[] source)
                {
                        var buffer = accelerator.Allocate1D<ulong>(source.Length);
                        if (source.Length > 0)
                        {
                                buffer.View.CopyFromCPU(source);
                        }

                        return buffer;
                }
        }

        private static readonly ConcurrentDictionary<Accelerator, DivisorGeneratorGpuTables> Cache = new(AcceleratorReferenceComparer.Instance);

        public static void WarmUpByDivisorTables(Accelerator accelerator)
        {
                if (accelerator is null || accelerator.AcceleratorType == AcceleratorType.CPU)
                {
                        return;
                }

                Cache.GetOrAdd(accelerator, acc => new DivisorGeneratorGpuTables(acc));
        }

        public static void Clear(Accelerator accelerator)
        {
                if (accelerator is null)
                {
                        return;
                }

                if (Cache.TryRemove(accelerator, out var tables))
                {
                        tables.Dispose();
                }
        }

        public static void DisposeAll()
        {
                foreach (var entry in Cache)
                {
                        entry.Value.Dispose();
                }

                Cache.Clear();
        }
}
