using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod128UInt128Benchmarks
{
    [Params(128UL, 8191UL, 131071UL, 2147483647UL, ulong.MaxValue - 511UL)]
    public UInt128 Value { get; set; }

    /// <summary>
    /// `% 128` baseline for <see cref="UInt128"/>; measured 2.95-3.42 ns across the suite (3.423 ns at value 128, 3.025 ns at
    /// 8191, 2.950 ns at 131071, 2.948 ns at 2147483647, 2.967 ns near ulong.Max).
    /// </summary>
    [Benchmark(Baseline = true)]
    public ulong ModuloOperator()
    {
        return (ulong)(Value % 128UL);
    }

    /// <summary>
    /// Bitmask helper (`value & 127UL`); durations were indistinguishable from the empty-loop cost at 0.004-0.025 ns—roughly
    /// 150-1000x faster than the modulo baseline on every sample.
    /// </summary>
    [Benchmark]
    public ulong ExtensionMethod()
    {
        return Value.Mod128();
    }
}
