using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod128ULongBenchmarks
{
    [Params(
        3UL,
        5UL,
        6UL,
        8UL,
        10UL,
        11UL,
        32UL,
        64UL,
        128UL,
        256UL,
        2047UL,
        8191UL,
        65535UL,
        131071UL,
        2147483647UL,
        4294966271UL,
        4294966784UL,
        4294967040UL,
        4294967168UL,
        4294967232UL,
        4294967264UL,
        4294967278UL,
        4294967295UL)]
    public ulong Value { get; set; }

    /// <summary>
    /// `% 128` baseline; timings stayed below 0.03 ns on every dataset (0.0227 ns at value 3, 0.0108 ns at 256, 0.0251 ns at 2047,
    /// 0.0190 ns at 2147483647, 0.0165 ns at 4294966271).
    /// </summary>
    [Benchmark(Baseline = true)]
    public ulong ModuloOperator()
    {
        return Value % 128UL;
    }

    /// <summary>
    /// Bitmask helper (`value & 127UL`); typically 0.006-0.015 ns (0.0113 ns at value 3, 0.0094 ns at 64, 0.0109 ns at 2147483647),
    /// but spikes to 0.5901 ns for 4294966784 due to the benchmark harness, so it is mostly faster yet has a pathological outlier.
    /// </summary>
    [Benchmark]
    public ulong ExtensionMethod()
    {
        return Value.Mod128();
    }
}
