using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod8UIntBenchmarks
{
    [Params(8U, 2047U, 65535U, 2147483647U)]
    public uint Value { get; set; }

    /// <summary>
    /// `% 8` baseline; means stayed around 0.011-0.015 ns (0.0109 ns at value 8, 0.0151 ns at 2047, 0.0123 ns at 65535, 0.0122 ns at
    /// 2147483647), i.e. indistinguishable from the empty-loop costs.
    /// </summary>
    [Benchmark(Baseline = true)]
    public uint ModuloOperator()
    {
        return Value % 8U;
    }

    /// <summary>
    /// Bitmask helper (`value & 7U`); returned 0.0032 ns at value 8, 0.0113 ns at 2047, 0.0191 ns at 65535, and 0.0150 ns at
    /// 2147483647—roughly matching the baseline within measurement noise.
    /// </summary>
    [Benchmark]
    public uint ExtensionMethod()
    {
        return Value.Mod8();
    }
}
