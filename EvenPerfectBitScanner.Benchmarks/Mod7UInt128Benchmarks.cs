using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod7UInt128Benchmarks
{
    [Params(7UL, 8191UL, 131071UL, 2147483647UL, ulong.MaxValue - 127UL)]
    public UInt128 Value { get; set; }

    /// <summary>
    /// `% 7` baseline; timings clustered around 2.86-2.91 ns (2.860 ns at value 7, 2.887 ns at 8191, 2.873 ns at 131071,
    /// 2.878 ns at 2147483647, 2.915 ns near ulong.Max).
    /// </summary>
    [Benchmark(Baseline = true)]
    public ulong ModuloOperator()
    {
        return (ulong)(Value % 7UL);
    }

    /// <summary>
    /// Extension helper; achieved 0.676 ns (value 7), 0.675 ns (8191), 0.722 ns (131071), 0.882 ns (2147483647), and 0.676 ns near
    /// ulong.Max—between 3.2x and 4.2x faster than the baseline.
    /// </summary>
    [Benchmark]
    public ulong ExtensionMethod()
    {
        return Value.Mod7();
    }
}
