using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod6UInt128Benchmarks
{
    [Params(6UL, 8191UL, 131071UL, 2147483647UL, ulong.MaxValue - 63UL)]
    public UInt128 Value { get; set; }

    /// <summary>
    /// `% 6` baseline; recorded 2.89-2.91 ns on the smaller values and 2.874 ns at 2147483647, 2.908 ns near ulong.Max.
    /// </summary>
    [Benchmark(Baseline = true)]
    public ulong ModuloOperator()
    {
        return (ulong)(Value % 6UL);
    }

    /// <summary>
    /// Extension helper (<see cref="UInt128Extensions.Mod6"/>); reduced the timings to 1.16-1.24 ns (1.180 ns at value 6, 1.164 ns at
    /// 8191, 1.158 ns at 131071, 1.241 ns at 2147483647, 1.167 ns near ulong.Max), roughly 2.4-2.5x faster.
    /// </summary>
    [Benchmark]
    public ulong ExtensionMethod()
    {
        return Value.Mod6();
    }
}
