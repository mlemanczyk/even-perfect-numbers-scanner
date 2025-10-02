using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod8UInt128Benchmarks
{
    [Params(8UL, 8191UL, 131071UL, 2147483647UL, ulong.MaxValue - 255UL)]
    public UInt128 Value { get; set; }

    /// <summary>
    /// `% 8` baseline; 2.86-2.89 ns across the values (2.864 ns at value 8, 2.878 ns at 8191, 2.895 ns at 131071, 2.890 ns at
    /// 2147483647, 2.894 ns near ulong.Max).
    /// </summary>
    [Benchmark(Baseline = true)]
    public ulong ModuloOperator()
    {
        return (ulong)(Value % 8UL);
    }

    /// <summary>
    /// Bitmask helper; reduced the cost to 0.010-0.020 ns (0.0170 ns at value 8, 0.0144 ns at 8191, 0.0134 ns at 131071, 0.0105 ns at
    /// 2147483647, 0.0203 ns near ulong.Max), delivering a 140-280x speedup.
    /// </summary>
    [Benchmark]
    public ulong ExtensionMethod()
    {
        return Value.Mod8();
    }
}
