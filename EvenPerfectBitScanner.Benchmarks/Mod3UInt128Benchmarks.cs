using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod3UInt128Benchmarks
{
    [Params(3UL, 8191UL, 131071UL, 2147483647UL, ulong.MaxValue - 17UL)]
    public UInt128 Value { get; set; }

    /// <summary>
    /// Direct `% 3` on <see cref="UInt128"/>; measured 2.88-2.97 ns (2.969 ns at value 3, 2.887 ns at 8191, 2.885 ns at 131071,
    /// 2.884 ns at 2147483647, 2.933 ns near ulong.Max).
    /// </summary>
    [Benchmark(Baseline = true)]
    public ulong ModuloOperator()
    {
        return (ulong)(Value % 3UL);
    }

    /// <summary>
    /// Extension helper folding the high and low halves; trimmed timings to 0.455-0.475 ns (0.4549 ns at value 3, 0.4562 ns at
    /// 8191, 0.4656 ns at 131071, 0.4710 ns at 2147483647, 0.4750 ns near ulong.Max), about 6.4x faster than the baseline.
    /// </summary>
    [Benchmark]
    public ulong ExtensionMethod()
    {
        return Value.Mod3();
    }
}
