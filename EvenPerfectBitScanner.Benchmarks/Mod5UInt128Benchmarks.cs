using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod5UInt128Benchmarks
{
    [Params(5UL, 8191UL, 131071UL, 2147483647UL, ulong.MaxValue - 31UL)]
    public UInt128 Value { get; set; }

    /// <summary>
    /// `% 5` baseline; ran at 2.84-2.93 ns (2.840 ns at value 5, 2.874 ns at 8191, 2.870 ns at 131071, 2.895 ns at 2147483647,
    /// 2.927 ns near ulong.Max).
    /// </summary>
    [Benchmark(Baseline = true)]
    public ulong ModuloOperator()
    {
        return (ulong)(Value % 5UL);
    }

    /// <summary>
    /// Extension helper that folds high and low halves; trimmed execution to 0.47-0.49 ns (0.4706 ns at value 5, 0.4691 ns at
    /// 8191, 0.4644 ns at 131071, 0.4847 ns at 2147483647, 0.4669 ns near ulong.Max), about 6x faster.
    /// </summary>
    [Benchmark]
    public ulong ExtensionMethod()
    {
        return Value.Mod5();
    }
}
