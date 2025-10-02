using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod11UInt128Benchmarks
{
    [Params(11UL, 8191UL, 131071UL, 2147483647UL, ulong.MaxValue - 255UL)]
    public UInt128 Value { get; set; }

    /// <summary>
    /// Direct `% 11` for <see cref="UInt128"/>; ran in ~2.85-2.87 ns on every dataset (2.8719 ns at value 11, 2.8715 ns at 8191,
    /// 2.8534 ns at 131071, 2.8716 ns at 2147483647, 2.8715 ns near ulong.Max).
    /// </summary>
    [Benchmark(Baseline = true)]
    public ulong ModuloOperator()
    {
        return (ulong)(Value % 11UL);
    }

    /// <summary>
    /// Extension helper that folds digits with lookup coefficients; 0.66-0.68 ns across the inputs (0.6721 ns at value 11, 0.6765 ns
    /// at 8191, 0.6760 ns at 131071, 0.6614 ns at 2147483647, 0.6661 ns near ulong.Max), about 4.2x faster than the baseline.
    /// </summary>
    [Benchmark]
    public ulong ExtensionMethod()
    {
        return Value.Mod11();
    }
}
