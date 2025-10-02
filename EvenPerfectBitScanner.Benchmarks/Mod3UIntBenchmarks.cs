using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod3UIntBenchmarks
{
    [Params(3U, 2047U, 65535U, 2147483647U)]
    public uint Value { get; set; }

    /// <summary>
    /// `% 3` baseline; the near-zero timings (0.0320 ns at value 3, 0.0918 ns at 2047, 0.0355 ns at 65535, 0.0309 ns at
    /// 2147483647) reflect the noise floor flagged by BenchmarkDotNet.
    /// </summary>
    [Benchmark(Baseline = true)]
    public uint ModuloOperator()
    {
        return Value % 3U;
    }

    /// <summary>
    /// Extension helper that folds digits; produced 0.0217 ns at value 3, 0.0303 ns at 2047, 0.0597 ns at 65535, and 0.0350 ns at
    /// 2147483647, tracking the baseline within measurement noise.
    /// </summary>
    [Benchmark]
    public uint ExtensionMethod()
    {
        return UIntExtensions.Mod3(Value);
    }
}
