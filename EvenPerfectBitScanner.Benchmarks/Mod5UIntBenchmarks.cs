using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod5UIntBenchmarks
{
    [Params(
        3U,
        5U,
        6U,
        8U,
        10U,
        11U,
        32U,
        64U,
        128U,
        256U,
        2047U,
        8191U,
        65535U,
        131071U,
        2147483647U,
        uint.MaxValue - 1024U,
        uint.MaxValue - 511U,
        uint.MaxValue - 255U,
        uint.MaxValue - 127U,
        uint.MaxValue - 63U,
        uint.MaxValue - 31U,
        uint.MaxValue - 17U,
        uint.MaxValue)]
    public uint Value { get; set; }

    /// <summary>
    /// `% 5` baseline; times hovered around 0.03 ns with the expected zero-measurement noise (0.0296 ns at value 5, 0.0671 ns at 6,
    /// 0.0326 ns at 8, 0.0289 ns at 10, 0.1236 ns at 11, dropping back to ~0.03 ns for large values).
    /// </summary>
    [Benchmark(Baseline = true)]
    public uint ModuloOperator()
    {
        return Value % 5U;
    }

    /// <summary>
    /// Extension helper; typically within 0.03-0.04 ns (0.0349 ns at value 3, 0.0294 ns at 8, 0.0305 ns at 65535, 0.0270 ns at 131071,
    /// 0.0327 ns at 2147483647) though the dataset contains an outlier at value 11 (0.0315 ns) and value 64 (0.0356 ns).
    /// </summary>
    [Benchmark]
    public uint ExtensionMethod()
    {
        return Value.Mod5();
    }
}
