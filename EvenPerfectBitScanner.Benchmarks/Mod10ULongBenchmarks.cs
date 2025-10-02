using BenchmarkDotNet.Attributes;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
public class Mod10ULongBenchmarks
{
    [Params(3UL, 8191UL, 131071UL, 2147483647UL)]
    public ulong Value { get; set; }

    /// <summary>
    /// Scalar `% 10` baseline; consistently measured around 0.26 ns (0.2591 ns at value 3, 0.2597 ns at 8191, 0.2605 ns at 131071,
    /// 0.2598 ns at 2147483647).
    /// </summary>
    [Benchmark(Baseline = true)]
    public ulong LegacyModulo()
    {
        return Value % 10UL;
    }

    /// <summary>
    /// Extension helper that applies the multiply-high mask; 0.73-0.75 ns across the suite (0.7503 ns at value 3, 0.7320 ns at
    /// 8191, 0.7346 ns at 131071, 0.7411 ns at 2147483647), roughly 2.8-2.9x slower than the raw modulo on `ulong`.
    /// </summary>
    [Benchmark]
    public ulong ModMethodModulo()
    {
        return Value.Mod10();
    }
}
