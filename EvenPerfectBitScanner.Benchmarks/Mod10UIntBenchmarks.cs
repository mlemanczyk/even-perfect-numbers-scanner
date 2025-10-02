using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod10UIntBenchmarks
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Mod10(uint value) => value - (uint)((value * UIntExtensions.Mod5Mask) >> 35) * 10U;

    [Params(10U, 2047U, 65535U, 2147483647U)]
    public uint Value { get; set; }

    /// <summary>
    /// Direct `% 10` on <see cref="uint"/>; BenchmarkDotNet flagged near-zero timings, but the measured means were 0.0407 ns at
    /// value 10, 0.3224 ns at 2047, 0.2382 ns at 65535, and 0.0555 ns at 2147483647.
    /// </summary>
    [Benchmark(Baseline = true)]
    public uint ModuloOperator()
    {
        return Value % 10U;
    }

    /// <summary>
    /// Optimized extension method that folds with a precomputed mask; still costs 0.77 ns on the tiny value 10 but drops to
    /// 0.0448 ns at 2047, 0.0405 ns at 65535, and 0.0372 ns at 2147483647 (roughly 6-9x faster where the baseline exceeded the
    /// empty-loop noise floor).
    /// </summary>
    [Benchmark]
    public uint ExtensionMethod()
    {
        return Mod10(Value);
    }
}
