using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod6UIntBenchmarks
{
    private static readonly byte[] Mod6Lookup = [0, 3, 4, 1, 2, 5];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Mod6(uint value) => Mod6Lookup[(int)(((value % 3U) << 1) | (value & 1U))];

    [Params(6U, 2047U, 65535U, 2147483647U)]
    public uint Value { get; set; }

    /// <summary>
    /// `% 6` baseline; BenchmarkDotNet highlighted zero-measurement noise with means of 0.0348 ns at value 6, 0.0277 ns at 2047,
    /// 0.0492 ns at 65535, and 0.0385 ns at 2147483647.
    /// </summary>
    [Benchmark(Baseline = true)]
    public uint ModuloOperator()
    {
        return Value % 6U;
    }

    /// <summary>
    /// Lookup-based helper; significantly slower at 0.433 ns (value 6), 0.452 ns (2047), 0.454 ns (65535), and 0.459 ns
    /// (2147483647), about 11-19x the raw modulo cost.
    /// </summary>
    [Benchmark]
    public uint ExtensionMethod()
    {
        return Mod6(Value);
    }
}
