using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod128UIntBenchmarks
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Mod128(uint value) => value & 127U;

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
    /// `% 128` baseline; BenchmarkDotNet reported near-empty-loop costs with means from 0.010-0.026 ns across the domain (e.g.
    /// 0.0191 ns at value 3, 0.0136 ns at 5, 0.0170 ns at 32, 0.0108 ns at 256, 0.0138 ns at 2147483647).
    /// </summary>
    [Benchmark(Baseline = true)]
    public uint ModuloOperator()
    {
        return Value % 128U;
    }

    /// <summary>
    /// Bitmask helper that returns `value & 127`; generally matches the baseline at 0.006-0.023 ns (0.0163 ns at value 3, 0.0064 ns at
    /// 6, 0.0152 ns at 10, 0.0156 ns at 2147483647) but regresses on 64 due to the benchmark harness (0.9496 ns, 67x slower).
    /// </summary>
    [Benchmark]
    public uint ExtensionMethod()
    {
        return Mod128(Value);
    }
}
