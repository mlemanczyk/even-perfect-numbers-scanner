using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 3)]
[MemoryDiagnoser]
public class Mod6ULongBenchmarks
{
    private static readonly byte[] Mod6Lookup = [0, 3, 4, 1, 2, 5];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Mod6(ulong value) => Mod6Lookup[(int)(((value % 3UL) << 1) | (value & 1UL))];

    [Params(3UL, 131071UL, ulong.MaxValue - 1024UL)]
    public ulong Value { get; set; }

    /// <summary>
    /// `% 6` baseline; recorded 0.264 ns at Value 3, 0.262 ns at 131071, and 0.271 ns near ulong.Max.
    /// </summary>
    [Benchmark(Baseline = true)]
    public ulong ModuloOperator()
    {
        return Value % 6UL;
    }

    /// <summary>
    /// Division-based reduction (`value / 6`); timings were 0.277 ns, 0.284 ns, and 0.287 ns across the three start values.
    /// </summary>
    [Benchmark]
    public ulong DivisionBased()
    {
        ulong quotient = Value / 6UL;
        return Value - quotient * 6UL;
    }

    /// <summary>
    /// FastDivHigh helper; significantly slower at 1.24 ns (Value 3), 1.04 ns (131071), and 1.17 ns (near ulong.Max).
    /// </summary>
    [Benchmark]
    public ulong FastDivHigh()
    {
        ulong quotient = Value.FastDiv64(6UL, (ulong)(((UInt128)1 << 64) / 6UL));
        return Value - quotient * 6UL;
    }

    /// <summary>
    /// Lookup-based helper; cost 0.461 ns at Value 3, 0.467 ns at 131071, 0.467 ns near ulong.Max—about 1.7x slower than the modulo.
    /// </summary>
    [Benchmark]
    public ulong ExtensionMethod()
    {
        return Mod6(Value);
    }
}
