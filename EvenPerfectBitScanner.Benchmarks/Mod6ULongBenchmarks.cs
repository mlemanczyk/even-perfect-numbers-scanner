using System.Numerics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 3)]
[MemoryDiagnoser]
public class Mod6ULongBenchmarks
{
    private static readonly ulong Divisor = 6UL;
    private static readonly ulong FastDivMul = (ulong)(((UInt128)1 << 64) / Divisor);

    [Params(3UL, 131071UL, ulong.MaxValue)]
    public ulong Value { get; set; }

    [Benchmark(Baseline = true)]
    public ulong ModuloOperator()
    {
                return Value % Divisor;
    }

    [Benchmark]
    public ulong DivisionBased()
    {
        ulong quotient = Value / Divisor;
        return Value - quotient * Divisor;
    }

    [Benchmark]
    public ulong FastDivHigh()
    {
        ulong quotient = Value.FastDiv64(Divisor, FastDivMul);
        return Value - quotient * Divisor;
    }

    [Benchmark]
    public ulong ExtensionMethod()
    {
        return Value.Mod6();
    }
}
