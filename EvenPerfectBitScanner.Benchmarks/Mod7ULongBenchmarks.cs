using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod7ULongBenchmarks
{
    [Params(7UL, 8191UL, 131071UL, 2147483647UL, ulong.MaxValue - 63UL)]
    public ulong Value { get; set; }

    [Benchmark(Baseline = true)]
    public ulong ModuloOperator()
    {
        return Value % 7UL;
    }

    [Benchmark]
    public ulong ExtensionMethod()
    {
        return Value.Mod7();
    }
}

