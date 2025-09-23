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

    [Benchmark(Baseline = true)]
    public ulong ModuloOperator()
    {
        return (ulong)(Value % 11UL);
    }

    [Benchmark]
    public ulong ExtensionMethod()
    {
        return Value.Mod11();
    }
}

