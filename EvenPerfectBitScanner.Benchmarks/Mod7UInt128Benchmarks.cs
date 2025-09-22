using System.Numerics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod7UInt128Benchmarks
{
    [Params(7UL, 8191UL, 131071UL, 2147483647UL, ulong.MaxValue - 127UL)]
    public UInt128 Value { get; set; }

    [Benchmark(Baseline = true)]
    public ulong ModuloOperator()
    {
        return (ulong)(Value % 7UL);
    }

    [Benchmark]
    public ulong ExtensionMethod()
    {
        return Value.Mod7();
    }
}

