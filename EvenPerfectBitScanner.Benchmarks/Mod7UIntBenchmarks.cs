using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod7UIntBenchmarks
{
    [Params(7U, 2047U, 65535U, 2147483647U)]
    public uint Value { get; set; }

    [Benchmark(Baseline = true)]
    public uint ModuloOperator()
    {
        return Value % 7U;
    }

    [Benchmark]
    public uint ExtensionMethod()
    {
        return Value.Mod7();
    }
}

