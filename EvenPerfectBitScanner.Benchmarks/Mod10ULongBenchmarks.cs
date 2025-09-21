using System.Numerics;
using BenchmarkDotNet.Attributes;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
public class Mod10ULongBenchmarks
{
    [Params(3UL, 8191UL, 131071UL, 2147483647UL)]
    public ulong Value { get; set; }

    [Benchmark(Baseline = true)]
    public ulong LegacyModulo()
    {
		return Value % 10UL;
    }

    [Benchmark]
    public ulong ModMethodModulo()
    {
        return Value.Mod10();
    }
}

