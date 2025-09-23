using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod128UInt128Benchmarks
{
    [Params(128UL, 8191UL, 131071UL, 2147483647UL, ulong.MaxValue - 511UL)]
    public UInt128 Value { get; set; }

	[Benchmark(Baseline = true)]
    public ulong ModuloOperator()
    {
        return (ulong)(Value % 128UL);
    }

	/// <summary>
	/// Fastest
	/// </summary>
	/// <returns></returns>
    [Benchmark]
    public ulong ExtensionMethod()
    {
        return Value.Mod128();
    }
}

