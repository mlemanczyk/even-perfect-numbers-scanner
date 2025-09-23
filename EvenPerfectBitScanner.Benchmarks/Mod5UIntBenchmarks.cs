using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod5UIntBenchmarks
{
    [Params(3UL, 5UL, 6UL, 8UL, 10UL, 11UL, 32UL, 64UL, 128UL, 256UL, 2_047UL, 8_191UL, 65_535UL, 131_071UL, 2_147_483_647UL, uint.MaxValue - 1_024UL, uint.MaxValue - 511UL, uint.MaxValue - 255UL, uint.MaxValue - 127UL, uint.MaxValue - 63UL, uint.MaxValue - 31UL, uint.MaxValue - 17UL, uint.MaxValue)]
    public uint Value { get; set; }

    [Benchmark(Baseline = true)]
    public uint ModuloOperator()
    {
        return Value % 5U;
    }

	/// <summary>
	/// Mixed results. Sometimes it's faster, in higher ranges, sometimes it's slower.
	/// </summary>
	[Benchmark]
    public uint ExtensionMethod()
    {
        return Value.Mod5();
    }
}

