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

    [Params(3UL, 5UL, 6UL, 8UL, 10UL, 11UL, 32UL, 64UL, 128UL, 256UL, 2_047UL, 8_191UL, 65_535UL, 131_071UL, 2_147_483_647UL, uint.MaxValue - 1_024UL, uint.MaxValue - 511UL, uint.MaxValue - 255UL, uint.MaxValue - 127UL, uint.MaxValue - 63UL, uint.MaxValue - 31UL, uint.MaxValue - 17UL, uint.MaxValue)]
    public uint Value { get; set; }

	/// <summary>
	/// Fastest most of the times. It's usually much faster than it's slower.
	/// </summary>
	/// <returns></returns>
	[Benchmark(Baseline = true)]
    public uint ModuloOperator()
    {
        return Value % 128U;
    }

    [Benchmark]
    public uint ExtensionMethod()
    {
        return Mod128(Value);
    }
}

