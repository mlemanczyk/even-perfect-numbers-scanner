using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod6UIntBenchmarks
{
	private static readonly byte[] Mod6Lookup = [0, 3, 4, 1, 2, 5];

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static uint Mod6(uint value) => Mod6Lookup[(int)(((value % 3U) << 1) | (value & 1U))];

    [Params(6U, 2047U, 65535U, 2147483647U)]
    public uint Value { get; set; }

	/// <summary>
	/// Fastest
	/// </summary>
	[Benchmark(Baseline = true)]
    public uint ModuloOperator()
    {
        return Value % 6U;
    }

    [Benchmark]
    public uint ExtensionMethod()
    {
        return Mod6(Value);
    }
}

