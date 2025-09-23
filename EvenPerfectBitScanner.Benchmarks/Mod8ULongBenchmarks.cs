using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod8ULongBenchmarks
{
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong Mod8(ulong value) => value & 7UL;

    [Params(8UL, 8191UL, 131071UL, 2147483647UL, ulong.MaxValue - 127UL)]
    public ulong Value { get; set; }

	/// <summary>
	/// Fastest
	/// </summary>
    [Benchmark(Baseline = true)]
    public ulong ModuloOperator()
    {
        return Value % 8UL;
    }

	[Benchmark]
    public ulong ExtensionMethod()
    {
        return Mod8(Value);
    }
}

