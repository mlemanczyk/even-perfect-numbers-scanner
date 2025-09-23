using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod5ULongBenchmarks
{
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong Mod5(ulong value) =>
			(
				(uint)(value & ULongExtensions.WordBitMask) +
				(uint)((value >> 16) & ULongExtensions.WordBitMask) +
				(uint)((value >> 32) & ULongExtensions.WordBitMask) +
				(uint)((value >> 48) & ULongExtensions.WordBitMask)
			) % 5;

    [Params(5UL, 8191UL, 131071UL, 2147483647UL, ulong.MaxValue - 31UL)]
    public ulong Value { get; set; }

	/// <summary>
	/// Fastest
	/// </summary>
	[Benchmark(Baseline = true)]
    public ulong ModuloOperator()
    {
        return Value % 5UL;
    }

    [Benchmark]
    public ulong ExtensionMethod()
    {
        return Mod5(Value);
    }
}

