using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod11ULongBenchmarks
{
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong Mod11(ulong value)
	{
		ulong remainder = ((uint)value % 11U) + ((uint)(value >> 32) % 11U) << 2;
		while (remainder >= 11UL)
		{
			remainder -= 11UL;
		}

		return remainder;
	}

    [Params(11UL, 8191UL, 131071UL, 2147483647UL, ulong.MaxValue - 127UL)]
    public ulong Value { get; set; }

	/// <summary>
	/// Fastest
	/// </summary>
	[Benchmark(Baseline = true)]
    public ulong ModuloOperator()
    {
        return Value % 11UL;
    }

    [Benchmark]
    public ulong ExtensionMethod()
    {
        return Mod11(Value);
    }
}

