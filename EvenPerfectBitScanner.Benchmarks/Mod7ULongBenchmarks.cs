using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod7ULongBenchmarks
{
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong Mod7(ulong value)
	{
		ulong remainder = ((uint)value % 7U) + (((uint)(value >> 32) % 7U) << 2);
		while (remainder >= 7UL)
		{
			remainder -= 7UL;
		}

		return remainder;
	}

    [Params(7UL, 8191UL, 131071UL, 2147483647UL, ulong.MaxValue - 63UL)]
    public ulong Value { get; set; }

	/// <summary>
	/// Fastest
	/// </summary>
	[Benchmark(Baseline = true)]
    public ulong ModuloOperator()
    {
        return Value % 7UL;
    }

    [Benchmark]
    public ulong ExtensionMethod()
    {
        return Mod7(Value);
    }
}

