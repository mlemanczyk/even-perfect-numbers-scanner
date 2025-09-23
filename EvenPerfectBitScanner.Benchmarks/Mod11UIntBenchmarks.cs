using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod11UIntBenchmarks
{
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static uint Mod11(uint value)
	{
		uint remainder = 0U;
		while (value != 0U)
		{
			remainder += value & 1023U;
			value >>= 10;
			remainder -= 11U * (remainder / 11U);
		}

		return remainder;
	}

    [Params(11U, 2047U, 65535U, 2147483647U)]
    public uint Value { get; set; }

	/// <summary>
	/// Fastest
	/// </summary>
	[Benchmark(Baseline = true)]
    public uint ModuloOperator()
    {
        return Value % 11U;
    }

    [Benchmark]
    public uint ExtensionMethod()
    {
        return Mod11(Value);
    }
}

