using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod7UIntBenchmarks
{
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static uint Mod7(uint value)
	{
		uint remainder = 0U;
		while (value != 0U)
		{
			remainder += value & 7U;
			value >>= 3;
			if (remainder >= 7U)
			{
				remainder -= 7U;
			}
		}

		return remainder >= 7U ? remainder - 7U : remainder;
	}

    [Params(7U, 2047U, 65535U, 2147483647U)]
    public uint Value { get; set; }

	/// <summary>
	/// Fastest
	/// </summary>
	[Benchmark(Baseline = true)]
    public uint ModuloOperator()
    {
        return Value % 7U;
    }

    [Benchmark]
    public uint ExtensionMethod()
    {
        return Mod7(Value);
    }
}

