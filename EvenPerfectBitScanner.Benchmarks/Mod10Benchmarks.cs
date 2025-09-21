using System.Numerics;
using BenchmarkDotNet.Attributes;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
public class Mod10Benchmarks
{
    [Params(3UL, 8191UL, 131071UL, 2147483647UL)]
    public ulong Value { get; set; }

    [Benchmark(Baseline = true)]
    public ulong LegacyModulo()
    {
		ulong value = Value;
		UInt128 zero = UInt128.Zero;
		if (value == zero)
		{
			return 0UL;
		}

		return value % 10UL;
    }

    [Benchmark]
    public ulong ModMethodModulo()
    {
        return Value.Mod10();
    }
}

