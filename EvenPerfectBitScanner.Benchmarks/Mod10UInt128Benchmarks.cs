using BenchmarkDotNet.Attributes;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
public class Mod10UInt128Benchmarks
{
	[Params(3UL, 8191UL, 131071UL, 2147483647UL)]
	public UInt128 Value { get; set; }

	[Benchmark(Baseline = true)]
	public UInt128 StandardModulo()
	{
		return Value % 10UL;
	}

	[Benchmark]
	public ulong ModWithHighLowModulo()
	{
		// Split and fold under mod 10: 2^64 ≡ 6 (mod 10)
		ulong value = (ulong)Value;
		ulong high = value >> 64;
		ulong low = value;
		ulong highRem = high.Mod10();
		ulong lowRem = low.Mod10();
		ulong combined = lowRem + highRem * 6UL;
		return combined.Mod10();
	}

    [Benchmark]
    public ulong ModWithLoopModulo()
    {
		UInt128 value = Value;
		UInt128 zero = UInt128.Zero;
		if (value == zero)
			return 0UL;

		ulong result = (ulong)value;
		UInt128 high = value >> 64;

		while (high != zero)
		{
			// 2^64 ≡ 6 (mod 10)
			result = (result + (ulong)high * 6UL).Mod10();
			high >>= 64;
		}

		return result.Mod10();
    }
}

