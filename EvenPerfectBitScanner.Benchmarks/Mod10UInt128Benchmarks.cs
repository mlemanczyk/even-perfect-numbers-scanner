using BenchmarkDotNet.Attributes;

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

	/// <summary>
	/// Fastest
	/// </summary>
	[Benchmark]
	public ulong ModWithHighLowModulo()
	{
		return ModWithHighLowModuloHelper(Value);
	}

	private static ulong ModWithHighLowModuloHelper(UInt128 value128)
	{
		// Split and fold under mod 10: 2^64 ≡ 6 (mod 10)
		ulong value = (ulong)value128;
		return ((value % 10UL) + ((value >> 64) % 10UL) * 6UL) % 10UL;
	}

	[Benchmark]
	public ulong ModWithLoopModulo()
	{
		return ModWithLoopModuloHelper(Value);
	}

	private static ulong ModWithLoopModuloHelper(UInt128 value)
	{
		UInt128 zero = UInt128.Zero;
		if (value == zero)
			return 0UL;

		ulong result = (ulong)value;
		value >>= 64;

		while (value != zero)
		{
			// 2^64 ≡ 6 (mod 10)
			result = (result + (ulong)value * 6UL) % 10UL;
			value >>= 64;
		}

		return result % 10UL;
	}
}

