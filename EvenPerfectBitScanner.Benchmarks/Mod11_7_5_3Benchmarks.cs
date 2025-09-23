using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
public class Mod11_7_5_3Benchmarks
{
	private static readonly byte[] Mod7ByteCoefficients = [1, 4, 2, 1, 4, 2, 1, 4];
	private static readonly byte[] Mod11ByteCoefficients = [1, 3, 9, 5, 4, 1, 3, 9];

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ulong ReduceMod7(uint value)
	{
		while (value >= 7U)
		{
			value = (value >> 3) + (value & 7U);
			if (value >= 7U && value < 14U)
			{
				value -= 7U;
			}
		}

		return value;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ulong ReduceMod11(uint value)
	{
		while (value >= 11U)
		{
			value = (value & 15U) + ((value >> 4) * 5U);
			if (value >= 11U && value < 22U)
			{
				value -= 11U;
			}
		}

		return value;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void Mod11_7_5_3(ulong value, out ulong mod11, out ulong mod7, out ulong mod5, out ulong mod3)
	{
		uint mod3And5Accumulator = 0U;
		uint mod7Accumulator = 0U;
		uint mod11Accumulator = 0U;

		int index = 0;
		byte current;
		ulong temp = value;

		byte[] mod7ByteCoefficients = Mod7ByteCoefficients;
		byte[] mod11ByteCoefficients = Mod11ByteCoefficients;
		do
		{
			current = (byte)temp;
			mod3And5Accumulator += current;
			mod7Accumulator += (uint)(current * mod7ByteCoefficients[index]);
			mod11Accumulator += (uint)(current * mod11ByteCoefficients[index]);

			temp >>= 8;
			index++;
		}
		while (temp != 0UL);

		mod3 = mod3And5Accumulator % 3;
		mod5 = mod3And5Accumulator % 5;
		mod7 = (uint)ReduceMod7(mod7Accumulator);
		mod11 = (uint)ReduceMod11(mod11Accumulator);
	}

    [Params(3UL, 8191UL, 131071UL, 2147483647UL)]
    public ulong Value { get; set; }

	/// <summary>
	/// Fastest
	/// </summary>
	[Benchmark(Baseline = true)]
    public (ulong Mod11, ulong Mod7, ulong Mod5, ulong Mod3) ModuloOperator()
    {
        ulong value = Value;
        return (value % 11UL, value % 7UL, value % 5UL, value % 3UL);
    }

    [Benchmark]
    public (ulong Mod11, ulong Mod7, ulong Mod5, ulong Mod3) CombinedMethod()
    {
        Mod11_7_5_3(Value, out ulong mod11, out ulong mod7, out ulong mod5, out ulong mod3);
        return (mod11, mod7, mod5, mod3);
    }
}
