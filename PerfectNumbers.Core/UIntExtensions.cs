using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public static class UIntExtensions
{
    private const ulong Mod3Mask = 0xAAAAAAABUL;
    public const ulong Mod5Mask = 0xCCCCCCCDUL;
    // TODO: Only add Mod7/Mod11 lookup helpers once we have a variant that beats the `%` baseline (current prototypes lose per Mod7/Mod11 benchmarks).
    // the small-prime sieves.

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Mod3(uint value) => value - (uint)((value * Mod3Mask) >> 33) * 3U;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static uint Mod5(this uint value) => value - (uint)((value * Mod5Mask) >> 34) * 5U;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static uint Mod8(this uint value) => value & 7U;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static uint ReduceCycleRemainder(this uint value, uint modulus)
	{
		if (modulus == 0U || value < modulus)
		{
			return value;
		}

		value -= modulus;
		if (value < modulus)
		{
			return value;
		}

		value -= modulus;
		if (value < modulus)
		{
			return value;
		}

		value -= modulus;
		if (value < modulus)
		{
			return value;
		}

		value -= modulus;
		if (value < modulus)
		{
			return value;
		}

		value -= modulus;
		if (value < modulus)
		{
			return value;
		}

		value -= modulus;
		if (value < modulus)
		{
			return value;
		}

		return value % modulus;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static uint ReduceCycleRemainder(this ulong value, uint modulus)
		=> (uint)value.ReduceCycleRemainder((ulong)modulus);
}
