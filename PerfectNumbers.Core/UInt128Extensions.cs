using System.Numerics;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public static class UInt128Extensions
{
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static UInt128 BinaryGcd(this UInt128 u, UInt128 v)
	{
		UInt128 zero = UInt128.Zero;
		if (u == zero)
		{
			return v;
		}

		if (v == zero)
		{
			return u;
		}

		int shift = CountTrailingZeros(u | v);
		u >>= CountTrailingZeros(u);

		do
		{
			v >>= CountTrailingZeros(v);
			if (u > v)
			{
				(u, v) = (v, u);
			}

			v -= u;
		}
		while (v != zero);

		return u << shift;
	}

	public static ulong CalculateOrder(this UInt128 q)
	{
		if (q <= UInt128Numbers.Two)
		{
			return 0UL;
		}

		UInt128 one = UInt128.One;
		UInt128 phi = q - one;
		if (phi > ulong.MaxValue)
		{
			throw new NotImplementedException("Such big values are not yet supported");
		}

		ulong order = (ulong)phi;
		uint[] smallPrimes = PrimesGenerator.SmallPrimes;
		ulong[] smallPrimesPow2 = PrimesGenerator.SmallPrimesPow2;

		int i = 0, primesLength = smallPrimes.Length;
		UInt128 cycle = MersenneDivisorCycles.GetCycle(q);
		ulong prime, temp;
		for (; i < primesLength; i++)
		{
			if (smallPrimesPow2[i] > order)
			{
				break;
			}

			prime = smallPrimes[i];
                        while (order % prime == 0UL)
                        {
                                temp = order / prime;
                                // TODO: Switch this divisor-order powmod to the ProcessEightBitWindows helper so the
                                // cycle factoring loop benefits from the faster windowed pow2 ladder measured in CPU benchmarks.
                                if (temp.PowModWithCycle(q, cycle) == one)
                                {
                                        order = temp;
                                }
				else
				{
					break;
				}
			}
		}

		return order;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static int CountTrailingZeros(this UInt128 value)
	{
		ulong valuePart = (ulong)value;
		if (valuePart != 0UL)
		{
			return BitOperations.TrailingZeroCount(valuePart);
		}

		return 64 + BitOperations.TrailingZeroCount((ulong)(value >> 64));
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsPrimeCandidate(this UInt128 n)
	{
		UInt128 p, zero = UInt128.Zero;

		uint[] smallPrimes = PrimesGenerator.SmallPrimes;
		ulong[] smallPrimesPow2 = PrimesGenerator.SmallPrimesPow2;
		ulong i, smallPrimesCount = (ulong)smallPrimes.Length;
                for (i = 0UL; i < smallPrimesCount; i++)
                {
                        if (smallPrimesPow2[i] > n)
                        {
                                break;
                        }

                        p = smallPrimes[i];
                        // TODO: Replace this direct `%` test with the shared divisor-cycle filter once the
                        // UInt128 path is wired into the cached cycle tables so wide candidates skip the slow
                        // modulo checks during primality pre-filtering.
                        if (n % p == zero)
                        {
                                return n == p;
                        }
                }

		return true;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong Mod3(this UInt128 value)
	{
		ulong remainder = ((ulong)value) % 3UL + ((ulong)(value >> 64)) % 3UL;
		return remainder >= 3UL ? remainder - 3UL : remainder;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong Mod5(this UInt128 value)
	{
		ulong remainder = (((ulong)value) % 5UL) + (((ulong)(value >> 64)) % 5UL);
		return remainder >= 5UL ? remainder - 5UL : remainder;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong Mod6(this UInt128 value) => ((value.Mod3() << 1) | ((ulong)value & 1UL)) switch
	{
		0UL => 0UL,
		1UL => 3UL,
		2UL => 4UL,
		3UL => 1UL,
		4UL => 2UL,
		_ => 5UL,
	};

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Mod7(this UInt128 value)
        {
                ulong low = (ulong)value % 7UL;
                ulong high = (ulong)(value >> 64) % 7UL;
                ulong remainder = low + (high * 2UL);
                if (remainder >= 7UL)
                {
                        remainder -= 7UL;
                        if (remainder >= 7UL)
                        {
                                remainder -= 7UL;
                        }
                }

                return remainder;
        }

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong Mod8(this UInt128 value) => (ulong)value & 7UL;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Mod10(this UInt128 value128)
        {
                // Split and fold under mod 10: 2^64 ≡ 6 (mod 10)
                ulong low = (ulong)value128;
                ulong high = (ulong)(value128 >> 64);
                return ((low % 10UL) + ((high % 10UL) * 6UL)) % 10UL;
        }

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static void Mod10_8_5_3(this UInt128 value, out ulong mod10, out ulong mod8, out ulong mod5, out ulong mod3)
	{
		byte current;
		uint byteSum = 0U;
		mod8 = (ulong)value & 7UL;
		UInt128 zero = UInt128.Zero;
		do
		{
			current = (byte)value;
			byteSum += current;
			value >>= 8;
		}
		while (value != zero);

		mod5 = byteSum.Mod5();
		mod10 = (mod8 & 1UL) == 0UL
						? mod5 switch
						{
							0U => 0UL,
							1U => 6UL,
							2U => 2UL,
							3U => 8UL,
							_ => 4UL,
						}
						: mod5 switch
						{
							0U => 5UL,
							1U => 1UL,
							2U => 7UL,
							3U => 3UL,
							_ => 9UL,
						};

		mod3 = byteSum % 3U;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong Mod11(this UInt128 value)
	{
		ulong remainder = (((ulong)value) % 11UL) + (((ulong)(value >> 64)) % 11UL) * 5UL;
		while (remainder >= 11UL)
		{
			remainder -= 11UL;
		}

		return remainder;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong Mod128(this UInt128 value) => (ulong)value & 127UL;


	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static UInt128 ModPow(this UInt128 value, UInt128 exponent, UInt128 modulus)
	{
		UInt128 zero = UInt128.Zero;
		UInt128 one = UInt128.One;
		UInt128 result = UInt128.One;
		value %= modulus;

		while (exponent != zero)
		{
			if ((exponent & one) != zero)
			{
				result = MulMod(result, value, modulus);
			}

			value = MulMod(value, value, modulus);
			exponent >>= 1;
		}

		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt128 Mul64(this UInt128 a, UInt128 b)
        {
                ulong aLow = (ulong)a;
                ulong bLow = (ulong)b;
                ulong bHigh = (ulong)(b >> 64);

                ulong low = aLow * bLow;

                // Keep the high-word accumulation in locals so the JIT does not rebuild the
                // expression tree around the shift. Mirroring the MulHigh layout lets RyuJIT
                // keep the intermediate sum in registers instead of reloading the partial
                // product from the stack.
                ulong high = aLow.MulHigh(bLow);
                high += aLow * bHigh;

                return ((UInt128)high << 64) | low;
        }

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static UInt128 MulMod(this UInt128 a, UInt128 b, UInt128 modulus)
	{
		UInt128 one = UInt128.One,
				result = UInt128.Zero,
				zero = UInt128.Zero;

		while (b != zero)
		{
			if ((b & one) != zero)
			{
				result += a;
				if (result >= modulus)
				{
					result -= modulus;
				}
			}

			a <<= 1;
			if (a >= modulus)
			{
				a -= modulus;
			}

			b >>= 1;
		}

		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static UInt128 Pow(this UInt128 value, ulong exponent)
	{
		UInt128 result = UInt128.One;
		while (exponent != 0UL)
		{
			if ((exponent & 1UL) != 0UL)
			{
				result *= value;
			}

			value *= value;
			exponent >>= 1;
		}

		return result;
	}
}
