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
			p = smallPrimes[i];
			if (smallPrimesPow2[i] > n)
			{
				break;
			}

			if (n % p == zero)
			{
				return n == p;
			}
		}

		return true;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static void Mod10_8_5_3(this UInt128 value, out ulong mod10, out ulong mod8, out ulong mod5, out ulong mod3)
	{
		UInt128 temp = value;
		uint byteSum = 0U;

		do
		{
			byteSum += (byte)temp;
			temp >>= 8;
		}
		while (temp != UInt128.Zero);

		mod8 = (ulong)value & 7UL;

		uint mod5Value = PerfectNumbersMath.FastRemainder5(byteSum);
		uint mod3Value = PerfectNumbersMath.FastRemainder3(byteSum);

		ulong parity = mod8 & 1UL;
		mod10 = parity == 0UL
				? mod5Value switch
				{
					0U => 0UL,
					1U => 6UL,
					2U => 2UL,
					3U => 8UL,
					_ => 4UL,
				}
				: mod5Value switch
				{
					0U => 5UL,
					1U => 1UL,
					2U => 7UL,
					3U => 3UL,
					_ => 9UL,
				};

		mod5 = mod5Value;
		mod3 = mod3Value;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static UInt128 Pow(this UInt128 value, ulong exponent)
	{
		UInt128 result = UInt128.One;
		UInt128 baseValue = value;

		while (exponent != 0UL)
		{
			if ((exponent & 1UL) != 0UL)
			{
				result *= baseValue;
			}

			baseValue *= baseValue;
			exponent >>= 1;
		}

		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static UInt128 ModPow(this UInt128 value, UInt128 exponent, UInt128 modulus)
	{
		UInt128 result, one = result = UInt128.One,
						zero = UInt128.Zero,
						baseValue = value % modulus;

		while (exponent != zero)
		{
			if ((exponent & one) != zero)
			{
				result = MulMod(result, baseValue, modulus);
			}

			baseValue = MulMod(baseValue, baseValue, modulus);
			exponent >>= 1;
		}

		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong Mod10(this UInt128 value)
	{
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

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong Mod8(this UInt128 value) => (ulong)value & 7UL;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong Mod3(this UInt128 value)
	{
		// 2^64 ≡ 1 (mod 3)
		ulong rem = ((ulong)value % 3UL) + ((ulong)(value >> 64) % 3UL);
		return rem >= 3UL ? rem - 3UL : rem;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong Mod5(this UInt128 value)
	{
		// 2^64 ≡ 1 (mod 5)
		ulong rem = ((ulong)value % 5UL) + ((ulong)(value >> 64) % 5UL);
		return rem >= 5UL ? rem - 5UL : rem;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static UInt128 Mul64(this UInt128 a, UInt128 b)
	{
		ulong aLow = (ulong)a;
		ulong bLow = (ulong)b;
		return ((UInt128)(aLow * (ulong)(b >> 64) + aLow.MulHigh(bLow)) << 64) | (aLow * bLow);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static UInt128 MulMod(this UInt128 a, UInt128 b, UInt128 modulus)
	{
		UInt128 zero, result = zero = UInt128.Zero, one = UInt128.One;

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
}