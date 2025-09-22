using System.Numerics;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public static class ULongExtensions
{
	public static ulong CalculateOrder(this ulong q)
	{
		if (q <= 2UL)
		{
			return 0UL;
		}

		ulong order = q - 1UL, prime, temp;
		uint[] smallPrimes = PrimesGenerator.SmallPrimes;
		ulong[] smallPrimesPow2 = PrimesGenerator.SmallPrimesPow2;

		int i = 0, primesLength = smallPrimes.Length;
		UInt128 q128 = q,
				cycle = MersenneDivisorCycles.GetCycle(q128);

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
				if (temp.PowModWithCycle(q128, cycle) == UInt128.One)
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
	public static ulong FastDiv64(this ulong value, ulong divisor, ulong mul)
	{
		ulong q = value.MulHigh(mul);
		if ((UInt128)value - q.Mul64(divisor) >= divisor)
		{
			q++;
		}

		return q;
	}

        private static readonly byte[] Mod6Lookup = { 0, 3, 4, 1, 2, 5 };
        private static readonly byte[] Mod7ByteCoefficients = { 1, 4, 2, 1, 4, 2, 1, 4 };
        private static readonly byte[] Mod11ByteCoefficients = { 1, 3, 9, 5, 4, 1, 3, 9 };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Mod10_8_5_3(this ulong value, out ulong mod10, out ulong mod8, out ulong mod5, out ulong mod3)
        {
                ulong temp = 0UL;
                uint byteSum = 0U;
                byte current = 0;
                uint accumulator = 0U;

                temp = value;
                do
                {
                        current = (byte)temp;
                        byteSum += current;
                        temp >>= 8;
                }
                while (temp != 0UL);

                mod8 = value & 7UL;

                accumulator = byteSum.Mod5();
                mod5 = accumulator;
                mod10 = (mod8 & 1UL) == 0UL
                                ? accumulator switch
                                {
                                        0U => 0UL,
                                        1U => 6UL,
                                        2U => 2UL,
                                        3U => 8UL,
                                        _ => 4UL,
                                }
                                : accumulator switch
                                {
                                        0U => 5UL,
                                        1U => 1UL,
                                        2U => 7UL,
                                        3U => 3UL,
                                        _ => 9UL,
                                };

                // Reusing accumulator to hold the mod 3 reduction now that the mod 5 value has been stored.
                accumulator = byteSum.Mod3();
                mod3 = accumulator;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Mod11_7_5_3(this ulong value, out ulong mod11, out ulong mod7, out ulong mod5, out ulong mod3)
        {
                uint mod3Accumulator = 0U;
                uint mod5Accumulator = 0U;
                uint mod7Accumulator = 0U;
                uint mod11Accumulator = 0U;
                ulong temp = 0UL;
                int index = 0;
                byte current = 0;

                temp = value;
                do
                {
                        current = (byte)temp;
                        mod3Accumulator += current;
                        mod5Accumulator += current;
                        mod7Accumulator += (uint)(current * Mod7ByteCoefficients[index]);
                        mod11Accumulator += (uint)(current * Mod11ByteCoefficients[index]);

                        temp >>= 8;
                        index++;
                }
                while (temp != 0UL);

                // Reusing mod3Accumulator to store the reduced mod 3 remainder before exposing it.
                mod3Accumulator = mod3Accumulator.Mod3();
                // Reusing mod5Accumulator to store the reduced mod 5 remainder before exposing it.
                mod5Accumulator = mod5Accumulator.Mod5();
                // Reusing mod7Accumulator to store the reduced mod 7 remainder before exposing it.
                mod7Accumulator = (uint)ReduceMod7(mod7Accumulator);
                // Reusing mod11Accumulator to store the reduced mod 11 remainder before exposing it.
                mod11Accumulator = (uint)ReduceMod11(mod11Accumulator);

                mod3 = mod3Accumulator;
                mod5 = mod5Accumulator;
                mod7 = mod7Accumulator;
                mod11 = mod11Accumulator;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Mod10_8_5_3Steps(this ulong value, out ulong step10, out ulong step8, out ulong step5, out ulong step3)
        {
                ulong mod10 = 0UL;
                ulong mod8 = 0UL;
                ulong mod5 = 0UL;
                ulong mod3 = 0UL;

                value.Mod10_8_5_3(out mod10, out mod8, out mod5, out mod3);

                step10 = (mod10 << 1).Mod10();
                step8 = ((mod8 << 1) & 7UL);
                step5 = (mod5 << 1).Mod5();
                step3 = (mod3 << 1).Mod3();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsPrimeCandidate(this ulong n)
        {
                int i = 0;
                ulong p = 0UL;
                uint[] smallPrimes = PrimesGenerator.SmallPrimes;
                ulong[] smallPrimesPow2 = PrimesGenerator.SmallPrimesPow2;
                int len = smallPrimes.Length;
                for (; i < len; i++)
                {
                        if (smallPrimesPow2[i] > n)
                        {
                                break;
                        }

                        p = smallPrimes[i];
                        if ((n % p) == 0UL)
                        {
                                return n == p;
                        }
                }

                return true;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Mod10(this ulong value)
        {
                ulong mod5 = 0UL;
                ulong result = 0UL;

                mod5 = value.Mod5();

                if ((value & 1UL) == 0UL)
                {
                        result = mod5 switch
                        {
                                0UL => 0UL,
                                1UL => 6UL,
                                2UL => 2UL,
                                3UL => 8UL,
                                _ => 4UL,
                        };
                }
                else
                {
                        result = mod5 switch
                        {
                                0UL => 5UL,
                                1UL => 1UL,
                                2UL => 7UL,
                                3UL => 3UL,
                                _ => 9UL,
                        };
                }

                return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Mod8(this ulong value) => value & 7UL;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Mod3(this ulong value)
        {
                uint sum = 0U;
                ulong temp = 0UL;

                temp = value & 0xFFFFUL;
                sum = (uint)temp;
                temp = (value >> 16) & 0xFFFFUL;
                sum += (uint)temp;
                temp = (value >> 32) & 0xFFFFUL;
                sum += (uint)temp;
                temp = (value >> 48) & 0xFFFFUL;
                sum += (uint)temp;

                return sum.Mod3();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Mod5(this ulong value)
        {
                uint sum = 0U;
                ulong temp = 0UL;

                temp = value & 0xFFFFUL;
                sum = (uint)temp;
                temp = (value >> 16) & 0xFFFFUL;
                sum += (uint)temp;
                temp = (value >> 32) & 0xFFFFUL;
                sum += (uint)temp;
                temp = (value >> 48) & 0xFFFFUL;
                sum += (uint)temp;

                return sum.Mod5();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Mod6(this ulong value)
        {
                return Mod6Lookup[(int)(((value.Mod3() << 1) | (value & 1UL)))];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Mod7(this ulong value)
        {
                uint low = 0U;
                uint high = 0U;
                ulong remainder = 0UL;

                low = (uint)value;
                high = (uint)(value >> 32);

                remainder = low.Mod7() + (ulong)high.Mod7() * 4UL;
                while (remainder >= 7UL)
                {
                        remainder -= 7UL;
                }

                return remainder;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Mod11(this ulong value)
        {
                uint low = 0U;
                uint high = 0U;
                ulong remainder = 0UL;

                low = (uint)value;
                high = (uint)(value >> 32);

                remainder = low.Mod11() + (ulong)high.Mod11() * 4UL;
                while (remainder >= 11UL)
                {
                        remainder -= 11UL;
                }

                return remainder;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Mod128(this ulong value) => value & 127UL;

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
	public static UInt128 Mul64(this ulong a, ulong b) => ((UInt128)a.MulHigh(b) << 64) | (UInt128)(a * b);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong MulHigh(this ulong x, ulong y)
	{
		ulong xLow = (uint)x;
		ulong xHigh = x >> 32;
		ulong yLow = (uint)y;
		ulong yHigh = y >> 32;

		ulong w1 = xLow * yHigh;
		ulong w2 = xHigh * yLow;
		ulong carry =
			(((xLow * yLow) >> 32) +
			(uint)w1 +
			(uint)w2) >> 32;

		return (xHigh * yHigh) + (w1 >> 32) + (w2 >> 32) + carry;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong ModPow64(this ulong value, ulong exponent, ulong modulus)
	{
		ulong result = 1UL;
		ulong baseValue = value % modulus;

		while (exponent != 0UL)
		{
			if ((exponent & 1UL) != 0UL)
			{
				result = MulMod64(result, baseValue, modulus);
			}

			baseValue = MulMod64(baseValue, baseValue, modulus);
			exponent >>= 1;
		}

		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong MulMod64(this ulong a, ulong b, ulong modulus) => (ulong)(UInt128)(((a % modulus) * (b % modulus)) % modulus);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt128 PowMod(this ulong exponent, UInt128 modulus)
        {
                UInt128 result = UInt128.One;
                ulong exponentLoopIndex = 0UL;

                // Return 1 because 2^0 = 1
                if (exponent == 0UL)
                        return result;

                // Any number mod 1 is 0
                if (modulus == UInt128.One)
                        return UInt128.Zero;

                // For small exponents, do classic method
                if (exponent < 64 || modulus < 4)
                {
                        for (; exponentLoopIndex < exponent; exponentLoopIndex++)
                        {
                                result <<= 1;
                                if (result >= modulus)
                                        result -= modulus;
                        }

                        return result;
                }

                // Special case: if modulus is a power of two, use bitmasking for efficiency and correctness
                if ((modulus & (modulus - 1)) == 0)
                {
                        result = UInt128.One << (int)(exponent & 127);
                        return result & (modulus - 1);
                }

                // Reusing exponentLoopIndex to iterate again over the exponent range for the general-case accumulation.
                exponentLoopIndex = 0UL;
                // Reusing result after resetting it for the general modulus accumulation phase.
                result = UInt128.One;
                for (; exponentLoopIndex < exponent; exponentLoopIndex++)
                {
                        result <<= 1;
                        if (result >= modulus)
                                result -= modulus;
                }

                return result;
        }

	/// <summary>
	/// Computes 2^exponent mod modulus using a known cycle length.
	/// </summary>
        public static UInt128 PowModWithCycle(this ulong exponent, UInt128 modulus, ulong cycleLength)
        {
                UInt128 result = UInt128.One;
                ulong exponentLoopIndex = 0UL;
                ulong rotationCount = 0UL;

                // Return 1 because 2^0 = 1
                if (exponent == 0UL)
                        return result;

                // Any number mod 1 is 0
                if (modulus == UInt128.One)
                        return UInt128.Zero;

                // For small exponents, do classic method
                if (exponent < 64 || modulus < 4)
                {
                        for (; exponentLoopIndex < exponent; exponentLoopIndex++)
                        {
                                result <<= 1;
                                if (result >= modulus)
                                        result -= modulus;
                        }

                        return result;
                }

                // Special case: if modulus is a power of two, use bitmasking for efficiency and correctness
                if ((modulus & (modulus - 1)) == 0)
                {
                        result = UInt128.One << (int)(exponent & 127);
                        return result & (modulus - 1);
                }

                // Reusing exponentLoopIndex to iterate over the rotation count for the cycle-aware path.
                exponentLoopIndex = 0UL;
                // Reusing result after resetting it for the rotation-accumulation pass.
                result = UInt128.One;
                rotationCount = exponent % cycleLength;
                for (; exponentLoopIndex < rotationCount; exponentLoopIndex++)
                {
                        result <<= 1;
                        if (result >= modulus)
                                result -= modulus;
                }

                return result;
        }

	/// <summary>
	/// Computes 2^exponent mod modulus using a known cycle length.
	/// </summary>
        public static UInt128 PowModWithCycle(this ulong exponent, UInt128 modulus, UInt128 cycleLength)
        {
                UInt128 result = UInt128.One;
                ulong exponentLoopIndex = 0UL;
                UInt128 rotationIndex = UInt128.Zero;
                UInt128 rotationCount = UInt128.Zero;

                // Return 1 because 2^0 = 1
                if (exponent == UInt128.Zero)
                        return result;

                // Any number mod 1 is 0
                if (modulus == UInt128.One)
                        return UInt128.Zero;

                // For small exponents, do classic method
                if (exponent < UInt128Numbers.SixtyFour || modulus < UInt128Numbers.Four)
                {
                        for (; exponentLoopIndex < exponent; exponentLoopIndex++)
                        {
                                result <<= 1;
                                if (result >= modulus)
                                        result -= modulus;
                        }

                        return result;
                }

                // Special case: if modulus is a power of two, use bitmasking for efficiency and correctness
                if ((modulus & (modulus - UInt128.One)) == UInt128.Zero)
                {
                        result = UInt128.One << (int)(exponent & UInt128Numbers.OneHundredTwentySeven);
                        return result & (modulus - 1);
                }

                // Reusing result after resetting it for the rotation-driven accumulation phase.
                result = UInt128.One;
                rotationCount = exponent % cycleLength;
                rotationIndex = UInt128.Zero;
                while (rotationIndex < rotationCount)
                {
                        result <<= 1;
                        if (result >= modulus)
                                result -= modulus;

                        rotationIndex += UInt128.One;
                }

                return result;
        }

	/// <summary>
	/// Computes 2^exponent mod modulus using a known cycle length.
	/// </summary>
        public static UInt128 PowModWithCycle(this UInt128 exponent, UInt128 modulus, UInt128 cycleLength)
        {
                UInt128 result = UInt128.One;
                ulong exponentLoopIndex = 0UL;
                UInt128 rotationIndex = UInt128.Zero;
                UInt128 rotationCount = UInt128.Zero;

                // Return 1 because 2^0 = 1
                if (exponent == UInt128.Zero)
                        return result;

                // Any number mod 1 is 0
                if (modulus == UInt128.One)
                        return UInt128.Zero;

                // For small exponents, do classic method
                if (exponent < UInt128Numbers.SixtyFour || modulus < UInt128Numbers.Four)
                {
                        for (; exponentLoopIndex < exponent; exponentLoopIndex++)
                        {
                                result <<= 1;
                                if (result >= modulus)
                                        result -= modulus;
                        }

                        return result;
                }

                // Special case: if modulus is a power of two, use bitmasking for efficiency and correctness
                if ((modulus & (modulus - UInt128.One)) == UInt128.Zero)
                {
                        result = UInt128.One << (int)(exponent & UInt128Numbers.OneHundredTwentySeven);
                        return result & (modulus - 1);
                }

                // Reusing result after resetting it for the rotation-driven accumulation phase.
                result = UInt128.One;
                rotationCount = exponent % cycleLength;
                rotationIndex = UInt128.Zero;
                while (rotationIndex < rotationCount)
                {
                        result <<= 1;
                        if (result >= modulus)
                                result -= modulus;

                        rotationIndex += UInt128.One;
                }

                return result;
        }

	/// <summary>
	/// Computes 2^exponent mod modulus using iterative CRT composition from mod 10 up to modulus.
	/// Only for modulus >= 10 and reasonable size.
	/// </summary>
        public static UInt128 PowModCrt(this ulong exponent, UInt128 modulus, MersenneDivisorCycles cycles)
        {
                if (modulus < 10)
                        return PowMod(exponent, modulus); // fallback to classic

                // Use cycle length 4 for mod 10
                UInt128 result = PowModWithCycle(exponent, 10, 4);
                UInt128 currentModulus = 10;
                UInt128 modulusCandidate = 11;
                UInt128 cycle = UInt128.Zero;
                UInt128 remainderForCandidate = UInt128.Zero;

                for (; modulusCandidate <= modulus; modulusCandidate++)
                {
                        cycle = MersenneDivisorCycles.GetCycle(modulusCandidate);
                        remainderForCandidate = cycle > UInt128.Zero
                                ? PowModWithCycle(exponent, modulusCandidate, cycle)
                                : PowMod(exponent, modulusCandidate);

                        // Solve x ≡ result mod currentModulus
                        //      x ≡ remM   mod m
                        // Find x mod (currentModulus * m)
                        // Since currentModulus and m are coprime, use CRT:
                        // x = result + currentModulus * t, where t ≡ (remM - result) * inv(currentModulus, m) mod m

                        result += currentModulus * ((remainderForCandidate + modulusCandidate - (result % modulusCandidate)) * ModInverse(currentModulus, modulusCandidate) % modulusCandidate);
                        currentModulus *= modulusCandidate;

                        if (currentModulus >= modulus)
                                break;
                }

                return result % modulus;
        }

        // Helper: modular inverse (extended Euclidean algorithm)
        private static UInt128 ModInverse(UInt128 a, UInt128 m)
        {
                UInt128 m0 = m, temp;
                UInt128 x0 = 0, x1 = 1;
                UInt128 originalA = UInt128.Zero;
                UInt128 originalM = UInt128.Zero;
                if (m == 1)
                {
                        return 0;
                }

                while (a > 1)
                {
                        originalA = a;
                        originalM = m;
                        m = originalA % originalM;
                        a = originalM;
                        temp = x0;
                        x0 = x1 - (originalA / originalM) * x0;
                        x1 = temp;
                }

                if (x1 < 0)
                {
                        x1 += m0;
                }

                return x1;
        }

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool SharesFactorWithExponentMinusOne(this ulong exponent)
	{
		ulong value = exponent - 1UL;
		value >>= BitOperations.TrailingZeroCount(value);
		uint[] smallPrimes = PrimesGenerator.SmallPrimes;
		ulong[] smallPrimesPow2 = PrimesGenerator.SmallPrimesPow2;
		int i = 1, smallPrimesLength = smallPrimes.Length;
		ulong prime;

		for (; i < smallPrimesLength && smallPrimesPow2[i] <= value; i++)
		{
			prime = smallPrimes[i];
			if (value % prime != 0UL)
			{
				continue;
			}

			if (exponent % prime.CalculateOrder() == 0UL)
			{
				return true;
			}

			do
			{
				value /= prime;
			}
			while (value % prime == 0UL);
		}

		if (value > 1UL && exponent % value.CalculateOrder() == 0UL)
		{
			return true;
		}

		return false;
	}
}
