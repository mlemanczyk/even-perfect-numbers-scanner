namespace PerfectNumbers.Core;

public class MersenneNumberLucasLehmerCpuTester
{
    public bool IsPrime(ulong exponent)
	{
		// Early rejections aligned with incremental/order sieves, but safe for small p:
		// - If 3 | p and p != 3, then 7 | M_p -> composite.
		// - If p ≡ 1 (mod 4) and p shares a factor with (p-1), reject fast.
		// - If p is divisible by divisors specific to numbers ending with 1 or 7, reject fast.
                ulong mod3 = exponent.Mod3();
                ulong mod5 = exponent.Mod5();
                ulong mod7 = exponent.Mod7();
                ulong mod11 = exponent.Mod11();
                if (
                    (mod3 == 0UL && exponent != 3UL) ||
                    (mod5 == 0UL && exponent != 5UL) ||
                    (mod7 == 0UL && exponent != 7UL) ||
                    (mod11 == 0UL && exponent != 11UL)
                )
		{
			return false;
		}

		if ((exponent & 3UL) == 1UL && exponent.SharesFactorWithExponentMinusOne())
		{
			return false;
		}

		UInt128 s = UInt128Numbers.Four, two = UInt128Numbers.Two,
			 m = (UInt128.One << (int)exponent) - UInt128.One;

		UInt128 limit = exponent - UInt128Numbers.Two;
		for (UInt128 i = UInt128.Zero; i < limit; i++)
		{
			s = s.PowModWithCycle(two, MersenneDivisorCycles.GetCycle(m)) - two;
			if (s < UInt128.Zero)
			{
				s += m;
			}
		}

		return s == UInt128.Zero;
	}
}
