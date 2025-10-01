namespace PerfectNumbers.Core;

public class MersenneNumberLucasLehmerCpuTester
{
    public bool IsPrime(ulong exponent)
    {
        // Early rejections aligned with incremental/order sieves, but safe for small p:
        // - If 3 | p and p != 3, then 7 | M_p -> composite.
        // - If p ≡ 1 (mod 4) and p shares a factor with (p-1), reject fast.
        // - If p is divisible by divisors specific to numbers ending with 1 or 7, reject fast.
        // TODO: Replace these `%` checks with Mod3/Mod5/Mod7/Mod11 helpers once Lucas–Lehmer CPU filtering
        // shares the benchmarked bitmask implementations and avoids slow modulo instructions in the hot path.
        if (
            ((exponent % 3UL) == 0UL && exponent != 3UL) ||
            ((exponent % 5UL) == 0UL && exponent != 5UL) ||
            ((exponent % 7UL) == 0UL && exponent != 7UL) ||
            ((exponent % 11UL) == 0UL && exponent != 11UL)
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
            // TODO: Port this Lucas–Lehmer powmod to the ProcessEightBitWindows helper so the CPU
            // fallback benefits from the same windowed ladder that halves runtime in the Pow2 benchmarks.
            s = s.PowModWithCycle(two, MersenneDivisorCycles.GetCycle(m)) - two;
            if (s < UInt128.Zero)
            {
                s += m;
            }
        }

        return s == UInt128.Zero;
    }
}
