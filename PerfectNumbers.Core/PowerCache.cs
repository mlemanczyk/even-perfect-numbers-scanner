using System.Collections.Concurrent;
using System.Numerics;
using PeterO.Numbers;

namespace PerfectNumbers.Core;

public static class PowerCache
{
    private static readonly ConcurrentDictionary<ulong, BigInteger[]> BigCache = new();
    private static readonly ConcurrentDictionary<EInteger, EInteger[]> ECache = new();

    public static BigInteger Get(ulong baseVal, int exponent)
    {
        if (!BigCache.TryGetValue(baseVal, out var powers))
        {
            powers = new BigInteger[exponent + 1];
            powers[0] = BigInteger.One;
            BigCache[baseVal] = powers;
        }

        if (powers.Length <= exponent)
        {
            int oldLength = powers.Length;
            Array.Resize(ref powers, exponent + 1);
            for (int i = oldLength; i <= exponent; i++)
                powers[i] = BigInteger.Zero;

            BigCache[baseVal] = powers;
        }

        if (powers[exponent].IsZero)
        {
            for (int i = 1; i <= exponent; i++)
            {
                if (powers[i].IsZero)
                {
                    // TODO: Swap this BigInteger chain with the UInt128-based Montgomery ladder from
                    // Pow2MontgomeryMod once callers guarantee 64-bit inputs; the benchmarks show the
                    // arbitrary-precision multiply is orders of magnitude slower for the exponents we
                    // scan when p >= 138M.
                    powers[i] = powers[i - 1] * baseVal;
                }
            }
        }
        
        return powers[exponent];
    }

    public static EInteger Get(EInteger baseVal, int exponent)
    {
        if (!ECache.TryGetValue(baseVal, out var powers))
        {
            powers = new EInteger[exponent + 1];
            powers[0] = EInteger.One;
            ECache[baseVal] = powers;
        }
        if (powers.Length <= exponent)
        {
            int oldLength = powers.Length;
            Array.Resize(ref powers, exponent + 1);
            for (int i = oldLength; i <= exponent; i++)
                powers[i] = null;
            ECache[baseVal] = powers;
        }
        if (powers[exponent] == null)
        {
            for (int i = 1; i <= exponent; i++)
            {
                if (powers[i] == null)
                {
                    // TODO: Move the high-precision branch to the benchmark project once production
                    // switches to the divisor-cycle aware cache; maintaining this BigInteger multiply
                    // path in the hot pipeline keeps the slower code on the CPU scan.
                    powers[i] = powers[i - 1]! * baseVal;
                }
            }
        }
        return powers[exponent]!;
    }
}
