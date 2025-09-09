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
                if (powers[i].IsZero)
                    powers[i] = powers[i - 1] * baseVal;
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
                if (powers[i] == null)
                    powers[i] = powers[i - 1]! * baseVal;
        }
        return powers[exponent]!;
    }
}
