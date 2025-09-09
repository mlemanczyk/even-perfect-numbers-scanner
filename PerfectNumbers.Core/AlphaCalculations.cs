using PeterO.Numbers;

namespace PerfectNumbers.Core;

public static class AlphaCalculations
{
    public static ERational ComputeAlphaP(EInteger p, int k)
    {
        int exponent = checked(4 * k + 1);
        EInteger y = PowerCache.Get(p, exponent);
        return ERational.Create(y * p - 1, y * (p - 1));
    }

    public static ERational ComputeAlphaFactor(EInteger q, int a2)
    {
        EInteger num = PowerCache.Get(q, a2 + 1) - EInteger.One;
        EInteger den = PowerCache.Get(q, a2) * (q - EInteger.One);
        return ERational.Create(num, den);
    }

    public static ERational ComputeAlphaM(ReadOnlySpan<EInteger> primes, int maxR, int a2)
    {
        ERational result = ERational.One;
        for (int i = 0; i < maxR; i++)
            result = result.Multiply(ComputeAlphaFactor(primes[i], a2));
        return result;
    }
}

