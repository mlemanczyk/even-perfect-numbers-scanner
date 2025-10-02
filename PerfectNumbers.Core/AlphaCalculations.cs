using PeterO.Numbers;

namespace PerfectNumbers.Core;

public static class AlphaCalculations
{
    public static ERational ComputeAlphaP(EInteger p, int k)
    {
        int exponent = checked(4 * k + 1);
        EInteger y = PowerCache.Get(p, exponent);
        // TODO: Reuse pooled numerator/denominator buffers here so the alphaP computation stops allocating
        // intermediate EInteger instances every invocation.
        return ERational.Create(y * p - 1, y * (p - 1));
    }

    public static ERational ComputeAlphaFactor(EInteger q, int a2)
    {
        EInteger num = PowerCache.Get(q, a2 + 1) - EInteger.One;
        EInteger den = PowerCache.Get(q, a2) * (q - EInteger.One);
        // TODO: Reuse pooled ERational builders here so repeated factor computations pull precomputed
        // numerator/denominator tuples instead of re-executing PowerCache chains each time, matching the
        // fastest alpha benchmarks.
        return ERational.Create(num, den);
    }

    public static ERational ComputeAlphaM(ReadOnlySpan<EInteger> primes, int maxR, int a2)
    {
        ERational result = ERational.One;
        for (int i = 0; i < maxR; i++)
        {
            // TODO: Replace the repeated Multiply calls with a span-based aggregator that batches factors,
            // avoiding transient ERational instances when maxR grows large.
            result = result.Multiply(ComputeAlphaFactor(primes[i], a2));
        }
        return result;
    }
}

