using PeterO.Numbers;

namespace PerfectNumbers.Core;

public static class KRangeFinder
{
    public static (int? Min, int? Max) FindForPrime(ERational alphaM, EInteger p, int kStart, int kEnd)
    {
        int? min = null;
        int? max = null;
        for (int k = kStart; k <= kEnd; k++)
        {
            ERational aP = AlphaCalculations.ComputeAlphaP(p, k);
            ERational prod = aP.Multiply(alphaM);
            int cmp = prod.CompareTo(RationalNumbers.Two);
            if (cmp == 0)
            {
                if (min == null || k < min)
                {
                    min = k;
                }
                if (max == null || k > max)
                {
                    max = k;
                }
            }
            if (cmp < 0)
            {
                break;
            }
        }
        return (min, max);
    }

    public static bool TryFindForPrime(ERational alphaM, EInteger p, out int? min, out int? max, int kStart, int kEnd)
    {
        (min, max) = FindForPrime(alphaM, p, kStart, kEnd);
        return min.HasValue && max.HasValue && min.Value <= max.Value;
    }

    public static (int? Min, int? Max) Find(ERational alphaM, EInteger pMin, EInteger pMax, int kStart, int kEnd, PrimeCache primes)
    {
        int? min = null;
        int? max = null;
        EInteger[] eulerPrimes = primes.GetEulerPrimes(pMin, pMax);
        object sync = new();
        Parallel.For(
            0,
            eulerPrimes.Length,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            i =>
            {
                (int? localMin, int? localMax) = FindForPrime(alphaM, eulerPrimes[i], kStart, kEnd);
                if (localMin == null)
                {
                    return;
                }
                lock (sync)
                {
                    if (min == null || localMin < min)
                    {
                        min = localMin;
                    }
                    if (max == null || localMax > max)
                    {
                        max = localMax;
                    }
                }
            });
        return (min, max);
    }

    public static bool TryFind(ERational alphaM, EInteger pMin, EInteger pMax, out int? min, out int? max, int kStart, int kEnd, PrimeCache primes)
    {
        (min, max) = Find(alphaM, pMin, pMax, kStart, kEnd, primes);
        return min.HasValue && max.HasValue && min.Value <= max.Value;
    }
}

