using System;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public static class PrimesGenerator
{
    public static readonly uint[] SmallPrimes;
    public static readonly ulong[] SmallPrimesPow2;
    public static readonly uint[] SmallPrimesLastOne;
    public static readonly ulong[] SmallPrimesPow2LastOne;
    public static readonly uint[] SmallPrimesLastSeven;
    public static readonly ulong[] SmallPrimesPow2LastSeven;

    static PrimesGenerator()
    {
        BuildSmallPrimes(
            out SmallPrimes,
            out SmallPrimesPow2,
            out SmallPrimesLastOne,
            out SmallPrimesPow2LastOne,
            out SmallPrimesLastSeven,
            out SmallPrimesPow2LastSeven);
    }

    private static void BuildSmallPrimes(
        out uint[] all,
        out ulong[] allPow2,
        out uint[] lastOne,
        out ulong[] lastOnePow2,
        out uint[] lastSeven,
        out ulong[] lastSevenPow2)
    {
        uint target = PerfectNumberConstants.PrimesLimit;
        int targetInt = checked((int)target);
        all = new uint[targetInt];
        allPow2 = new ulong[targetInt];
        lastOne = new uint[targetInt];
        lastOnePow2 = new ulong[targetInt];
        lastSeven = new uint[targetInt];
        lastSevenPow2 = new ulong[targetInt];

        List<uint> primes = new(targetInt * 4 / 3 + 16);
        uint candidate = 2U;
        int allCount = 0;
        int lastOneCount = 0;
        int lastSevenCount = 0;

        while (allCount < targetInt || lastOneCount < targetInt || lastSevenCount < targetInt)
        {
            bool isPrime = true;
            int primesCount = primes.Count;
            for (int i = 0; i < primesCount; i++)
            {
                uint p = primes[i];
                // TODO: Reuse a single squared local for `p` within this loop so we avoid recalculating
                // `p * p` repeatedly without storing every squared prime globally (the cache must remain
                // limited to the small-divisor tables to honor the memory constraints).
                if (p * p > candidate)
                {
                    break;
                }

                // TODO: Replace this trial-division `%` with the sieve-based generator that avoids
                // per-candidate modulo work so building the small-prime tables stops dominating
                // startup time for large scans.
                if (candidate % p == 0U)
                {
                    isPrime = false;
                    break;
                }
            }

            if (isPrime)
            {
                primes.Add(candidate);
                if (allCount < targetInt)
                {
                    // TODO: Reuse a single cached `candidateSquared` per iteration so we avoid issuing
                    // three separate 64-bit multiplications when populating the pow2 arrays here and
                    // in the last-one/last-seven branches below.
                    all[allCount] = candidate;
                    allPow2[allCount] = checked((ulong)candidate * candidate);
                    allCount++;
                }

                if (candidate != 2U)
                {
                    LastDigit lastDigit = GetLastDigit(candidate);

                    if (lastOneCount < targetInt && IsAllowedForLastOne(candidate, lastDigit))
                    {
                        lastOne[lastOneCount] = candidate;
                        lastOnePow2[lastOneCount] = checked((ulong)candidate * candidate);
                        lastOneCount++;
                    }

                    if (lastSevenCount < targetInt && IsAllowedForLastSeven(candidate, lastDigit))
                    {
                        lastSeven[lastSevenCount] = candidate;
                        lastSevenPow2[lastSevenCount] = checked((ulong)candidate * candidate);
                        lastSevenCount++;
                    }
                }
            }

            // TODO: Switch this increment to the Mod6 stepping table identified in the Mod6ComparisonBenchmarks
            // so we skip the composite residues without relying on per-iteration `%` checks.
            candidate = candidate == 2U ? 3U : candidate + 2U;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static LastDigit GetLastDigit(uint prime)
    {
        return (prime % 10U) switch
        {
            1U => LastDigit.One,
            3U => LastDigit.Three,
            5U => LastDigit.Five,
            7U => LastDigit.Seven,
            9U => LastDigit.Nine,
            _ => throw new ArgumentOutOfRangeException(nameof(prime)),
        };
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsAllowedForLastOne(uint prime, LastDigit lastDigit)
    {
        // TODO: Swap the `% 10` usage for ULongExtensions.Mod10 so the hot classification path
        // reuses the benchmarked residue helper instead of repeated divisions.
        return lastDigit switch
        {
            LastDigit.One or LastDigit.Three or LastDigit.Nine => true,
            LastDigit.Seven => prime == 7U,
            _ => prime == 11U,
        };
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsAllowedForLastSeven(uint prime, LastDigit lastDigit)
    {
        // TODO: Route this `% 10` classification through ULongExtensions.Mod10 to match the faster
        // residue helper used elsewhere in the scanner.
        return lastDigit switch
        {
            LastDigit.Three or LastDigit.Seven or LastDigit.Nine => true,
            _ => prime == 11U,
        };
    }
}
