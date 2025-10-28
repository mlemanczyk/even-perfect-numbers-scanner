using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

internal static class DivisorGenerator
{
    public static readonly uint[] SmallPrimes;
    public static readonly ulong[] SmallPrimesPow2;
    public static readonly uint[] SmallPrimesLastOne;
    public static readonly ulong[] SmallPrimesPow2LastOne;
    public static readonly uint[] SmallPrimesLastSeven;
    public static readonly ulong[] SmallPrimesPow2LastSeven;
    public static readonly uint[] SmallPrimesLastThree;
    public static readonly ulong[] SmallPrimesPow2LastThree;
    public static readonly uint[] SmallPrimesLastNine;
    public static readonly ulong[] SmallPrimesPow2LastNine;

    private const ushort DecimalMaskWhenLastIsSeven = (1 << 3) | (1 << 7) | (1 << 9);
    private const ushort DecimalMaskWhenLastIsThree = (1 << 1) | (1 << 3) | (1 << 7) | (1 << 9);
    private const ushort DecimalMaskWhenLastIsNine = (1 << 1) | (1 << 3) | (1 << 7) | (1 << 9);
    private const ushort DecimalMaskOtherwise = (1 << 1) | (1 << 3) | (1 << 9);

    static DivisorGenerator()
    {
        BuildSmallPrimes(
            out SmallPrimes,
            out SmallPrimesPow2,
            out SmallPrimesLastOne,
            out SmallPrimesPow2LastOne,
            out SmallPrimesLastSeven,
            out SmallPrimesPow2LastSeven,
            out SmallPrimesLastThree,
            out SmallPrimesPow2LastThree,
            out SmallPrimesLastNine,
            out SmallPrimesPow2LastNine);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ushort GetDecimalMask(LastDigit lastDigit)
    {
        return lastDigit switch
        {
            LastDigit.Seven => DecimalMaskWhenLastIsSeven,
            LastDigit.Three => DecimalMaskWhenLastIsThree,
            LastDigit.Nine => DecimalMaskWhenLastIsNine,
            _ => DecimalMaskOtherwise,
        };
    }

    private static void BuildSmallPrimes(
        out uint[] all,
        out ulong[] allPow2,
        out uint[] lastOne,
        out ulong[] lastOnePow2,
        out uint[] lastSeven,
        out ulong[] lastSevenPow2,
        out uint[] lastThree,
        out ulong[] lastThreePow2,
        out uint[] lastNine,
        out ulong[] lastNinePow2)
    {
        uint target = PerfectNumberConstants.PrimesLimit;
        int targetInt = checked((int)target);
        all = new uint[targetInt];
        allPow2 = new ulong[targetInt];
        lastOne = new uint[targetInt];
        lastOnePow2 = new ulong[targetInt];
        lastSeven = new uint[targetInt];
        lastSevenPow2 = new ulong[targetInt];
        lastThree = new uint[targetInt];
        lastThreePow2 = new ulong[targetInt];
        lastNine = new uint[targetInt];
        lastNinePow2 = new ulong[targetInt];

        List<uint> primes = new(targetInt * 4 / 3 + 16);
        uint candidate = 2U;
        int allCount = 0;
        int lastOneCount = 0;
        int lastSevenCount = 0;
        int lastThreeCount = 0;
        int lastNineCount = 0;

        while (allCount < targetInt || lastOneCount < targetInt || lastSevenCount < targetInt || lastThreeCount < targetInt || lastNineCount < targetInt)
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

                    if (lastThreeCount < targetInt && IsAllowedForLastThree(candidate, lastDigit))
                    {
                        lastThree[lastThreeCount] = candidate;
                        lastThreePow2[lastThreeCount] = checked((ulong)candidate * candidate);
                        lastThreeCount++;
                    }

                    if (lastNineCount < targetInt && IsAllowedForLastNine(candidate, lastDigit))
                    {
                        lastNine[lastNineCount] = candidate;
                        lastNinePow2[lastNineCount] = checked((ulong)candidate * candidate);
                        lastNineCount++;
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsAllowedForLastThree(uint prime, LastDigit lastDigit)
    {
        // TODO: Replace the `% 10` residue classification with ULongExtensions.Mod10 once the helper
        // migrates to UInt32 so we can reuse the optimized decimal residue path everywhere.
        return lastDigit switch
        {
            LastDigit.Three or LastDigit.Seven => true,
            LastDigit.One => prime == 11U,
            LastDigit.Nine => prime == 19U,
            _ => false,
        };
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsAllowedForLastNine(uint prime, LastDigit lastDigit)
    {
        // TODO: Replace the `% 10` residue classification with ULongExtensions.Mod10 when the helper
        // supports UInt32 inputs so the residue tables share the optimized decimal path.
        return lastDigit switch
        {
            LastDigit.Three or LastDigit.Seven or LastDigit.Nine => true,
            LastDigit.One => prime == 11U,
            _ => false,
        };
    }
}
