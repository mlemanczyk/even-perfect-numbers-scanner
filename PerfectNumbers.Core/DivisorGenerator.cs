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
    public static readonly uint[] SmallPrimesLastOneWithoutLastThree;
    public static readonly ulong[] SmallPrimesPow2LastOneWithoutLastThree;
    public static readonly uint[] SmallPrimesLastSevenWithoutLastThree;
    public static readonly ulong[] SmallPrimesPow2LastSevenWithoutLastThree;
    public static readonly uint[] SmallPrimesLastNineWithoutLastThree;
    public static readonly ulong[] SmallPrimesPow2LastNineWithoutLastThree;

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
            out SmallPrimesPow2LastNine,
            out SmallPrimesLastOneWithoutLastThree,
            out SmallPrimesPow2LastOneWithoutLastThree,
            out SmallPrimesLastSevenWithoutLastThree,
            out SmallPrimesPow2LastSevenWithoutLastThree,
            out SmallPrimesLastNineWithoutLastThree,
            out SmallPrimesPow2LastNineWithoutLastThree);
    }

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ushort GetDecimalMask(LastDigit lastDigit) => lastDigit switch
	{
		LastDigit.Seven => DecimalMaskWhenLastIsSeven,
		LastDigit.Three => DecimalMaskWhenLastIsThree,
		LastDigit.Nine => DecimalMaskWhenLastIsNine,
		_ => DecimalMaskOtherwise,
	};

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static ushort GetDecimalMask(bool lastIsSeven) => lastIsSeven ? DecimalMaskWhenLastIsSeven : DecimalMaskOtherwise;

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
        out ulong[] lastNinePow2,
        out uint[] lastOneWithoutThree,
        out ulong[] lastOneWithoutThreePow2,
        out uint[] lastSevenWithoutThree,
        out ulong[] lastSevenWithoutThreePow2,
        out uint[] lastNineWithoutThree,
        out ulong[] lastNineWithoutThreePow2)
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
        List<uint> lastOneWithoutThreeList = new(targetInt);
        List<ulong> lastOneWithoutThreePow2List = new(targetInt);
        List<uint> lastSevenWithoutThreeList = new(targetInt);
        List<ulong> lastSevenWithoutThreePow2List = new(targetInt);
        List<uint> lastNineWithoutThreeList = new(targetInt);
        List<ulong> lastNineWithoutThreePow2List = new(targetInt);

        uint candidate = 2U;
        int allCount = 0;
        int lastOneCount = 0;
        int lastSevenCount = 0;
        int lastThreeCount = 0;
        int lastNineCount = 0;

        while (allCount < targetInt ||
            lastOneCount < targetInt ||
            lastSevenCount < targetInt ||
            lastThreeCount < targetInt ||
            lastNineCount < targetInt)
        {
            bool isPrime = true;
            int primesCount = primes.Count;
            for (int i = 0; i < primesCount; i++)
            {
                uint p = primes[i];
                if (p * p > candidate)
                {
                    break;
                }

                if (candidate % p == 0U)
                {
                    isPrime = false;
                    break;
                }
            }

            if (isPrime)
            {
                primes.Add(candidate);
                ulong candidateSquared = checked((ulong)candidate * candidate);
                if (allCount < targetInt)
                {
                    all[allCount] = candidate;
                    allPow2[allCount] = candidateSquared;
                    allCount++;
                }

                if (candidate != 2U)
                {
                    LastDigit lastDigit = GetLastDigit(candidate);
                    bool allowedForLastOne = IsAllowedForLastOne(candidate, lastDigit);
                    bool allowedForLastSeven = IsAllowedForLastSeven(candidate, lastDigit);
                    bool allowedForLastThree = IsAllowedForLastThree(candidate, lastDigit);
                    bool allowedForLastNine = IsAllowedForLastNine(candidate, lastDigit);
                    bool lastDigitIsThree = lastDigit == LastDigit.Three;

                    if (allowedForLastOne && lastOneCount < targetInt)
                    {
                        lastOne[lastOneCount] = candidate;
                        lastOnePow2[lastOneCount] = candidateSquared;
                        lastOneCount++;
                    }

                    if (allowedForLastSeven && lastSevenCount < targetInt)
                    {
                        lastSeven[lastSevenCount] = candidate;
                        lastSevenPow2[lastSevenCount] = candidateSquared;
                        lastSevenCount++;
                    }

                    if (allowedForLastThree && lastThreeCount < targetInt)
                    {
                        lastThree[lastThreeCount] = candidate;
                        lastThreePow2[lastThreeCount] = candidateSquared;
                        lastThreeCount++;
                    }

                    if (allowedForLastNine && lastNineCount < targetInt)
                    {
                        lastNine[lastNineCount] = candidate;
                        lastNinePow2[lastNineCount] = candidateSquared;
                        lastNineCount++;
                    }

                    if (!lastDigitIsThree && allowedForLastOne)
                    {
                        lastOneWithoutThreeList.Add(candidate);
                        lastOneWithoutThreePow2List.Add(candidateSquared);
                    }

                    if (!lastDigitIsThree && allowedForLastSeven)
                    {
                        lastSevenWithoutThreeList.Add(candidate);
                        lastSevenWithoutThreePow2List.Add(candidateSquared);
                    }

                    if (!lastDigitIsThree && allowedForLastNine)
                    {
                        lastNineWithoutThreeList.Add(candidate);
                        lastNineWithoutThreePow2List.Add(candidateSquared);
                    }
                }
            }

            candidate = candidate == 2U ? 3U : candidate + 2U;
        }

        lastOneWithoutThree = lastOneWithoutThreeList.ToArray();
        lastOneWithoutThreePow2 = lastOneWithoutThreePow2List.ToArray();
        lastSevenWithoutThree = lastSevenWithoutThreeList.ToArray();
        lastSevenWithoutThreePow2 = lastSevenWithoutThreePow2List.ToArray();
        lastNineWithoutThree = lastNineWithoutThreeList.ToArray();
        lastNineWithoutThreePow2 = lastNineWithoutThreePow2List.ToArray();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static LastDigit GetLastDigit(uint prime)
    {
        // TODO: Swap the `% 10` usage for ULongExtensions.Mod10 so the hot classification path
        // reuses the benchmarked residue helper instead of repeated divisions.
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

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static bool IsAllowedForLastOne(uint prime, LastDigit lastDigit) => lastDigit switch
	{
		LastDigit.One or LastDigit.Three or LastDigit.Nine => true,
		LastDigit.Seven => prime == 7U,
		_ => prime == 11U,
	};

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
