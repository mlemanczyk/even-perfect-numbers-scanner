using System;

namespace PerfectNumbers.Core;

public static class HeuristicPrimeSieves
{
    private const int SieveLength = PerfectNumberConstants.MaxQForDivisorCycles / 4;
    private const int Wheel210 = 210;

    private static readonly int[] GroupAConstants = [3, 7, 11, 13];
    private static readonly int[] GroupAIncrementPattern = [20, 10];

    private static readonly ushort[] GroupBResiduesEnding1 = [1, 11, 31, 41, 61, 71, 101, 121, 131, 151, 181, 191];
    private static readonly ushort[] GroupBResiduesEnding3 = [13, 23, 43, 53, 73, 83, 103, 113, 143, 163, 173, 193];
    private static readonly ushort[] GroupBResiduesEnding7 = [17, 37, 47, 67, 97, 107, 127, 137, 157, 167, 187, 197];
    private static readonly ushort[] GroupBResiduesEnding9 = [19, 29, 59, 79, 89, 109, 139, 149, 169, 179, 199, 209];

    private static readonly int[] GroupADivisorsStorage;
    private static readonly ulong[] GroupADivisorSquaresStorage;

    private static readonly int[] GroupBDivisorsEnding1Storage;
    private static readonly ulong[] GroupBDivisorSquaresEnding1Storage;

    private static readonly int[] GroupBDivisorsEnding3Storage;
    private static readonly ulong[] GroupBDivisorSquaresEnding3Storage;

    private static readonly int[] GroupBDivisorsEnding7Storage;
    private static readonly ulong[] GroupBDivisorSquaresEnding7Storage;

    private static readonly int[] GroupBDivisorsEnding9Storage;
    private static readonly ulong[] GroupBDivisorSquaresEnding9Storage;

    static HeuristicPrimeSieves()
    {
        Console.WriteLine("[HeuristicPrimeSieves] Initializing heuristic divisor sieves...");

        (GroupADivisorsStorage, GroupADivisorSquaresStorage) = BuildGroupADivisors();
        (GroupBDivisorsEnding1Storage, GroupBDivisorSquaresEnding1Storage) = BuildGroupBResidueSieve(GroupBResiduesEnding1);
        (GroupBDivisorsEnding3Storage, GroupBDivisorSquaresEnding3Storage) = BuildGroupBResidueSieve(GroupBResiduesEnding3);
        (GroupBDivisorsEnding7Storage, GroupBDivisorSquaresEnding7Storage) = BuildGroupBResidueSieve(GroupBResiduesEnding7);
        (GroupBDivisorsEnding9Storage, GroupBDivisorSquaresEnding9Storage) = BuildGroupBResidueSieve(GroupBResiduesEnding9);
    }

    public static void EnsureInitialized()
    {
        // Intentionally left blank. Accessing this method forces the static constructor to run.
    }

    public static ReadOnlySpan<int> GroupADivisors => GroupADivisorsStorage;

    public static ReadOnlySpan<ulong> GroupADivisorSquares => GroupADivisorSquaresStorage;

    public static ReadOnlySpan<int> GroupBDivisorsEnding1 => GroupBDivisorsEnding1Storage;

    public static ReadOnlySpan<ulong> GroupBDivisorSquaresEnding1 => GroupBDivisorSquaresEnding1Storage;

    public static ReadOnlySpan<int> GroupBDivisorsEnding3 => GroupBDivisorsEnding3Storage;

    public static ReadOnlySpan<ulong> GroupBDivisorSquaresEnding3 => GroupBDivisorSquaresEnding3Storage;

    public static ReadOnlySpan<int> GroupBDivisorsEnding7 => GroupBDivisorsEnding7Storage;

    public static ReadOnlySpan<ulong> GroupBDivisorSquaresEnding7 => GroupBDivisorSquaresEnding7Storage;

    public static ReadOnlySpan<int> GroupBDivisorsEnding9 => GroupBDivisorsEnding9Storage;

    public static ReadOnlySpan<ulong> GroupBDivisorSquaresEnding9 => GroupBDivisorSquaresEnding9Storage;

    private static (int[] divisors, ulong[] squares) BuildGroupADivisors()
    {
        var divisors = new int[SieveLength];
        var squares = new ulong[SieveLength];

        int index = 0;
        for (; index < GroupAConstants.Length && index < divisors.Length; index++)
        {
            int value = GroupAConstants[index];
            divisors[index] = value;
            squares[index] = (ulong)value * (ulong)value;
        }

        int candidate = 23;
        int incrementIndex = 0;
        while (index < divisors.Length)
        {
            divisors[index] = candidate;
            squares[index] = (ulong)candidate * (ulong)candidate;

            candidate += GroupAIncrementPattern[incrementIndex];
            incrementIndex ^= 1;
            index++;
        }

        return (divisors, squares);
    }

    private static (int[] divisors, ulong[] squares) BuildGroupBResidueSieve(ReadOnlySpan<ushort> residues)
    {
        var divisors = new int[SieveLength];
        var squares = new ulong[SieveLength];

        int index = 0;
        ulong baseValue = 0UL;

        while (index < divisors.Length)
        {
            for (int i = 0; i < residues.Length && index < divisors.Length; i++)
            {
                ulong candidate = baseValue + residues[i];
                if (candidate <= 13UL)
                {
                    continue;
                }

                divisors[index] = (int)candidate;
                squares[index] = candidate * candidate;
                index++;
            }

            baseValue += (ulong)Wheel210;
        }

        return (divisors, squares);
    }
}
