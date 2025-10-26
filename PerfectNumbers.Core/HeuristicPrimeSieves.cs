using System;

namespace PerfectNumbers.Core;

public static class HeuristicPrimeSieves
{
    private const int SieveLength = PerfectNumberConstants.MaxQForDivisorCycles;
    private const int Wheel210 = 210;

    private static readonly int[] GroupAConstants = [3, 7, 11, 13];
    private static readonly int[] GroupAIncrementPattern = [20, 10];

    private static readonly ushort[] GroupBResiduesEnding1 = [1, 11, 31, 41, 61, 71, 101, 121, 131, 151, 181, 191];
    private static readonly ushort[] GroupBResiduesEnding3 = [13, 23, 43, 53, 73, 83, 103, 113, 143, 163, 173, 193];
    private static readonly ushort[] GroupBResiduesEnding7 = [17, 37, 47, 67, 97, 107, 127, 137, 157, 167, 187, 197];
    private static readonly ushort[] GroupBResiduesEnding9 = [19, 29, 59, 79, 89, 109, 139, 149, 169, 179, 199, 209];

    private static int[]? s_groupADivisors;
    private static ulong[]? s_groupADivisorSquares;

    private static int[]? s_groupBDivisorsEnding1;
    private static ulong[]? s_groupBDivisorSquaresEnding1;

    private static int[]? s_groupBDivisorsEnding3;
    private static ulong[]? s_groupBDivisorSquaresEnding3;

    private static int[]? s_groupBDivisorsEnding7;
    private static ulong[]? s_groupBDivisorSquaresEnding7;

    private static int[]? s_groupBDivisorsEnding9;
    private static ulong[]? s_groupBDivisorSquaresEnding9;

    private static readonly object InitLock = new();

    public static ReadOnlySpan<int> GroupADivisors
    {
        get
        {
            EnsureGroupAInitialized();
            return s_groupADivisors!;
        }
    }

    public static ReadOnlySpan<ulong> GroupADivisorSquares
    {
        get
        {
            EnsureGroupAInitialized();
            return s_groupADivisorSquares!;
        }
    }

    public static ReadOnlySpan<int> GroupBDivisorsEnding1
    {
        get
        {
            EnsureGroupBInitialized(ref s_groupBDivisorsEnding1, ref s_groupBDivisorSquaresEnding1, GroupBResiduesEnding1);
            return s_groupBDivisorsEnding1!;
        }
    }

    public static ReadOnlySpan<ulong> GroupBDivisorSquaresEnding1
    {
        get
        {
            EnsureGroupBInitialized(ref s_groupBDivisorsEnding1, ref s_groupBDivisorSquaresEnding1, GroupBResiduesEnding1);
            return s_groupBDivisorSquaresEnding1!;
        }
    }

    public static ReadOnlySpan<int> GroupBDivisorsEnding3
    {
        get
        {
            EnsureGroupBInitialized(ref s_groupBDivisorsEnding3, ref s_groupBDivisorSquaresEnding3, GroupBResiduesEnding3);
            return s_groupBDivisorsEnding3!;
        }
    }

    public static ReadOnlySpan<ulong> GroupBDivisorSquaresEnding3
    {
        get
        {
            EnsureGroupBInitialized(ref s_groupBDivisorsEnding3, ref s_groupBDivisorSquaresEnding3, GroupBResiduesEnding3);
            return s_groupBDivisorSquaresEnding3!;
        }
    }

    public static ReadOnlySpan<int> GroupBDivisorsEnding7
    {
        get
        {
            EnsureGroupBInitialized(ref s_groupBDivisorsEnding7, ref s_groupBDivisorSquaresEnding7, GroupBResiduesEnding7);
            return s_groupBDivisorsEnding7!;
        }
    }

    public static ReadOnlySpan<ulong> GroupBDivisorSquaresEnding7
    {
        get
        {
            EnsureGroupBInitialized(ref s_groupBDivisorsEnding7, ref s_groupBDivisorSquaresEnding7, GroupBResiduesEnding7);
            return s_groupBDivisorSquaresEnding7!;
        }
    }

    public static ReadOnlySpan<int> GroupBDivisorsEnding9
    {
        get
        {
            EnsureGroupBInitialized(ref s_groupBDivisorsEnding9, ref s_groupBDivisorSquaresEnding9, GroupBResiduesEnding9);
            return s_groupBDivisorsEnding9!;
        }
    }

    public static ReadOnlySpan<ulong> GroupBDivisorSquaresEnding9
    {
        get
        {
            EnsureGroupBInitialized(ref s_groupBDivisorsEnding9, ref s_groupBDivisorSquaresEnding9, GroupBResiduesEnding9);
            return s_groupBDivisorSquaresEnding9!;
        }
    }

    private static void EnsureGroupAInitialized()
    {
        if (s_groupADivisors is not null)
        {
            return;
        }

        lock (InitLock)
        {
            if (s_groupADivisors is not null)
            {
                return;
            }

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

            s_groupADivisors = divisors;
            s_groupADivisorSquares = squares;
        }
    }

    private static void EnsureGroupBInitialized(ref int[]? divisorsField, ref ulong[]? squaresField, ReadOnlySpan<ushort> residues)
    {
        if (divisorsField is not null)
        {
            return;
        }

        lock (InitLock)
        {
            if (divisorsField is not null)
            {
                return;
            }

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

            divisorsField = divisors;
            squaresField = squares;
        }
    }
}
