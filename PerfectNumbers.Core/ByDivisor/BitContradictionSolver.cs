using System;

namespace PerfectNumbers.Core.ByDivisor;

/// <summary>
/// Helper utilities for the BitContradiction divisor search. These methods implement the
/// interval-based carry propagation described in the BitContradictions plan document.
/// The solver itself will build on top of these primitives.
/// </summary>
internal static class BitContradictionSolver
{
    internal enum BitState
    {
        Zero = 0,
        One = 1,
        Unknown = 2,
    }

    internal readonly struct ColumnBounds
    {
        public readonly long ForcedOnes;
        public readonly long PossibleOnes;

        public ColumnBounds(long forcedOnes, long possibleOnes)
        {
            if (forcedOnes < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(forcedOnes));
            }

            if (possibleOnes < forcedOnes)
            {
                throw new ArgumentOutOfRangeException(nameof(possibleOnes), "possibleOnes must be >= forcedOnes.");
            }

            ForcedOnes = forcedOnes;
            PossibleOnes = possibleOnes;
        }
    }

    internal readonly struct CarryRange
    {
        public readonly long Min;
        public readonly long Max;

        public CarryRange(long min, long max)
        {
            if (min < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(min));
            }

            if (max < min)
            {
                throw new ArgumentOutOfRangeException(nameof(max), "max must be >= min.");
            }

            Min = min;
            Max = max;
        }

        public static CarryRange Single(long value) => new(value, value);
    }

    internal enum ContradictionReason
    {
        None,
        ParityUnreachable,
        TruncatedLength,
        TailNotZero,
    }

    internal static ColumnBounds ComputeColumnBounds(
        ReadOnlySpan<BitState> multiplicand,
        ReadOnlySpan<BitState> multiplier,
        int columnIndex)
    {
        if (columnIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(columnIndex));
        }

        long forced = 0;
        long possible = 0;

        for (int i = 0; i <= columnIndex; i++)
        {
            BitState a = i < multiplicand.Length ? multiplicand[i] : BitState.Zero;
            BitState q = columnIndex - i < multiplier.Length ? multiplier[columnIndex - i] : BitState.Zero;

            if (a == BitState.Zero || q == BitState.Zero)
            {
                continue;
            }

            if (a == BitState.One && q == BitState.One)
            {
                forced++;
                possible++;
                continue;
            }

            possible++;
        }

        return new ColumnBounds(forced, possible);
    }

    internal static bool TryPropagateCarry(
        CarryRange currentCarry,
        long forcedOnes,
        long possibleOnes,
        int requiredResultBit,
        out CarryRange nextCarry,
        out ContradictionReason reason)
    {
        if (forcedOnes < 0 || possibleOnes < forcedOnes)
        {
            throw new ArgumentOutOfRangeException(nameof(possibleOnes), "possibleOnes must be >= forcedOnes and both non-negative.");
        }

        long minSum = currentCarry.Min + forcedOnes;
        long maxSum = currentCarry.Max + possibleOnes;
        int parity = requiredResultBit & 1;

        long minAligned = AlignUpToParity(minSum, parity);
        long maxAligned = AlignDownToParity(maxSum, parity);

        if (minAligned > maxAligned)
        {
            nextCarry = default;
            reason = ContradictionReason.ParityUnreachable;
            return false;
        }

        nextCarry = new CarryRange((minAligned - requiredResultBit) >> 1, (maxAligned - requiredResultBit) >> 1);
        reason = ContradictionReason.None;
        return true;
    }

    /// <summary>
    /// Full column-wise propagation for a given set of q one-bit offsets (LSB=0) and exponent p.
    /// Returns true if the solver could decide locally (divides or contradiction).
    /// </summary>
    internal static bool TryCheckDivisibilityFromOneOffsets(
        ReadOnlySpan<int> qOneOffsets,
        ulong p,
        out bool divides,
        out ContradictionReason reason)
    {
        divides = false;
        reason = ContradictionReason.None;

        if (p == 0 || qOneOffsets.Length == 0)
        {
            reason = ContradictionReason.ParityUnreachable;
            return true;
        }

        // Exact small-case evaluation to short-circuit both success and failure deterministically.
        int highestBit = qOneOffsets[^1];
        if (p <= 64 && highestBit < 63)
        {
            ulong q = 0;
            foreach (int bit in qOneOffsets)
            {
                q |= 1UL << bit;
            }

            ulong mersenne = (1UL << (int)p) - 1UL;
            divides = q != 0 && (mersenne % q == 0);
            reason = divides ? ContradictionReason.None : ContradictionReason.ParityUnreachable;
            return true;
        }

        int maxOffset = qOneOffsets[^1];
        if ((ulong)(maxOffset + 1) >= p)
        {
            divides = false;
            reason = ContradictionReason.TruncatedLength;
            return true;
        }

        if (qOneOffsets[0] != 0)
        {
            divides = false;
            reason = ContradictionReason.ParityUnreachable;
            return true;
        }

        var aBits = new Dictionary<long, sbyte>(capacity: qOneOffsets.Length * 4);
        long carryMin = 0;
        long carryMax = 0;
        long maxKnownA = -1;
        long maxAllowedA = (long)p - (maxOffset + 1L);
        if (maxAllowedA < 0)
        {
            divides = false;
            reason = ContradictionReason.TruncatedLength;
            return true;
        }

        for (long column = 0; column < (long)p; column++)
        {
            long forced = 0;
            long unknown = 0;
            long chosenIndex = -1;

            foreach (int offset in qOneOffsets)
            {
                long aIndex = column - offset;
                if (aIndex < 0)
                {
                    break;
                }

                if (aIndex > maxAllowedA)
                {
                    continue; // a must be zero here
                }

                if (aBits.TryGetValue(aIndex, out sbyte aVal))
                {
                    if (aVal == 1)
                    {
                        forced++;
                    }
                }
                else
                {
                    unknown++;
                    if (chosenIndex == -1)
                    {
                        chosenIndex = aIndex;
                    }
                }
            }

            long minSum = carryMin + forced;
            long maxSum = carryMax + forced + unknown;
            const int requiredParity = 1;

            if (minSum == maxSum)
            {
                if ((minSum & 1) != requiredParity)
                {
                    divides = false;
                    reason = ContradictionReason.ParityUnreachable;
                    return true;
                }

                long nextCarry = (minSum - requiredParity) >> 1;
                carryMin = nextCarry;
                carryMax = nextCarry;
            }
            else
            {
                if ((minSum & 1) != requiredParity)
                {
                    if (unknown == 0)
                    {
                        divides = false;
                        return true;
                    }

                    if (chosenIndex >= 0)
                    {
                        aBits[chosenIndex] = 1;
                        forced++;
                        unknown--;
                        if (chosenIndex > maxKnownA)
                        {
                            maxKnownA = chosenIndex;
                        }
                    }
                }

                if (!TryPropagateCarry(
                        new CarryRange(carryMin, carryMax),
                        forced,
                        forced + unknown,
                        requiredParity,
                        out var next,
                        out var propagateReason))
                {
                    divides = false;
                    reason = propagateReason;
                    return true;
                }

                carryMin = next.Min;
                carryMax = next.Max;
            }

            if (chosenIndex >= 0 && !aBits.ContainsKey(chosenIndex))
            {
                aBits[chosenIndex] = 1;
                if (chosenIndex > maxKnownA)
                {
                    maxKnownA = chosenIndex;
                }
            }
        }

        // Tail: keep propagating until carry collapses to zero. No fixed budget; carry halves every column when no ones remain.
        for (long column = (long)p; carryMax > 0; column++)
        {
            long forced = 0;
            long unknown = 0;
            foreach (int offset in qOneOffsets)
            {
                long aIndex = column - offset;
                if (aIndex < 0)
                {
                    break;
                }

                if (aIndex > maxAllowedA)
                {
                    continue; // forced zero
                }

                if (aBits.TryGetValue(aIndex, out sbyte aVal))
                {
                    if (aVal == 1)
                    {
                        forced++;
                    }
                }
                else
                {
                    unknown++;
                }
            }

            if (!TryPropagateCarry(
                    new CarryRange(carryMin, carryMax),
                    forced,
                    forced + unknown,
                    requiredResultBit: 0,
                    out var next,
                    out var reasonTail))
            {
                divides = false;
                reason = reasonTail;
                return true;
            }

            carryMin = next.Min;
            carryMax = next.Max;
        }

        divides = carryMin == 0 && carryMax == 0;
        reason = divides ? ContradictionReason.None : ContradictionReason.TailNotZero;
        return true;
    }

    internal static bool TryCheckDivisibilityFromOneOffsets(
        ReadOnlySpan<int> qOneOffsets,
        ulong p,
        out bool divides)
    {
        return TryCheckDivisibilityFromOneOffsets(qOneOffsets, p, out divides, out _);
    }

    private static long AlignUpToParity(long value, int parity)
    {
        return (value & 1) == parity ? value : value + 1;
    }

    private static long AlignDownToParity(long value, int parity)
    {
        return (value & 1) == parity ? value : value - 1;
    }
}
