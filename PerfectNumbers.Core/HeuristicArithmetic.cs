using System;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

internal static class HeuristicArithmetic
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong MultiplyShiftRight(ulong value, ulong multiplier, int shift)
    {
        if (shift < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(shift), "Shift must be non-negative.");
        }

        UInt128 product = (UInt128)value * multiplier;
        return (ulong)(product >> shift);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong MultiplyShiftRightShiftFirst(ulong value, ulong multiplier, int shift)
    {
        if (shift < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(shift), "Shift must be non-negative.");
        }

        if (shift == 0)
        {
            return MultiplyShiftRight(value, multiplier, 0);
        }

        if (shift >= 64)
        {
            return 0UL;
        }

        ulong high = value >> shift;
        ulong mask = (1UL << shift) - 1UL;
        ulong low = value & mask;

        UInt128 highContribution = (UInt128)high * multiplier;
        UInt128 lowContribution = (UInt128)low * multiplier;

        UInt128 combined = highContribution + (lowContribution >> shift);
        return (ulong)combined;
    }
}
