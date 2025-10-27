using System;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

internal static class DivisorGenerator
{
    private const ushort DecimalMaskWhenLastIsSeven = (1 << 3) | (1 << 7) | (1 << 9);
    private const ushort DecimalMaskOtherwise = (1 << 1) | (1 << 3) | (1 << 9);

    private const int DecimalFilterLength = 10;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PopulateDecimalFilter(LastDigit lastDigit, Span<byte> destination)
    {
        ushort mask = GetDecimalMask(lastDigit);
        int length = destination.Length;
        int limit = length < DecimalFilterLength ? length : DecimalFilterLength;

        for (int index = 0; index < limit; index++)
        {
            destination[index] = (byte)((mask >> index) & 1);
        }

        for (int index = limit; index < length; index++)
        {
            destination[index] = 0;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ushort GetDecimalMask(LastDigit lastDigit)
    {
        return lastDigit == LastDigit.Seven ? DecimalMaskWhenLastIsSeven : DecimalMaskOtherwise;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsValidDivisor(
        byte remainder10,
        byte remainder8,
        byte remainder3,
        byte remainder5,
        byte remainder7,
        byte remainder11,
        ushort decimalMask)
    {
        if (((decimalMask >> remainder10) & 1) == 0)
        {
            return false;
        }

        if (remainder8 != 1 && remainder8 != 7)
        {
            return false;
        }

        if (remainder3 == 0 || remainder5 == 0 || remainder7 == 0 || remainder11 == 0)
        {
            return false;
        }

        return true;
    }
}
