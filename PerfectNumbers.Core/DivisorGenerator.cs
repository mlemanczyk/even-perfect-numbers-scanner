using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

internal static class DivisorGenerator
{
    private const ushort DecimalMaskWhenLastIsSeven = (1 << 3) | (1 << 7) | (1 << 9);
    private const ushort DecimalMaskOtherwise = (1 << 1) | (1 << 3) | (1 << 9);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ushort GetDecimalMask(LastDigit lastDigit)
    {
        return lastDigit == LastDigit.Seven ? DecimalMaskWhenLastIsSeven : DecimalMaskOtherwise;
    }

}
