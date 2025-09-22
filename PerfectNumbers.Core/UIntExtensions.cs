using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public static class UIntExtensions
{
    private static readonly byte[] Mod6Lookup = { 0, 3, 4, 1, 2, 5 };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Mod3(this uint value)
    {
        ulong wideValue = 0UL;
        uint quotient = 0U;
        uint remainder = 0U;

        wideValue = (ulong)value * 0xAAAAAAABUL;
        quotient = (uint)(wideValue >> 33);
        remainder = value - quotient * 3U;

        return remainder;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Mod5(this uint value)
    {
        ulong wideValue = 0UL;
        uint quotient = 0U;
        uint remainder = 0U;

        wideValue = (ulong)value * 0xCCCCCCCDUL;
        quotient = (uint)(wideValue >> 34);
        remainder = value - quotient * 5U;

        return remainder;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Mod6(this uint value)
    {
        return Mod6Lookup[(int)(((value.Mod3() << 1) | (value & 1U)))];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Mod7(this uint value)
    {
        uint remainder = 0U;
        uint temp = 0U;
        uint chunk = 0U;

        temp = value;
        while (temp != 0U)
        {
            chunk = temp & 7U;
            remainder += chunk;
            temp >>= 3;
            if (remainder >= 7U)
            {
                remainder -= 7U;
            }
        }

        return remainder >= 7U ? remainder - 7U : remainder;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Mod8(this uint value) => value & 7U;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Mod10(this uint value)
    {
        ulong wideValue = 0UL;
        uint quotient = 0U;
        uint remainder = 0U;

        wideValue = (ulong)value * 0xCCCCCCCDUL;
        quotient = (uint)(wideValue >> 35);
        remainder = value - quotient * 10U;

        return remainder;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Mod11(this uint value)
    {
        uint remainder = 0U;
        uint temp = 0U;
        uint chunk = 0U;

        temp = value;
        while (temp != 0U)
        {
            chunk = temp & 1023U;
            remainder += chunk;
            temp >>= 10;
            remainder -= 11U * (remainder / 11U);
        }

        return remainder;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Mod128(this uint value) => value & 127U;
}

