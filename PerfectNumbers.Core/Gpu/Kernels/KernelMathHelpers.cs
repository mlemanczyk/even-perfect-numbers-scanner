using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core.Gpu;

internal static class KernelMathHelpers
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong FastDiv64Gpu(ulong value, ulong divisor, ulong mul)
    {
        ulong quotient = GpuUInt128.MulHigh(value, mul);
        GpuUInt128 remainder = new(0UL, value);
        GpuUInt128 product = new(GpuUInt128.MulHigh(quotient, divisor), quotient * divisor);
        remainder.Sub(product);

        if (remainder.High != 0UL || remainder.Low >= divisor)
        {
            quotient++;
        }

        return quotient;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Mod128By64(GpuUInt128 value, ulong modulus)
    {
        ulong rem = 0UL;
        ulong part = value.High;
        for (int i = 0; i < 64; i++)
        {
            rem = (rem << 1) | (part >> 63);
            if (rem >= modulus)
            {
                rem -= modulus;
            }
            part <<= 1;
        }

        part = value.Low;
        for (int i = 0; i < 64; i++)
        {
            rem = (rem << 1) | (part >> 63);
            if (rem >= modulus)
            {
                rem -= modulus;
            }
            part <<= 1;
        }

        return rem;
    }

    public static ulong CalculateCycleLengthSmall(ulong divisor)
    {
        if ((divisor & (divisor - 1UL)) == 0UL)
            return 1UL;

        ulong order = 1UL, pow = 2UL;
        while (pow != 1UL)
        {
            pow <<= 1;
            if (pow > divisor)
                pow -= divisor;

            order++;
        }

        return order;
    }
}
