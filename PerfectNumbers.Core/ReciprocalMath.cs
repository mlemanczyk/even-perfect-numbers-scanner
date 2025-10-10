using System.Numerics;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

internal static class ReciprocalMath
{
    private const int ReciprocalShift = 64;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong ComputeReciprocalEstimate(ulong divisor)
    {
        if (divisor <= 1UL)
        {
            return ulong.MaxValue;
        }

        UInt128 scaledOne = UInt128.One << ReciprocalShift;
        UInt128 estimate = scaledOne / divisor;
        if ((scaledOne % divisor) != UInt128.Zero)
        {
            estimate += UInt128.One;
        }

        if (estimate == UInt128.Zero)
        {
            return 1UL;
        }

        if (estimate > ulong.MaxValue)
        {
            return ulong.MaxValue;
        }

        return (ulong)estimate;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong RefineReciprocalEstimate(ulong reciprocal, ulong divisor)
    {
        if (divisor <= 1UL)
        {
            return ulong.MaxValue;
        }

        if (reciprocal == 0UL || reciprocal == ulong.MaxValue)
        {
            return ComputeReciprocalEstimate(divisor);
        }

        UInt128 reciprocal128 = reciprocal;
        UInt128 product = (UInt128)divisor * reciprocal128;
        UInt128 scaled = product >> ReciprocalShift;
        UInt128 twoScaled = UInt128.One << (ReciprocalShift + 1);
        if (scaled >= twoScaled)
        {
            return ComputeReciprocalEstimate(divisor);
        }

        UInt128 correction = twoScaled - scaled;
        UInt128 refined = (reciprocal128 * correction) >> ReciprocalShift;
        if (refined == UInt128.Zero)
        {
            return 1UL;
        }

        return refined > ulong.MaxValue ? ulong.MaxValue : (ulong)refined;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Reduce(UInt128 numerator, ulong divisor, ulong reciprocal)
    {
        if (divisor <= 1UL)
        {
            return 0UL;
        }

        if (numerator == UInt128.Zero)
        {
            return 0UL;
        }

        if (reciprocal == 0UL || reciprocal == ulong.MaxValue)
        {
            return (ulong)(numerator % divisor);
        }

        ulong high = (ulong)(numerator >> ReciprocalShift);
        ulong low = (ulong)numerator;

        if (high >= divisor)
        {
            return (ulong)(numerator % divisor);
        }

        UInt128 highProduct = (UInt128)high * reciprocal;
        UInt128 lowProduct = (UInt128)low * reciprocal;
        UInt128 estimate = highProduct + (lowProduct >> ReciprocalShift);

        if (estimate == UInt128.Zero)
        {
            return (ulong)(numerator % divisor);
        }

        UInt128 product = estimate * divisor;
        while (product > numerator)
        {
            if (estimate == UInt128.Zero)
            {
                return (ulong)(numerator % divisor);
            }

            estimate--;
            product -= divisor;
        }

        UInt128 remainder = numerator - product;
        while (remainder >= divisor)
        {
            remainder -= divisor;
        }

        return (ulong)remainder;
    }
}
