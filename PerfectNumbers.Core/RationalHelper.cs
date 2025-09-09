using System.Numerics;
using PeterO.Numbers;

namespace PerfectNumbers.Core;

public static class RationalHelper
{
    public static ERational FromDoubleRounded(double value, int scale = 100)
    {
        long numerator = checked((long)Math.Round(value * scale));
        long denominator = scale;
        long gcd = (long)BigInteger.GreatestCommonDivisor(numerator, denominator);
        return ERational.Create(EInteger.FromInt64(numerator / gcd), EInteger.FromInt64(denominator / gcd));
    }

    public static EInteger ToEInteger(BigInteger value)
    {
        return EInteger.FromBytes(value.ToByteArray(), true);
    }

    public static ulong FloorToUInt64(ERational value)
    {
        EInteger ei = value.ToEInteger();
        return ei.ToUInt64Checked();
    }

    public static ulong CeilingToUInt64(ERational value)
    {
        EInteger ei = value.ToEInteger();
        if (value.CompareTo(ERational.Create(ei, EInteger.One)) > 0)
            ei += EInteger.One;
        return ei.ToUInt64Checked();
    }
}

