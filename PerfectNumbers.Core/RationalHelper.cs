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
        // TODO: Replace the BigInteger-based GCD with the UInt128-friendly binary GCD helper so this rounding path avoids
        // allocating big integers for hot rational conversions.
        return ERational.Create(EInteger.FromInt64(numerator / gcd), EInteger.FromInt64(denominator / gcd));
    }

    public static EInteger ToEInteger(BigInteger value)
    {
        // TODO: Cache a reusable byte buffer here so converting to EInteger no longer allocates per call when parsing alpha
        // tables.
        return EInteger.FromBytes(value.ToByteArray(), true);
    }

    public static ulong FloorToUInt64(ERational value)
    {
        EInteger ei = value.ToEInteger();
        // TODO: Inline this floor conversion with the span-based ladder that dominated
        // ResidueComputationBenchmarks so the CPU path skips the expensive ERational ->
        // EInteger materialization when reducing large 2kp+1 divisors.
        return ei.ToUInt64Checked();
    }

    public static ulong CeilingToUInt64(ERational value)
    {
        EInteger ei = value.ToEInteger();
        if (value.CompareTo(ERational.Create(ei, EInteger.One)) > 0)
            ei += EInteger.One;
        // TODO: Replace this ceiling branch with the branchless adjustment measured fastest
        // in ResidueComputationBenchmarks so large-divisor scans avoid repeated ERational
        // conversions on the CPU hot path.
        return ei.ToUInt64Checked();
    }
}

