using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core.Gpu;

public struct GpuRational : IEquatable<GpuRational>, IComparable<GpuRational>
{
    private const double Pow64 = 18446744073709551616.0;

    public GpuUInt128 Numerator;

    public GpuUInt128 Denominator;

    public static readonly GpuRational Zero = new(GpuUInt128.Zero, GpuUInt128.One);

    public static readonly GpuRational One = new(GpuUInt128.One, GpuUInt128.One);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuRational(GpuUInt128 numerator)
        : this(numerator, GpuUInt128.One)
    {
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuRational(ulong numerator)
        : this(new GpuUInt128(numerator), GpuUInt128.One)
    {
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuRational(GpuUInt128 numerator, GpuUInt128 denominator)
    {
        if (denominator.IsZero)
        {
            throw new DivideByZeroException("GpuRational denominator cannot be zero.");
        }

        Numerator = numerator;
        Denominator = denominator;
        Normalize();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuRational(ulong numerator, ulong denominator)
        : this(new GpuUInt128(numerator), new GpuUInt128(denominator))
    {
    }

    public readonly bool IsZero
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Numerator.IsZero;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator GpuRational(GpuUInt128 value) => new(value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator GpuRational(ulong value) => new(value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Add(in GpuRational other)
    {
        GpuUInt128 leftDenominator = Denominator;
        GpuUInt128 rightDenominator = other.Denominator;

        GpuUInt128 gcd = GpuUInt128.BinaryGcd(leftDenominator, rightDenominator);
        if (gcd != GpuUInt128.One)
        {
            leftDenominator = GpuUInt128.DivideExact(leftDenominator, gcd);
            rightDenominator = GpuUInt128.DivideExact(rightDenominator, gcd);
            // Reusing both denominator variables to store their reduced values because the unreduced inputs are no longer needed.
        }

        Numerator.Mul(rightDenominator);
        GpuUInt128 scaledOtherNumerator = other.Numerator;
        scaledOtherNumerator.Mul(leftDenominator);
        Numerator.Add(scaledOtherNumerator);

        Denominator = leftDenominator;
        // Denominator now holds the reduced left denominator; multiply by the original right denominator to form the LCM.
        Denominator.Mul(other.Denominator);
        Normalize();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Sub(in GpuRational other)
    {
        GpuUInt128 leftDenominator = Denominator;
        GpuUInt128 rightDenominator = other.Denominator;

        GpuUInt128 gcd = GpuUInt128.BinaryGcd(leftDenominator, rightDenominator);
        if (gcd != GpuUInt128.One)
        {
            leftDenominator = GpuUInt128.DivideExact(leftDenominator, gcd);
            rightDenominator = GpuUInt128.DivideExact(rightDenominator, gcd);
            // Reusing the denominator variables to keep the reduced values since the raw denominators are no longer required.
        }

        Numerator.Mul(rightDenominator);
        GpuUInt128 scaledOtherNumerator = other.Numerator;
        scaledOtherNumerator.Mul(leftDenominator);
        if (Numerator.CompareTo(scaledOtherNumerator) < 0)
        {
            throw new InvalidOperationException("Resulting GpuRational would be negative.");
        }

        Numerator.Sub(scaledOtherNumerator);

        Denominator = leftDenominator;
        Denominator.Mul(other.Denominator);
        Normalize();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Mul(in GpuRational other)
    {
        GpuUInt128 rightDenominator = other.Denominator;
        GpuUInt128 gcd = GpuUInt128.BinaryGcd(Numerator, rightDenominator);
        if (gcd != GpuUInt128.One)
        {
            Numerator = GpuUInt128.DivideExact(Numerator, gcd);
            rightDenominator = GpuUInt128.DivideExact(rightDenominator, gcd);
            // Reusing rightDenominator to keep the reduced value because the original denominator is no longer required.
        }

        GpuUInt128 rightNumerator = other.Numerator;
        GpuUInt128 leftDenominator = Denominator;
        gcd = GpuUInt128.BinaryGcd(rightNumerator, leftDenominator);
        if (gcd != GpuUInt128.One)
        {
            rightNumerator = GpuUInt128.DivideExact(rightNumerator, gcd);
            leftDenominator = GpuUInt128.DivideExact(leftDenominator, gcd);
            // Reusing rightNumerator and leftDenominator after cross-cancellation keeps the intermediates compact.
        }

        Numerator.Mul(rightNumerator);
        Denominator = leftDenominator;
        Denominator.Mul(rightDenominator);
        Normalize();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Div(in GpuRational other)
    {
        if (other.Numerator.IsZero)
        {
            throw new DivideByZeroException("Cannot divide by a zero GpuRational.");
        }

        GpuUInt128 rightNumerator = other.Numerator;
        GpuUInt128 gcd = GpuUInt128.BinaryGcd(Numerator, rightNumerator);
        if (gcd != GpuUInt128.One)
        {
            Numerator = GpuUInt128.DivideExact(Numerator, gcd);
            rightNumerator = GpuUInt128.DivideExact(rightNumerator, gcd);
            // Reusing rightNumerator to store the reduced numerator keeps temporary allocations out of the hot path.
        }

        GpuUInt128 rightDenominator = other.Denominator;
        GpuUInt128 leftDenominator = Denominator;
        gcd = GpuUInt128.BinaryGcd(rightDenominator, leftDenominator);
        if (gcd != GpuUInt128.One)
        {
            rightDenominator = GpuUInt128.DivideExact(rightDenominator, gcd);
            leftDenominator = GpuUInt128.DivideExact(leftDenominator, gcd);
            // Reusing both denominator variables here mirrors the cross-cancellation performed on the numerators.
        }

        Numerator.Mul(rightDenominator);
        Denominator = leftDenominator;
        Denominator.Mul(rightNumerator);
        Normalize();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuRational operator +(GpuRational left, GpuRational right)
    {
        left.Add(right);
        return left;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuRational operator -(GpuRational left, GpuRational right)
    {
        left.Sub(right);
        return left;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuRational operator *(GpuRational left, GpuRational right)
    {
        left.Mul(right);
        return left;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuRational operator /(GpuRational left, GpuRational right)
    {
        left.Div(right);
        return left;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly int CompareTo(GpuRational other)
    {
        GpuUInt128 leftScaled = Numerator;
        leftScaled.Mul(other.Denominator);
        GpuUInt128 rightScaled = other.Numerator;
        rightScaled.Mul(Denominator);
        return leftScaled.CompareTo(rightScaled);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly bool Equals(GpuRational other) => Numerator == other.Numerator && Denominator == other.Denominator;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override readonly bool Equals(object? obj) => obj is GpuRational other && Equals(other);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override readonly int GetHashCode() => HashCode.Combine(Numerator, Denominator);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool operator ==(GpuRational left, GpuRational right) => left.Equals(right);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool operator !=(GpuRational left, GpuRational right) => !left.Equals(right);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool operator <(GpuRational left, GpuRational right) => left.CompareTo(right) < 0;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool operator >(GpuRational left, GpuRational right) => left.CompareTo(right) > 0;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool operator <=(GpuRational left, GpuRational right) => left.CompareTo(right) <= 0;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool operator >=(GpuRational left, GpuRational right) => left.CompareTo(right) >= 0;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly double ToDouble()
    {
        double numerator = Numerator.High == 0UL ? Numerator.Low : Numerator.High * Pow64 + Numerator.Low;
        double denominator = Denominator.High == 0UL ? Denominator.Low : Denominator.High * Pow64 + Denominator.Low;
        return numerator / denominator;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override readonly string ToString() => $"{Numerator}/{Denominator}";

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Normalize()
    {
        if (Numerator.IsZero)
        {
            Denominator = GpuUInt128.One;
            return;
        }

        GpuUInt128 gcd = GpuUInt128.BinaryGcd(Numerator, Denominator);
        if (gcd != GpuUInt128.One)
        {
            Numerator = GpuUInt128.DivideExact(Numerator, gcd);
            Denominator = GpuUInt128.DivideExact(Denominator, gcd);
        }
    }
}
