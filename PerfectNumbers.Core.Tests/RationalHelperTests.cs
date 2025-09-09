using System.Numerics;
using FluentAssertions;
using PeterO.Numbers;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class RationalHelperTests
{
    [Fact]
    [Trait("Category", "Fast")]
    public void FromDoubleRounded_ReducesFraction()
    {
        ERational result = RationalHelper.FromDoubleRounded(0.75);

        result.Numerator.ToInt32Checked().Should().Be(3);
        result.Denominator.ToInt32Checked().Should().Be(4);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ToEInteger_ReturnsMatchingValue()
    {
        BigInteger value = BigInteger.Parse("12345678901234567890");

        EInteger einteger = RationalHelper.ToEInteger(value);

        einteger.ToString().Should().Be(value.ToString());
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void FloorToUInt64_ReturnsFloor()
    {
        ERational value = ERational.Create(5, 2);

        ulong result = RationalHelper.FloorToUInt64(value);

        result.Should().Be(2UL);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void CeilingToUInt64_RoundsUpWhenFractional()
    {
        ERational fractional = ERational.Create(5, 2);
        ERational integral = ERational.Create(6, 2);

        RationalHelper.CeilingToUInt64(fractional).Should().Be(3UL);
        RationalHelper.CeilingToUInt64(integral).Should().Be(3UL);
    }
}

