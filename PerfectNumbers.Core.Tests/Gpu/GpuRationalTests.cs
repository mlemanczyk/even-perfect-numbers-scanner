using FluentAssertions;
using PerfectNumbers.Core.Gpu;
using Xunit;

namespace PerfectNumbers.Core.Tests.Gpu;

public class GpuRationalTests
{
    [Fact, Trait("Category", "Fast")]
    public void Constructor_ReducesFraction()
    {
        GpuRational value = new(6UL, 8UL);
        value.Numerator.Should().Be(new GpuUInt128(3UL));
        value.Denominator.Should().Be(new GpuUInt128(4UL));
    }

    [Fact, Trait("Category", "Fast")]
    public void Addition_ReturnsExpectedValue()
    {
        GpuRational left = new(1UL, 3UL);
        GpuRational right = new(1UL, 6UL);

        GpuRational sum = left + right;

        sum.Numerator.Should().Be(new GpuUInt128(1UL));
        sum.Denominator.Should().Be(new GpuUInt128(2UL));
    }

    [Fact, Trait("Category", "Fast")]
    public void Subtraction_ReturnsExpectedValue()
    {
        GpuRational left = new(5UL, 6UL);
        GpuRational right = new(1UL, 6UL);

        GpuRational difference = left - right;

        difference.Numerator.Should().Be(new GpuUInt128(2UL));
        difference.Denominator.Should().Be(new GpuUInt128(3UL));
    }

    [Fact, Trait("Category", "Fast")]
    public void Multiplication_ReturnsExpectedValue()
    {
        GpuRational left = new(3UL, 4UL);
        GpuRational right = new(2UL, 3UL);

        GpuRational product = left * right;

        product.Numerator.Should().Be(new GpuUInt128(1UL));
        product.Denominator.Should().Be(new GpuUInt128(2UL));
    }

    [Fact, Trait("Category", "Fast")]
    public void Division_ReturnsExpectedValue()
    {
        GpuRational left = new(3UL, 5UL);
        GpuRational right = new(9UL, 10UL);

        GpuRational quotient = left / right;

        quotient.Numerator.Should().Be(new GpuUInt128(2UL));
        quotient.Denominator.Should().Be(new GpuUInt128(3UL));
    }

    [Fact, Trait("Category", "Fast")]
    public void CompareTo_DetectsOrdering()
    {
        GpuRational smaller = new(1UL, 4UL);
        GpuRational larger = new(2UL, 3UL);

        smaller.CompareTo(larger).Should().BeLessThan(0);
        larger.CompareTo(smaller).Should().BeGreaterThan(0);
        smaller.CompareTo(new GpuRational(1UL, 4UL)).Should().Be(0);
    }

    [Fact, Trait("Category", "Fast")]
    public void ToDouble_ReturnsExpectedResult()
    {
        GpuRational value = new(1UL, 5UL);
        value.ToDouble().Should().BeApproximately(0.2, 1e-12);
    }
}
