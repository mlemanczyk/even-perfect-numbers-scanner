using FluentAssertions;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class UIntExtensionsTests
{
    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(0U)]
    [InlineData(1U)]
    [InlineData(1234567890U)]
    [InlineData(uint.MaxValue)]
    public void Mod3_matches_operator(uint value)
    {
        value.Mod3().Should().Be(value % 3U);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(0U)]
    [InlineData(1U)]
    [InlineData(1234567890U)]
    [InlineData(uint.MaxValue)]
    public void Mod5_matches_operator(uint value)
    {
        value.Mod5().Should().Be(value % 5U);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(0U)]
    [InlineData(1U)]
    [InlineData(1234567890U)]
    [InlineData(uint.MaxValue)]
    public void Mod6_matches_operator(uint value)
    {
        value.Mod6().Should().Be(value % 6U);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(0U)]
    [InlineData(1U)]
    [InlineData(1234567890U)]
    [InlineData(uint.MaxValue)]
    public void Mod7_matches_operator(uint value)
    {
        value.Mod7().Should().Be(value % 7U);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(0U)]
    [InlineData(1U)]
    [InlineData(1234567890U)]
    [InlineData(uint.MaxValue)]
    public void Mod8_matches_operator(uint value)
    {
        value.Mod8().Should().Be(value % 8U);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(0U)]
    [InlineData(1U)]
    [InlineData(1234567890U)]
    [InlineData(uint.MaxValue)]
    public void Mod10_matches_operator(uint value)
    {
        value.Mod10().Should().Be(value % 10U);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(0U)]
    [InlineData(1U)]
    [InlineData(1234567890U)]
    [InlineData(uint.MaxValue)]
    public void Mod11_matches_operator(uint value)
    {
        value.Mod11().Should().Be(value % 11U);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(0U)]
    [InlineData(1U)]
    [InlineData(1234567890U)]
    [InlineData(uint.MaxValue)]
    public void Mod128_matches_operator(uint value)
    {
        value.Mod128().Should().Be(value % 128U);
    }
}

