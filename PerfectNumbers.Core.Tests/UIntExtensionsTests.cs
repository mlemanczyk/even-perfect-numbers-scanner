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
        UIntExtensions.Mod3(value).Should().Be(value % 3U);
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
    public void Mod8_matches_operator(uint value)
    {
        value.Mod8().Should().Be(value % 8U);
    }
}

