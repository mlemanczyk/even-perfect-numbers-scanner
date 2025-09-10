using FluentAssertions;
using PerfectNumbers.Core.Gpu;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class MersenneNumberDivisorGpuTesterTests
{
    [Theory]
    [InlineData(3UL, 7UL, true)]
    [InlineData(11UL, 23UL, true)]
    [InlineData(7UL, 35UL, false)]
    [InlineData(13UL, 23UL, false)]
    [Trait("Category", "Fast")]
    public void IsDivisible_returns_expected(ulong exponent, ulong divisor, bool expected)
    {
        var tester = new MersenneNumberDivisorGpuTester();
        tester.IsDivisible(exponent, divisor).Should().Be(expected);
    }
}
