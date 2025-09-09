using FluentAssertions;
using Xunit;

namespace PerfectNumbers.Core.Tests;

[Collection("GpuNtt")]
[Trait("Category", "Fast")]
public class MersenneNumberTesterTests
{
    [Theory]
    [InlineData(136_000_002UL)]
    [InlineData(136_000_005UL)]
    public void IsMersennePrime_detects_small_prime_divisors_for_large_exponents(ulong p)
    {
        var tester = new MersenneNumberTester();
        tester.IsMersennePrime(p).Should().BeFalse();
    }

    [Theory]
    [InlineData(125UL)]
    public void WarmUpOrders_populates_cache_without_affecting_results(ulong p)
    {
        var tester = new MersenneNumberTester(useOrderCache: true, useGpuOrder: true);
        tester.WarmUpOrders(p, 1_000UL);
        tester.IsMersennePrime(p).Should().BeFalse();
    }

    [Theory]
    [InlineData(125UL, true)]
    [InlineData(127UL, false)]
    public void SharesFactorWithExponentMinusOne_detects_common_factors(ulong p, bool expected)
    {
        p.SharesFactorWithExponentMinusOne().Should().Be(expected);
    }
}

