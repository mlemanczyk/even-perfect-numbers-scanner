using System.Numerics;
using FluentAssertions;
using PeterO.Numbers;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class PowerCacheTests
{
    [Fact]
    [Trait("Category", "Fast")]
    public void Get_returns_big_integer_power()
    {
        PowerCache.Get(2UL, 5).Should().Be(new BigInteger(32));
        PowerCache.Get(2UL, 10).Should().Be(new BigInteger(1024));
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void Get_returns_einteger_power()
    {
        EInteger baseVal = EInteger.FromInt32(3);
        PowerCache.Get(baseVal, 4).Should().Be(EInteger.FromInt32(81));
        PowerCache.Get(baseVal, 2).Should().Be(EInteger.FromInt32(9));
    }
}

