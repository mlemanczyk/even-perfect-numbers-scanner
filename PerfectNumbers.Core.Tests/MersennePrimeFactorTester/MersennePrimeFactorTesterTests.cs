using FluentAssertions;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class MersennePrimeFactorTesterTests
{
    [Theory]
    [InlineData(5UL, 31UL, true)]
    [InlineData(11UL, 23UL, true)]
    [InlineData(11UL, 89UL, true)]
    [InlineData(5UL, 7UL, false)]
    [InlineData(7UL, 17UL, false)]
    [InlineData(11UL, 331UL, false)]
    [Trait("Category", "Fast")]
    public void IsPrimeFactor_identifies_divisibility(ulong p, ulong q, bool expected)
    {
        MersennePrimeFactorTester.IsPrimeFactor(p, q, CancellationToken.None).Should().Be(expected);
    }

    [Fact]
    [Trait("Category", "Slow")]
    public void GetOrderOf2ModPrime_handles_large_divisors()
    {
        UInt128 q = UInt128.Parse("618970019642690137449562111");
        MersennePrimeFactorTester.GetOrderOf2ModPrime(q, CancellationToken.None)
            .Should().Be((UInt128)89);
    }
}
