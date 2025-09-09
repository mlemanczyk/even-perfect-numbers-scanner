using FluentAssertions;
using PeterO.Numbers;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class PrimeCacheTests
{
    [Fact]
    [Trait("Category", "Fast")]
    public void GetEulerPrimes_returns_primes_congruent_to_one_mod_four()
    {
        var cache = new PrimeCache();
        EInteger[] result = cache.GetEulerPrimes(EInteger.FromInt32(2), EInteger.FromInt32(20));

        result.Should().Equal(
            EInteger.FromInt32(5),
            EInteger.FromInt32(13),
            EInteger.FromInt32(17));
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void EnumeratePrimes_returns_sequence_starting_from_value()
    {
        var cache = new PrimeCache();
        EInteger[] result = cache.EnumeratePrimes(EInteger.FromInt32(10))
            .Take(5)
            .ToArray();

        result.Should().Equal(
            EInteger.FromInt32(11),
            EInteger.FromInt32(13),
            EInteger.FromInt32(17),
            EInteger.FromInt32(19),
            EInteger.FromInt32(23));
    }
}

