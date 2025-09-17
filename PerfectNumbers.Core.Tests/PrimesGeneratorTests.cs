using FluentAssertions;
using PerfectNumbers.Core;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class PrimesGeneratorTests
{
    [Fact]
    [Trait("Category", "Fast")]
    public void Small_primes_and_squares_match()
    {
        var primes = PrimesGenerator.SmallPrimes;
        var squares = PrimesGenerator.SmallPrimesPow2;

        primes[..5].Should().Equal([2, 3, 5, 7, 11]);
        squares[..5].Should().Equal([4UL, 9UL, 25UL, 49UL, 121UL]);

        for (int i = 0; i < 5; i++)
        {
            squares[i].Should().Be(primes[i] * primes[i]);
        }

        primes.Length.Should().Be((int)PerfectNumberConstants.PrimesLimit);
        squares.Length.Should().Be((int)PerfectNumberConstants.PrimesLimit);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void Residue_prime_tables_follow_digit_rules()
    {
        var lastOne = PrimesGenerator.SmallPrimesLastOne;
        var lastOnePow2 = PrimesGenerator.SmallPrimesPow2LastOne;
        var lastSeven = PrimesGenerator.SmallPrimesLastSeven;
        var lastSevenPow2 = PrimesGenerator.SmallPrimesPow2LastSeven;

        lastOne.Length.Should().Be((int)PerfectNumberConstants.PrimesLimit);
        lastSeven.Length.Should().Be((int)PerfectNumberConstants.PrimesLimit);
        lastOnePow2.Length.Should().Be(lastOne.Length);
        lastSevenPow2.Length.Should().Be(lastSeven.Length);

        lastOne[..5].Should().Equal([3U, 7U, 11U, 13U, 19U]);
        lastSeven[..5].Should().Equal([3U, 7U, 11U, 13U, 17U]);

        for (int i = 0; i < lastOne.Length; i++)
        {
            uint prime = lastOne[i];
            uint mod10 = prime % 10U;
            (mod10 == 1U || mod10 == 3U || mod10 == 9U || prime == 7U || prime == 11U)
                .Should().BeTrue();
            lastOnePow2[i].Should().Be(prime * (ulong)prime);
        }

        for (int i = 0; i < lastSeven.Length; i++)
        {
            uint prime = lastSeven[i];
            uint mod10 = prime % 10U;
            (mod10 == 3U || mod10 == 7U || mod10 == 9U || prime == 11U)
                .Should().BeTrue();
            lastSevenPow2[i].Should().Be(prime * (ulong)prime);
        }
    }
}

