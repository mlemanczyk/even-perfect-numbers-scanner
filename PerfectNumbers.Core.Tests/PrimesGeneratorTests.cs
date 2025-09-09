using FluentAssertions;
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

        primes[..5].Should().Equal(new ulong[] { 2UL, 3UL, 5UL, 7UL, 11UL });

        squares[..5].Should().Equal(new ulong[] { 4UL, 9UL, 25UL, 49UL, 121UL });

        for (int i = 0; i < 5; i++)
        {
            squares[i].Should().Be(primes[i] * primes[i]);
        }
    }
}

