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

        primes[..5].Should().Equal([2, 3, 5, 7, 11]);

        squares[..5].Should().Equal([4UL, 9UL, 25UL, 49UL, 121UL]);

        for (int i = 0; i < 5; i++)
        {
            squares[i].Should().Be(primes[i] * primes[i]);
        }
    }
}

