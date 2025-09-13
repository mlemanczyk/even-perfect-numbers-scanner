using FluentAssertions;
using Xunit;

namespace PerfectNumbers.Core.Tests;

[Trait("Category", "Fast")]
public class MersenneResidueModeTests
{
    [Fact]
    public void IsMersennePrime_residue_mode_rejects_composite_exponent()
    {
        var tester = new MersenneNumberTester(useResidue: true, useIncremental: false);
        tester.IsMersennePrime(136_000_002UL).Should().BeFalse();
    }

    [Fact]
    public void IsMersennePrime_residue_mode_accepts_prime_exponent()
    {
        var tester = new MersenneNumberTester(useResidue: true, useIncremental: true, maxK: 1_000UL);
        tester.IsMersennePrime(31UL).Should().BeTrue();
    }
}

