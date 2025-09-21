using FluentAssertions;
using PerfectNumbers.Core.Gpu;
using System.Numerics;
using System.Reflection;
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

    [Fact]
    [Trait("Category", "Fast")]
    public void IsDivisible_handles_large_divisors()
    {
        var tester = new MersenneNumberDivisorGpuTester();
        UInt128 divisor = (UInt128.One << 65) - UInt128.One;
        tester.IsDivisible(65UL, divisor).Should().BeTrue();
        tester.IsDivisible(65UL, divisor + 2).Should().BeFalse();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void IsPrime_sets_divisorsExhausted_false_when_search_range_not_exhausted()
    {
        var tester = new MersenneNumberDivisorGpuTester();
        typeof(MersenneNumberDivisorGpuTester)
            .GetField("_divisorCandidates", BindingFlags.NonPublic | BindingFlags.Static)!
            .SetValue(null, Array.Empty<(ulong, uint)>());

        tester.IsPrime(11UL, UInt128.Zero, 0UL, out bool divisorsExhausted).Should().BeTrue();
        divisorsExhausted.Should().BeFalse();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void IsPrime_sets_divisorsExhausted_true_when_divisible_by_specified_divisor()
    {
        var tester = new MersenneNumberDivisorGpuTester();
        tester.IsPrime(3UL, 7UL, 0UL, out bool divisorsExhausted).Should().BeFalse();
        divisorsExhausted.Should().BeTrue();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void IsPrime_accepts_large_search_limits()
    {
        var tester = new MersenneNumberDivisorGpuTester();
        tester.IsPrime(3UL, 7UL, ulong.MaxValue, out bool exhausted).Should().BeFalse();
        exhausted.Should().BeTrue();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_tester_tracks_divisors_across_primes()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester();
        tester.IsPrime(5UL, 64UL, out bool exhausted).Should().BeTrue();
        exhausted.Should().BeTrue();

        tester.IsPrime(7UL, 96UL, out exhausted).Should().BeTrue();
        exhausted.Should().BeTrue();

        tester.IsPrime(11UL, 128UL, out exhausted).Should().BeFalse();
        exhausted.Should().BeTrue();
    }
}

