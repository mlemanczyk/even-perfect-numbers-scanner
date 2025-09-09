using FluentAssertions;
using PerfectNumbers.Core.Gpu;
using Xunit;

namespace PerfectNumbers.Core.Tests.Gpu;

[Trait("Category", "Slow")]
public class LucasLehmerPrimeTesterTests
{
    [Theory]
    [InlineData(3UL, true)]
    [InlineData(5UL, true)]
    [InlineData(11UL, false)]
    [Trait("Category", "Fast")]
    public void IsMersennePrime_gpu_returns_expected_results(ulong p, bool expected)
    {
        new MersenneNumberLucasLehmerGpuTester().IsMersennePrime(p).Should().Be(expected);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void IsMersennePrime_gpu_large_exponent_without_cpu_fallback()
    {
        // Use a modest exponent that still exercises the NTT path while keeping
        // runtime short enough for the unit test environment.
        const ulong exponent = 128UL;
        var tester = new MersenneNumberLucasLehmerGpuTester();
        tester.WarmUpNttParameters(exponent);
        tester.IsMersennePrime(exponent).Should().BeFalse();
    }

    [Theory]
    [InlineData(132UL)]
    [InlineData(1000UL)]
    [Trait("Category", "Slow")]
    public void WarmUpNttParameters_populates_cache_for_large_exponent(ulong exponent)
    {
        var tester = new MersenneNumberLucasLehmerGpuTester();
        tester.WarmUpNttParameters(exponent);
        tester.IsMersennePrime(exponent).Should().BeFalse();
    }

    [Theory]
    [InlineData(3UL, true)]
    [InlineData(5UL, true)]
    [InlineData(11UL, false)]
    [Trait("Category", "Fast")]
    public void IsMersennePrime_cpu_matches_expected_results(ulong p, bool expected)
    {
        new MersenneNumberLucasLehmerCpuTester().IsPrime(p).Should().Be(expected);
    }
}

