using System;
using System.Reflection;
using FluentAssertions;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Cpu;
using PerfectNumbers.Core.Gpu;
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
        tester.ConfigureFromMaxPrime(11UL);

        tester.IsPrime(5UL, out bool exhausted).Should().BeTrue();
        exhausted.Should().BeTrue();

        tester.IsPrime(7UL, out exhausted).Should().BeTrue();
        exhausted.Should().BeTrue();

        tester.IsPrime(11UL, out exhausted).Should().BeFalse();
        exhausted.Should().BeTrue();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_tester_requires_configuration()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester();
        Action act = () => tester.IsPrime(5UL, out _);
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_tester_respects_filter_based_limit()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester();
        tester.ConfigureFromMaxPrime(5UL);

        tester.IsPrime(5UL, out _).Should().BeTrue();
        tester.IsPrime(7UL, out bool exhausted).Should().BeTrue();
        exhausted.Should().BeTrue();

        tester.IsPrime(11UL, out bool divisorsExhausted).Should().BeTrue();
        divisorsExhausted.Should().BeTrue();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_gpu_tester_uses_cycle_remainders_per_divisor()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester
        {
            GpuBatchSize = 8,
        };

        tester.ConfigureFromMaxPrime(13UL);

        tester.IsPrime(11UL, out bool divisorsExhausted).Should().BeFalse();
        divisorsExhausted.Should().BeTrue();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_session_checks_divisors_across_primes()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester();
        tester.ConfigureFromMaxPrime(13UL);

        using var session = tester.CreateDivisorSession();
        ulong[] primes = { 5UL, 7UL, 11UL, 13UL };
        byte[] hits = new byte[primes.Length];

        ulong cycle23 = MersenneDivisorCycles.CalculateCycleLength(23UL);
        session.CheckDivisor(23UL, cycle23, primes, hits);
        hits.Should().ContainInOrder(new byte[] { 0, 0, 1, 0 });

        Array.Fill(hits, (byte)0);
        ulong cycle31 = MersenneDivisorCycles.CalculateCycleLength(31UL);
        session.CheckDivisor(31UL, cycle31, primes, hits);
        hits.Should().ContainInOrder(new byte[] { 1, 0, 0, 0 });
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_gpu_session_reuses_cycle_remainders_across_batches()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester
        {
            GpuBatchSize = 2,
        };

        tester.ConfigureFromMaxPrime(17UL);

        using var session = tester.CreateDivisorSession();
        ulong[] primes = { 5UL, 7UL, 11UL, 13UL, 17UL };
        byte[] hits = new byte[primes.Length];

        ulong cycle23 = MersenneDivisorCycles.CalculateCycleLength(23UL);
        session.CheckDivisor(23UL, cycle23, primes, hits);

        hits.Should().ContainInOrder(new byte[] { 0, 0, 1, 0, 0 });
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_cpu_session_reuses_cycle_remainders_across_primes()
    {
        var tester = new MersenneNumberDivisorByDivisorCpuTester
        {
            BatchSize = 2,
        };

        tester.ConfigureFromMaxPrime(17UL);

        using var session = tester.CreateDivisorSession();
        ulong[] primes = { 5UL, 7UL, 11UL, 13UL, 17UL };
        byte[] hits = new byte[primes.Length];

        ulong cycle23 = MersenneDivisorCycles.CalculateCycleLength(23UL);
        session.CheckDivisor(23UL, cycle23, primes, hits);

        hits.Should().ContainInOrder(new byte[] { 0, 0, 1, 0, 0 });
    }
}

