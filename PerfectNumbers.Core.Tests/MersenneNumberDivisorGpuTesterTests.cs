using System;
using System.Collections.Generic;
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
    [InlineData(117_652UL, 4_000_169UL, true)]
    [InlineData(333_380UL, 4_000_561UL, true)]
    [InlineData(59UL, 4_000_169UL, false)]
    [InlineData(61UL, 4_000_561UL, false)]
    [InlineData(333_379UL, 4_000_561UL, false)]
    [InlineData(117_653UL, 4_000_169UL, false)]
    [Trait("Category", "Fast")]
    public void IsDivisible_returns_expected(ulong exponent, ulong divisor, bool expected)
    {
        var tester = new MersenneNumberDivisorGpuTester();
        ReadOnlyGpuUInt128 divisorValue = new ReadOnlyGpuUInt128(divisor);
        tester.IsDivisible(exponent, in divisorValue).Should().Be(expected);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void IsDivisible_handles_large_divisors()
    {
        var tester = new MersenneNumberDivisorGpuTester();
        UInt128 divisor = (UInt128.One << 65) - UInt128.One;
        ReadOnlyGpuUInt128 divisorValue = new ReadOnlyGpuUInt128(divisor);
        tester.IsDivisible(65UL, in divisorValue).Should().BeTrue();
        divisorValue = new ReadOnlyGpuUInt128(divisor + 2); // Reusing divisorValue for the shifted candidate.
        tester.IsDivisible(65UL, in divisorValue).Should().BeFalse();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void IsPrime_sets_divisorsExhausted_false_when_search_range_not_exhausted()
    {
        var tester = new MersenneNumberDivisorGpuTester();
        typeof(MersenneNumberDivisorGpuTester)
            .GetField("_divisorCandidates", BindingFlags.NonPublic | BindingFlags.Static)!
            .SetValue(null, Array.Empty<(ulong, uint)>());

        tester.IsPrime(31UL, UInt128.Zero, 0UL, out bool divisorsExhausted).Should().BeTrue();
        divisorsExhausted.Should().BeFalse();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void IsPrime_sets_divisorsExhausted_true_when_divisible_by_specified_divisor()
    {
        var tester = new MersenneNumberDivisorGpuTester();
        tester.IsPrime(35UL, 31UL, 0UL, out bool divisorsExhausted).Should().BeFalse();
        divisorsExhausted.Should().BeTrue();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void IsPrime_accepts_large_search_limits()
    {
        var tester = new MersenneNumberDivisorGpuTester();
        tester.IsPrime(35UL, 31UL, ulong.MaxValue, out bool exhausted).Should().BeFalse();
        exhausted.Should().BeTrue();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_tester_tracks_divisors_across_primes()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester();
        tester.ConfigureFromMaxPrime(43UL);

        tester.IsPrime(31UL, out bool exhausted).Should().BeTrue();
        exhausted.Should().BeTrue();

        tester.IsPrime(37UL, out exhausted).Should().BeFalse();
        exhausted.Should().BeTrue();

        tester.IsPrime(43UL, out exhausted).Should().BeFalse();
        exhausted.Should().BeTrue();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_gpu_prime_test_limit_stops_scan_immediately()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester();
        tester.ConfigureFromMaxPrime(43UL);

        tester.IsPrime(31UL, out bool divisorsExhausted, TimeSpan.Zero).Should().BeTrue();
        divisorsExhausted.Should().BeFalse();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_tester_requires_configuration()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester();
        Action act = () => tester.IsPrime(31UL, out _);
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_tester_respects_filter_based_limit()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester();
        tester.ConfigureFromMaxPrime(41UL);

        tester.IsPrime(31UL, out _).Should().BeTrue();
        tester.IsPrime(37UL, out bool exhausted).Should().BeFalse();
        exhausted.Should().BeTrue();

        tester.IsPrime(41UL, out bool divisorsExhausted).Should().BeFalse();
        divisorsExhausted.Should().BeTrue();
    }


    [Theory]
    [InlineData(71UL, false)]
    [InlineData(89UL, true)]
    [InlineData(127UL, true)]
    [Trait("Category", "Fast")]
    public void ByDivisor_tester_matches_incremental_expectations(ulong exponent, bool expectedPrime)
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester();
        tester.ConfigureFromMaxPrime(exponent);

        tester.IsPrime(exponent, out bool exhausted).Should().Be(expectedPrime);
        exhausted.Should().BeTrue();
    }
    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_gpu_tester_uses_cycle_remainders_per_divisor()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester
        {
            GpuBatchSize = 8,
        };

        tester.ConfigureFromMaxPrime(43UL);

        tester.IsPrime(41UL, out bool divisorsExhausted).Should().BeFalse();
        divisorsExhausted.Should().BeTrue();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_session_checks_divisors_across_exponents()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester();
        tester.ConfigureFromMaxPrime(197UL);

        using var session = tester.CreateDivisorSession();
        ulong[] exponents = { 59UL, 61UL, 67UL, 71UL, 73UL, 79UL, 117_652UL, 333_380UL };
        byte[] hits = new byte[exponents.Length];

        MontgomeryDivisorData divisor1 = MontgomeryDivisorData.FromModulus(4_000_169UL);
        ulong cycle1 = MersenneDivisorCycles.CalculateCycleLength(4_000_169UL, divisor1);
        session.CheckDivisor(4_000_169UL, divisor1, cycle1, exponents, hits);

        byte[] expected1 = new byte[exponents.Length];
        for (int i = 0; i < exponents.Length; i++)
        {
            expected1[i] = exponents[i].Pow2ModWindowedCpu(divisor1.Modulus) == 1UL ? (byte)1 : (byte)0;
        }

        hits.Should().Equal(expected1);

        Array.Fill(hits, (byte)0);
        MontgomeryDivisorData divisor2 = MontgomeryDivisorData.FromModulus(4_000_561UL);
        ulong cycle2 = MersenneDivisorCycles.CalculateCycleLength(4_000_561UL, divisor2);
        session.CheckDivisor(4_000_561UL, divisor2, cycle2, exponents, hits);

        byte[] expected2 = new byte[exponents.Length];
        for (int i = 0; i < exponents.Length; i++)
        {
            expected2[i] = exponents[i].Pow2ModWindowedCpu(divisor2.Modulus) == 1UL ? (byte)1 : (byte)0;
        }

        hits.Should().Equal(expected2);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_gpu_session_reuses_cycle_remainders_across_batches()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester
        {
            GpuBatchSize = 2,
        };

        tester.ConfigureFromMaxPrime(197UL);

        using var session = tester.CreateDivisorSession();
        ulong[] exponents = { 59UL, 61UL, 67UL, 71UL, 73UL, 79UL, 117_652UL };
        byte[] hits = new byte[exponents.Length];

        MontgomeryDivisorData divisor = MontgomeryDivisorData.FromModulus(4_000_169UL);
        ulong cycle = MersenneDivisorCycles.CalculateCycleLength(4_000_169UL, divisor);
        session.CheckDivisor(4_000_169UL, divisor, cycle, exponents, hits);

        byte[] expected = new byte[exponents.Length];
        for (int i = 0; i < exponents.Length; i++)
        {
            expected[i] = exponents[i].Pow2ModWindowedCpu(divisor.Modulus) == 1UL ? (byte)1 : (byte)0;
        }

        hits.Should().Equal(expected);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_session_marks_mersenne_numbers_divisible_by_large_divisor1_as_composite()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester
        {
            GpuBatchSize = 5,
        };
        tester.ConfigureFromMaxPrime(197UL);

        using var session = tester.CreateDivisorSession();
        MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(4_000_169UL);
        ulong cycle = MersenneDivisorCycles.CalculateCycleLength(4_000_169UL, divisorData);

        ulong[] exponents = { 59UL, 67UL, cycle, cycle * 2UL, cycle + 1UL };
        byte[] hits = new byte[exponents.Length];

        session.CheckDivisor(4_000_169UL, divisorData, cycle, exponents, hits);

        var hostBufferField = typeof(MersenneNumberDivisorByDivisorGpuTester.DivisorScanSession)
            .GetField("_hostBuffer", BindingFlags.NonPublic | BindingFlags.Instance)!;
        ulong[] hostBuffer = (ulong[])hostBufferField.GetValue(session)!;
        Span<ulong> residues = hostBuffer.AsSpan(0, exponents.Length);
        byte[] expected = new byte[exponents.Length];
        for (int i = 0; i < residues.Length; i++)
        {
            expected[i] = residues[i] == 1UL ? (byte)1 : (byte)0;
        }

        byte[] direct = new byte[exponents.Length];
        for (int i = 0; i < exponents.Length; i++)
        {
            direct[i] = exponents[i].Pow2ModWindowedCpu(divisorData.Modulus) == 1UL ? (byte)1 : (byte)0;
        }

        expected.Should().Equal(direct);
        hits.Should().Equal(direct);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_session_marks_mersenne_numbers_divisible_by_large_divisor2_as_composite()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester
        {
            GpuBatchSize = 5,
        };
        tester.ConfigureFromMaxPrime(197UL);

        using var session = tester.CreateDivisorSession();
        MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(4_000_561UL);
        ulong cycle = MersenneDivisorCycles.CalculateCycleLength(4_000_561UL, divisorData);

        ulong[] exponents = { 61UL, 73UL, cycle, cycle * 2UL, cycle - 1UL };
        byte[] hits = new byte[exponents.Length];

        session.CheckDivisor(4_000_561UL, divisorData, cycle, exponents, hits);

        var hostBufferField = typeof(MersenneNumberDivisorByDivisorGpuTester.DivisorScanSession)
            .GetField("_hostBuffer", BindingFlags.NonPublic | BindingFlags.Instance)!;
        ulong[] hostBuffer = (ulong[])hostBufferField.GetValue(session)!;
        Span<ulong> residues = hostBuffer.AsSpan(0, exponents.Length);
        byte[] expected = new byte[exponents.Length];
        for (int i = 0; i < residues.Length; i++)
        {
            expected[i] = residues[i] == 1UL ? (byte)1 : (byte)0;
        }

        byte[] direct = new byte[exponents.Length];
        for (int i = 0; i < exponents.Length; i++)
        {
            direct[i] = exponents[i].Pow2ModWindowedCpu(divisorData.Modulus) == 1UL ? (byte)1 : (byte)0;
        }

        expected.Should().Equal(direct);
        hits.Should().Equal(direct);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_gpu_tester_generates_cycles_for_large_divisors()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester();
        tester.ConfigureFromMaxPrime(197UL);

        using var session = tester.CreateDivisorSession();
        ulong[] exponents = { 59UL, 67UL, 117_652UL };
        byte[] hits = new byte[exponents.Length];

        MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(4_000_169UL);
        ulong cycle = MersenneDivisorCycles.CalculateCycleLength(4_000_169UL, divisorData);

        session.CheckDivisor(4_000_169UL, divisorData, cycle, exponents, hits);

        byte[] expected = new byte[exponents.Length];
        for (int i = 0; i < exponents.Length; i++)
        {
            expected[i] = exponents[i].Pow2ModWindowedCpu(divisorData.Modulus) == 1UL ? (byte)1 : (byte)0;
        }

        hits.Should().Equal(expected);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_cpu_session_reuses_cycle_remainders_across_primes()
    {
        var tester = new MersenneNumberDivisorByDivisorCpuTester
        {
            BatchSize = 2,
        };

        tester.ConfigureFromMaxPrime(197UL);

        using var session = tester.CreateDivisorSession();
        ulong[] exponents = { 59UL, 61UL, 67UL, 71UL, 73UL, 79UL, 117_652UL };
        byte[] hits = new byte[exponents.Length];

        MontgomeryDivisorData divisor = MontgomeryDivisorData.FromModulus(4_000_169UL);
        ulong cycle = MersenneDivisorCycles.CalculateCycleLength(4_000_169UL, divisor);
        session.CheckDivisor(4_000_169UL, divisor, cycle, exponents, hits);

        byte[] expected = new byte[exponents.Length];
        for (int i = 0; i < exponents.Length; i++)
        {
            expected[i] = exponents[i].Pow2ModWindowedCpu(divisor.Modulus) == 1UL ? (byte)1 : (byte)0;
        }

        hits.Should().Equal(expected);
    }
}

