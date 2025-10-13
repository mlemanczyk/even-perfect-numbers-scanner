using System;
using FluentAssertions;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Cpu;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class MersenneNumberDivisorCpuTesterTests
{
    private const ulong LargeDivisor1 = 4_000_169UL;
    private const ulong LargeDivisor2 = 4_000_561UL;

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_tester_tracks_divisors_across_primes()
    {
        var tester = new MersenneNumberDivisorByDivisorCpuTester();
        tester.ConfigureFromMaxPrime(83UL);

        tester.IsPrime(59UL, out bool divisorsExhausted).Should().BeTrue();
        divisorsExhausted.Should().BeTrue();

        tester.IsPrime(61UL, out divisorsExhausted).Should().BeTrue();
        divisorsExhausted.Should().BeTrue();

        tester.IsPrime(67UL, out divisorsExhausted).Should().BeTrue();
        divisorsExhausted.Should().BeTrue();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_prime_test_limit_disables_divisor_exhaustion()
    {
        var tester = new MersenneNumberDivisorByDivisorCpuTester();
        tester.ConfigureFromMaxPrime(83UL);

        tester.IsPrime(59UL, out bool divisorsExhausted, TimeSpan.Zero).Should().BeTrue();
        divisorsExhausted.Should().BeFalse();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_session_checks_divisors_across_exponents()
    {
        var tester = new MersenneNumberDivisorByDivisorCpuTester();
        tester.ConfigureFromMaxPrime(197UL);

        var session = tester.CreateDivisorSession();
        ulong[] exponents = { 59UL, 61UL, 67UL, 71UL, 73UL, 79UL, 117_652UL, 333_380UL };
        byte[] hits = new byte[exponents.Length];

        MontgomeryDivisorData divisor1 = MontgomeryDivisorData.FromModulus(LargeDivisor1);
        ulong cycle1 = MersenneDivisorCycles.CalculateCycleLength(LargeDivisor1, divisor1);
        session.CheckDivisor(LargeDivisor1, divisor1, cycle1, exponents, hits);

        byte[] expected1 = new byte[exponents.Length];
        for (int i = 0; i < exponents.Length; i++)
        {
            expected1[i] = exponents[i].Pow2ModWindowedCpu(divisor1.Modulus) == 1UL ? (byte)1 : (byte)0;
        }

        hits.Should().Equal(expected1);

        Array.Fill(hits, (byte)0);
        MontgomeryDivisorData divisor2 = MontgomeryDivisorData.FromModulus(LargeDivisor2);
        ulong cycle2 = MersenneDivisorCycles.CalculateCycleLength(LargeDivisor2, divisor2);
        session.CheckDivisor(LargeDivisor2, divisor2, cycle2, exponents, hits);

        byte[] expected2 = new byte[exponents.Length];
        for (int i = 0; i < exponents.Length; i++)
        {
            expected2[i] = exponents[i].Pow2ModWindowedCpu(divisor2.Modulus) == 1UL ? (byte)1 : (byte)0;
        }

        hits.Should().Equal(expected2);

        session.Dispose();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_session_marks_mersenne_numbers_divisible_by_large_divisor1_as_composite()
    {
        var tester = new MersenneNumberDivisorByDivisorCpuTester();
        tester.ConfigureFromMaxPrime(197UL);

        var session = tester.CreateDivisorSession();
        MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(LargeDivisor1);
        ulong cycle = MersenneDivisorCycles.CalculateCycleLength(LargeDivisor1, divisorData);

        ulong[] exponents = { 59UL, 67UL, cycle, cycle * 2UL, cycle + 1UL };
        byte[] hits = new byte[exponents.Length];
        session.CheckDivisor(LargeDivisor1, divisorData, cycle, exponents, hits);

        byte[] expected = new byte[exponents.Length];
        for (int i = 0; i < exponents.Length; i++)
        {
            expected[i] = exponents[i].Pow2ModWindowedCpu(divisorData.Modulus) == 1UL ? (byte)1 : (byte)0;
        }

        hits.Should().Equal(expected);

        session.Dispose();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_session_marks_mersenne_numbers_divisible_by_large_divisor2_as_composite()
    {
        var tester = new MersenneNumberDivisorByDivisorCpuTester();
        tester.ConfigureFromMaxPrime(197UL);

        var session = tester.CreateDivisorSession();
        MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(LargeDivisor2);
        ulong cycle = MersenneDivisorCycles.CalculateCycleLength(LargeDivisor2, divisorData);

        ulong[] exponents = { 61UL, 73UL, cycle, cycle * 2UL, cycle - 1UL };
        byte[] hits = new byte[exponents.Length];
        session.CheckDivisor(LargeDivisor2, divisorData, cycle, exponents, hits);

        byte[] expected = new byte[exponents.Length];
        for (int i = 0; i < exponents.Length; i++)
        {
            expected[i] = exponents[i].Pow2ModWindowedCpu(divisorData.Modulus) == 1UL ? (byte)1 : (byte)0;
        }

        hits.Should().Equal(expected);

        session.Dispose();
    }
}
