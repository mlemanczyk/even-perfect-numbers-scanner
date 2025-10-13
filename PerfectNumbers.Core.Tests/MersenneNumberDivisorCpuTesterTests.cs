using System;
using FluentAssertions;
using PerfectNumbers.Core;
using System.Globalization;
using System.Reflection;
using PerfectNumbers.Core.Cpu;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class MersenneNumberDivisorCpuTesterTests
{
    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_tester_tracks_divisors_across_primes()
    {
        var tester = new MersenneNumberDivisorByDivisorCpuTester();
        tester.ConfigureFromMaxPrime(11UL);

        RunIsPrimeWithSnapshot(tester, 5UL, out bool divisorsExhausted);
        divisorsExhausted.Should().BeTrue();

        RunIsPrimeWithSnapshot(tester, 7UL, out divisorsExhausted);
        divisorsExhausted.Should().BeTrue();

        RunIsPrimeWithSnapshot(tester, 11UL, out divisorsExhausted);
        divisorsExhausted.Should().BeTrue();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_prime_test_limit_disables_divisor_exhaustion()
    {
        var tester = new MersenneNumberDivisorByDivisorCpuTester();
        tester.ConfigureFromMaxPrime(11UL);

        RunIsPrimeWithSnapshot(tester, 5UL, out bool divisorsExhausted, TimeSpan.Zero);
        divisorsExhausted.Should().BeFalse();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_session_checks_divisors_across_primes()
    {
        var tester = new MersenneNumberDivisorByDivisorCpuTester();
        tester.ConfigureFromMaxPrime(173UL);

        using var session = tester.CreateDivisorSession();
        ulong[] primes = { 59UL, 61UL, 67UL, 71UL, 73UL };
        byte[] hits = new byte[primes.Length];

        MontgomeryDivisorData divisor223 = MontgomeryDivisorData.FromModulus(223UL);
        ulong cycle223 = MersenneDivisorCycles.CalculateCycleLength(223UL, divisor223);
        session.CheckDivisor(223UL, divisor223, cycle223, primes, hits);

        byte[] expected223 = new byte[primes.Length];
        for (int i = 0; i < primes.Length; i++)
        {
            expected223[i] = primes[i].Pow2ModWindowedCpu(divisor223.Modulus) == 1UL ? (byte)1 : (byte)0;
        }

        hits.Should().Equal(expected223);

        Array.Fill(hits, (byte)0);
        MontgomeryDivisorData divisor431 = MontgomeryDivisorData.FromModulus(431UL);
        ulong cycle431 = MersenneDivisorCycles.CalculateCycleLength(431UL, divisor431);
        session.CheckDivisor(431UL, divisor431, cycle431, primes, hits);

        byte[] expected431 = new byte[primes.Length];
        for (int i = 0; i < primes.Length; i++)
        {
            expected431[i] = primes[i].Pow2ModWindowedCpu(divisor431.Modulus) == 1UL ? (byte)1 : (byte)0;
        }

        hits.Should().Equal(expected431);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_session_marks_mersenne_numbers_divisible_by_223_as_composite()
    {
        var tester = new MersenneNumberDivisorByDivisorCpuTester();
        tester.ConfigureFromMaxPrime(173UL);

        using var session = tester.CreateDivisorSession();
        ulong[] exponents = { 59UL, 67UL, 118UL, 134UL, 177UL };
        byte[] hits = new byte[exponents.Length];

        MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(223UL);
        ulong cycle = MersenneDivisorCycles.CalculateCycleLength(223UL, divisorData);
        session.CheckDivisor(223UL, divisorData, cycle, exponents, hits);

        byte[] expected = new byte[exponents.Length];
        for (int i = 0; i < exponents.Length; i++)
        {
            expected[i] = exponents[i].Pow2ModWindowedCpu(divisorData.Modulus) == 1UL ? (byte)1 : (byte)0;
        }

        hits.Should().Equal(expected);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_session_marks_mersenne_numbers_divisible_by_431_as_composite()
    {
        var tester = new MersenneNumberDivisorByDivisorCpuTester();
        tester.ConfigureFromMaxPrime(193UL);

        using var session = tester.CreateDivisorSession();
        ulong[] exponents = { 67UL, 71UL, 86UL, 134UL, 172UL };
        byte[] hits = new byte[exponents.Length];

        MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(431UL);
        ulong cycle = MersenneDivisorCycles.CalculateCycleLength(431UL, divisorData);
        session.CheckDivisor(431UL, divisorData, cycle, exponents, hits);

        byte[] expected = new byte[exponents.Length];
        for (int i = 0; i < exponents.Length; i++)
        {
            expected[i] = exponents[i].Pow2ModWindowedCpu(divisorData.Modulus) == 1UL ? (byte)1 : (byte)0;
        }

        hits.Should().Equal(expected);
    }
    private static bool RunIsPrimeWithSnapshot(MersenneNumberDivisorByDivisorCpuTester tester, ulong prime, out bool divisorsExhausted, TimeSpan? timeLimit = null)
    {
        while (true)
        {
            try
            {
                return tester.IsPrime(prime, out divisorsExhausted, timeLimit);
            }
            catch (InvalidDataException ex) when (TryPopulateMissingCycle(ex))
            {
            }
        }
    }

    private static bool TryPopulateMissingCycle(InvalidDataException ex)
    {
        const string prefix = "Divisor cycle is missing for ";
        if (!ex.Message.StartsWith(prefix, StringComparison.Ordinal))
        {
            return false;
        }

        string suffix = ex.Message[prefix.Length..];
        if (!ulong.TryParse(suffix, NumberStyles.Integer, CultureInfo.InvariantCulture, out ulong divisor))
        {
            return false;
        }

        var cache = DivisorCycleCache.Shared;
        var snapshotField = typeof(DivisorCycleCache).GetField("_snapshot", BindingFlags.NonPublic | BindingFlags.Instance)!;
        ulong[] snapshot = (ulong[])snapshotField.GetValue(cache)!;
        if (divisor >= (ulong)snapshot.Length)
        {
            return false;
        }

        if (snapshot[divisor] != 0UL)
        {
            return true;
        }

        MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(divisor);
        snapshot[divisor] = MersenneDivisorCycles.CalculateCycleLength(divisor, divisorData);
        return true;
    }
}

