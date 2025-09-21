using FluentAssertions;
using PerfectNumbers.Core.Cpu;
using Xunit;
using UInt128 = System.UInt128;

namespace PerfectNumbers.Core.Tests;

public class MersenneNumberResidueCpuTesterTests
{
    private static bool LastDigitIsSeven(ulong exponent) => (exponent & 3UL) == 3UL;

    [Fact]
    [Trait("Category", "Fast")]
    public void Scan_handles_various_prime_exponents()
    {
        var tester = new MersenneNumberResidueCpuTester();

        RunCase(tester, 23UL, 1UL, expectedPrime: false);
        RunCase(tester, 29UL, 36UL, expectedPrime: false);
        RunCase(tester, 89UL, 1_000UL, expectedPrime: true);
        RunCase(tester, 127UL, 1_000UL, expectedPrime: true);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void Scan_recognizes_all_known_small_mersenne_primes()
    {
        var tester = new MersenneNumberResidueCpuTester();

        foreach (ulong exponent in MersennePrimeTestData.Exponents)
        {
            RunCase(tester, exponent, 1UL, expectedPrime: true);
        }
    }

    [Theory]
    [InlineData(31UL)]
    [Trait("Category", "Fast")]
    public void Scan_matches_cli_residue_configuration_for_primes(ulong exponent)
    {
        RunCase(new MersenneNumberResidueCpuTester(), exponent, 32UL, expectedPrime: true);
    }

    [Theory]
    [InlineData(31UL, 1_024UL, 1UL)]
    [InlineData(31UL, 512UL, 2UL)]
    [InlineData(31UL, 256UL, 4UL)]
    [InlineData(127UL, 256UL, 1UL)]
    [InlineData(127UL, 128UL, 2UL)]
    [InlineData(127UL, 64UL, 4UL)]
    [Trait("Category", "Fast")]
    public void Scan_handles_multiple_residue_sets_for_known_primes(ulong exponent, ulong perSetLimit, ulong setCount)
    {
        ulong overallLimit = perSetLimit * setCount;
        RunCase(new MersenneNumberResidueCpuTester(), exponent, perSetLimit, setCount, overallLimit, expectedPrime: true);
    }

    private static void RunCase(MersenneNumberResidueCpuTester tester, ulong exponent, ulong maxK, bool expectedPrime)
    {
        RunCase(tester, exponent, maxK, 1UL, maxK, expectedPrime);
    }

    private static void RunCase(
        MersenneNumberResidueCpuTester tester,
        ulong exponent,
        ulong perSetLimit,
        ulong setCount,
        ulong overallLimit,
        bool expectedPrime)
    {
        bool isPrime = true;
        tester.Scan(exponent, (UInt128)exponent << 1, LastDigitIsSeven(exponent), (UInt128)maxK, ref isPrime);
        isPrime.Should().Be(expectedPrime);
    }
}

