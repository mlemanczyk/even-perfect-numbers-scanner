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

    private static void RunCase(MersenneNumberResidueCpuTester tester, ulong exponent, ulong maxK, bool expectedPrime)
    {
        bool isPrime = true;
        bool exhausted = false;
        tester.Scan(
                exponent,
                (UInt128)exponent << 1,
                LastDigitIsSeven(exponent),
                (UInt128)maxK,
                UInt128.One,
                (UInt128)maxK,
                ref isPrime,
                ref exhausted);
        isPrime.Should().Be(expectedPrime);
        exhausted.Should().BeTrue();
    }
}

