using FluentAssertions;
using PerfectNumbers.Core.Gpu;
using Xunit;
using UInt128 = System.UInt128;

namespace PerfectNumbers.Core.Tests;

public class MersenneNumberResidueGpuTesterTests
{
    private static bool LastDigitIsSeven(ulong exponent) => (exponent & 3UL) == 3UL;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    [Trait("Category", "Fast")]
    public void Scan_handles_various_prime_exponents(bool useGpuOrder)
    {
        var tester = new MersenneNumberResidueGpuTester(useGpuOrder);

        RunCase(tester, 23UL, 2UL, expectedPrime: false);
        RunCase(tester, 29UL, 37UL, expectedPrime: false);
        RunCase(tester, 31UL, 1_000UL, expectedPrime: true);
        RunCase(tester, 89UL, 1_001UL, expectedPrime: true);
        RunCase(tester, 107UL, 1_001UL, expectedPrime: true);
        RunCase(tester, 127UL, 1_001UL, expectedPrime: true);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    [Trait("Category", "Fast")]
    public void Scan_handles_multiple_sets_without_false_positives(bool useGpuOrder)
    {
        var tester = new MersenneNumberResidueGpuTester(useGpuOrder);

        bool isPrime = true;
        bool exhausted = false;
        UInt128 perSetLimit = 1_666_667UL;
        UInt128 setCount = 3UL;
        UInt128 overallLimit = 5_000_000UL;

        tester.Scan(
            127UL,
            (UInt128)127UL << 1,
            LastDigitIsSeven(127UL),
            perSetLimit,
            setCount,
            overallLimit,
            ref isPrime,
            ref exhausted);

        isPrime.Should().BeTrue();
        exhausted.Should().BeTrue();

        isPrime = true;
        exhausted = false;
        perSetLimit = 20_000_000UL;
        setCount = 5UL;
        overallLimit = 100_000_000UL;

        tester.Scan(
            107UL,
            (UInt128)107UL << 1,
            LastDigitIsSeven(107UL),
            perSetLimit,
            setCount,
            overallLimit,
            ref isPrime,
            ref exhausted);

        isPrime.Should().BeTrue();
        exhausted.Should().BeTrue();
    }

    private static void RunCase(MersenneNumberResidueGpuTester tester, ulong exponent, ulong maxK, bool expectedPrime)
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

