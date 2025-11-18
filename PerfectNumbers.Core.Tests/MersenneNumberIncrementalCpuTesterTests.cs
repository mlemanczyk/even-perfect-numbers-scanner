using FluentAssertions;
using PerfectNumbers.Core.Cpu;
using Xunit;
using UInt128 = System.UInt128;

namespace PerfectNumbers.Core.Tests;

public class MersenneNumberIncrementalCpuTesterTests
{
    private static LastDigit GetLastDigit(ulong exponent) => (exponent & 3UL) == 3UL ? LastDigit.Seven : LastDigit.One;

    [Theory]
    [InlineData(GpuKernelType.Incremental)]
    [InlineData(GpuKernelType.Pow2Mod)]
    [Trait("Category", "Fast")]
    public void Scan_handles_various_prime_exponents(GpuKernelType type)
    {
        var tester = new MersenneNumberIncrementalCpuTester(type);

        if (type == GpuKernelType.Pow2Mod)
        {
            RunCase(tester, 11UL, 1UL, expectedPrime: false);
            RunCase(tester, 71UL, 341_860UL, expectedPrime: false);
        }

        RunCase(tester, 89UL, 1_000UL, expectedPrime: true);
        RunCase(tester, 127UL, 1_000UL, expectedPrime: true);
    }

    private static void RunCase(MersenneNumberIncrementalCpuTester tester, ulong exponent, ulong maxK, bool expectedPrime)
    {
        bool isPrime = true;
        tester.Scan(exponent, (UInt128)exponent << 1, GetLastDigit(exponent), (UInt128)maxK, ref isPrime);
        isPrime.Should().Be(expectedPrime);
    }
}
