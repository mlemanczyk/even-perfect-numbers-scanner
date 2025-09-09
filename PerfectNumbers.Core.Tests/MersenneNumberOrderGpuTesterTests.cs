using FluentAssertions;
using PerfectNumbers.Core.Gpu;
using Xunit;
using UInt128 = System.UInt128;

namespace PerfectNumbers.Core.Tests;

public class MersenneNumberOrderGpuTesterTests
{
    private static bool LastDigitIsSeven(ulong exponent) => (exponent & 3UL) == 3UL;

    [Theory]
    [InlineData(GpuKernelType.Pow2Mod, false)]
    [InlineData(GpuKernelType.Incremental, false)]
    [InlineData(GpuKernelType.Pow2Mod, true)]
    [Trait("Category", "Fast")]
    public void Scan_handles_various_prime_exponents(GpuKernelType type, bool useGpuOrder)
    {
        var tester = new MersenneNumberOrderGpuTester(type, useGpuOrder);

        if (type == GpuKernelType.Pow2Mod)
        {
            RunCase(tester, 23UL, 1UL, expectedPrime: false);
            RunCase(tester, 29UL, 36UL, expectedPrime: false);
        }

        RunCase(tester, 89UL, 1_000UL, expectedPrime: true);
        RunCase(tester, 127UL, 1_000UL, expectedPrime: true);
    }

    private static void RunCase(MersenneNumberOrderGpuTester tester, ulong exponent, ulong maxK, bool expectedPrime)
    {
        bool isPrime = true;
        tester.Scan(exponent, (UInt128)exponent << 1, LastDigitIsSeven(exponent), (UInt128)maxK, ref isPrime);
        isPrime.Should().Be(expectedPrime);
    }
}

