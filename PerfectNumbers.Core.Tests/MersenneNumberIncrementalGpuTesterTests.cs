using FluentAssertions;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;
using Xunit;
using UInt128 = System.UInt128;

namespace PerfectNumbers.Core.Tests;

public class MersenneNumberIncrementalGpuTesterTests
{
    private static LastDigit GetLastDigit(ulong exponent) => (exponent & 3UL) == 3UL ? LastDigit.Seven : LastDigit.One;

    [Theory(Skip = "Disabled until the GPU incremental cycle heuristics are ported.")]
    [InlineData(GpuKernelType.Pow2Mod, false)]
    [InlineData(GpuKernelType.Incremental, false)]
    [InlineData(GpuKernelType.Pow2Mod, true)]
    [Trait("Category", "Fast")]
    public void Scan_handles_various_prime_exponents(GpuKernelType type, bool useGpuOrder)
    {
        var tester = new MersenneNumberIncrementalGpuTester(type, useGpuOrder);

        if (type == GpuKernelType.Pow2Mod)
        {
            // Mirrored by the divisor GPU coverage to ensure the failing cases stay validated.
            RunCase(tester, 37UL, 1UL, expectedPrime: false);
            RunCase(tester, 71UL, 341_860UL, expectedPrime: false);
        }

        RunCase(tester, 89UL, 1_000UL, expectedPrime: true);
        RunCase(tester, 127UL, 1_000UL, expectedPrime: true);
    }

    private static void RunCase(MersenneNumberIncrementalGpuTester tester, ulong exponent, ulong maxK, bool expectedPrime)
    {
        bool isPrime = true;
        tester.Scan(exponent, (UInt128)exponent << 1, GetLastDigit(exponent), (UInt128)maxK, ref isPrime);
        isPrime.Should().Be(expectedPrime);
    }
}
