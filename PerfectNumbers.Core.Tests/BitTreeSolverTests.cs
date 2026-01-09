using System.Numerics;
using FluentAssertions;
using PerfectNumbers.Core.Cpu;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class BitTreeSolverTests
{
    [Fact]
    public void BitTree_FindsKnownDivisor()
    {
        // 2^11-1 = 2047 = 23 * 89, so k=1 yields divisor 23.
        var tester = new MersenneNumberDivisorByDivisorCpuTesterWithForBitTreeDivisorSetForCpuOrder
        {
            DivisorLimit = new BigInteger(1000),
            MinK = BigInteger.One,
        };

        bool isPrime = tester.IsPrime(11, out bool exhausted, out BigInteger divisor);

        isPrime.Should().BeFalse();
        exhausted.Should().BeTrue();
        divisor.Should().Be(23);
    }

    [Fact]
    public void BitTree_ReportsPrimeWhenNoDivisorFound()
    {
        // 2^5-1 = 31 is prime.
        var tester = new MersenneNumberDivisorByDivisorCpuTesterWithForBitTreeDivisorSetForCpuOrder
        {
            DivisorLimit = new BigInteger(10_000),
            MinK = BigInteger.One,
        };

        bool isPrime = tester.IsPrime(5, out bool exhausted, out BigInteger divisor);

        isPrime.Should().BeTrue();
        exhausted.Should().BeTrue();
        divisor.Should().Be(BigInteger.Zero);
    }
}
