using System.Numerics;
using FluentAssertions;
using PerfectNumbers.Core.ByDivisor;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class BitContradictionIntegrationTests
{
    // 2^11 - 1 = 2047 = 23 * 89
    [Fact]
    public void BitContradictionAcceptsKnownDivisorWithDecision()
    {
        BigInteger divisor = 23;
        ulong p = 11;
        bool decided = Invoke(divisor, p, out bool divides);
        decided.Should().BeTrue("small p should be decided exactly");
        divides.Should().BeTrue();
    }

    [Fact]
    public void BitContradictionRejectsEvenDivisorQuickly()
    {
        BigInteger divisor = 24;
        ulong p = 11;
        bool decided = Invoke(divisor, p, out bool divides);
        decided.Should().BeTrue();
        divides.Should().BeFalse();
    }

    [Fact]
    public void BitContradictionRejectsTooLongDivisor()
    {
        // q bit length >= p => impossible
        BigInteger divisor = (BigInteger.One << 15) + 1;
        ulong p = 11;
        bool decided = Invoke(divisor, p, out bool divides);
        decided.Should().BeTrue();
        divides.Should().BeFalse();
    }

    [Fact]
    public void BitContradictionRejectsNonDivisorWithDecision()
    {
        // 2^11 - 1 is not divisible by 25
        BigInteger divisor = 25;
        ulong p = 11;
        bool decided = Invoke(divisor, p, out bool divides);
        decided.Should().BeTrue("small p should be decided exactly");
        divides.Should().BeFalse();
    }

    private static bool Invoke(BigInteger divisor, ulong p, out bool divides)
    {
        int bitLen = GetBitLength(divisor);
        Span<int> offsets = stackalloc int[bitLen];
        int count = 0;
        for (int i = 0; i < bitLen; i++)
        {
            if (((divisor >> i) & BigInteger.One) == BigInteger.One)
            {
                offsets[count++] = i;
            }
        }

        return BitContradictionSolver.TryCheckDivisibilityFromOneOffsets(offsets[..count], p, out divides);
    }

    private static int GetBitLength(BigInteger value)
    {
        if (value.IsZero)
        {
            return 0;
        }

        return (int)Math.Floor(BigInteger.Log(value, 2.0)) + 1;
    }
}
