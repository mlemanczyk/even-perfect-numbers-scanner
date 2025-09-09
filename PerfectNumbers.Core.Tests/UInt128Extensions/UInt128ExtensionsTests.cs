using System.Numerics;
using FluentAssertions;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class UInt128ExtensionsTests
{
    public static IEnumerable<object[]> BinaryGcdCases()
    {
        yield return new object[] { UInt128.Zero, (UInt128)5, (UInt128)5 };
        yield return new object[] { (UInt128)18, (UInt128)24, (UInt128)6 };

        UInt128 big1 = (UInt128.One << 96) + 12345;
        UInt128 big2 = (UInt128.One << 80) + 67890;
        yield return new object[]
        {
            big1,
            big2,
            (UInt128)BigInteger.GreatestCommonDivisor((BigInteger)big1, (BigInteger)big2)
        };
    }

    [Theory]
    [Trait("Category", "Fast")]
    [MemberData(nameof(BinaryGcdCases))]
    public void BinaryGcd_matches_BigInteger(UInt128 a, UInt128 b, UInt128 expected)
    {
        a.BinaryGcd(b).Should().Be(expected);
    }

    public static IEnumerable<object[]> TrailingZeroCases()
    {
        yield return new object[] { UInt128.Zero, 128 };
        yield return new object[] { (UInt128)1, 0 };
        yield return new object[] { (UInt128)8, 3 };
        yield return new object[] { UInt128.One << 80, 80 };
    }

    [Theory]
    [Trait("Category", "Fast")]
    [MemberData(nameof(TrailingZeroCases))]
    public void CountTrailingZeros_returns_expected(UInt128 value, int expected)
    {
        value.CountTrailingZeros().Should().Be(expected);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(3UL)]
    [InlineData(5UL)]
    [InlineData(7UL)]
    public void CalculateOrder_matches_naive(ulong q)
    {
        UInt128 q128 = q;
        ulong expected = NaiveOrder(q);
        q128.CalculateOrder().Should().Be(expected);
    }

    private static ulong NaiveOrder(ulong q)
    {
        if (q <= 2UL)
        {
            return 0UL;
        }

        ulong order = 1UL;
        ulong value = 2UL % q;
        while (value != 1UL)
        {
            value = (value << 1) % q;
            order++;
        }

        return order;
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(2UL, true)]
    [InlineData(9UL, false)]
    [InlineData(97UL, true)]
    public void IsPrimeCandidate_checks_small(ulong n, bool expected)
    {
        ((UInt128)n).IsPrimeCandidate().Should().Be(expected);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void IsPrimeCandidate_detects_large_composite()
    {
        UInt128 n = (UInt128.One << 64) + 1;
        n.IsPrimeCandidate().Should().BeFalse();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void Mod10_8_5_3_matches_operator()
    {
        UInt128 value = (UInt128.One << 100) + 12345678901234567890UL;

        value.Mod10_8_5_3(out ulong mod10, out ulong mod8, out ulong mod5, out ulong mod3);

        ulong expected10 = (ulong)(value % 10);
        ulong expected8 = (ulong)(value % 8);
        ulong expected5 = (ulong)(value % 5);
        ulong expected3 = (ulong)(value % 3);

        mod10.Should().Be(expected10);
        mod8.Should().Be(expected8);
        mod5.Should().Be(expected5);
        mod3.Should().Be(expected3);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(2UL, 10UL, 17UL)]
    [InlineData(123456789UL, 3UL, 97UL)]
    public void ModPow_matches_BigInteger(ulong value, ulong exponent, ulong modulus)
    {
        UInt128 v = value;
        UInt128 e = exponent;
        UInt128 m = modulus;

        UInt128 expected = (UInt128)BigInteger.ModPow(value, exponent, modulus);

        v.ModPow(e, m).Should().Be(expected);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ModPow_handles_large_values()
    {
        UInt128 v = (UInt128.One << 96) + 12345;
        UInt128 e = 17;
        UInt128 m = 97;

        UInt128 expected = (UInt128)BigInteger.ModPow((BigInteger)v, (BigInteger)e, (BigInteger)m);

        v.ModPow(e, m).Should().Be(expected);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(0UL)]
    [InlineData(1UL)]
    [InlineData(123456789UL)]
    public void Mod10_matches_operator(ulong low)
    {
        UInt128 value = (UInt128.One << 100) + low;
        value.Mod10().Should().Be((ulong)(value % 10));
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void Mod8_matches_operator()
    {
        UInt128 value = (UInt128.One << 100) + 12345;
        value.Mod8().Should().Be((ulong)(value % 8));
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void Mod3_matches_operator()
    {
        UInt128 value = (UInt128.One << 100) + 12345;
        value.Mod3().Should().Be((ulong)(value % 3));
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void Mod5_matches_operator()
    {
        UInt128 value = (UInt128.One << 100) + 12345;
        value.Mod5().Should().Be((ulong)(value % 5));
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void Mul64_matches_expected_product()
    {
        UInt128 a = (UInt128.One << 80) + 123456789;
        UInt128 b = (UInt128.One << 64) + 987654321;

        UInt128 expected = (UInt128)((UInt128)(ulong)a * b);

        a.Mul64(b).Should().Be(expected);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(123UL, 456UL, 789UL)]
    [InlineData(123456789UL, 987654321UL, 1009UL)]
    public void MulMod_matches_BigInteger(ulong a, ulong b, ulong modulus)
    {
        UInt128 ua = a % modulus;
        UInt128 ub = b % modulus;
        UInt128 mod = modulus;

        UInt128 expected = (UInt128)(((BigInteger)ua * (BigInteger)ub) % modulus);

        ua.MulMod(ub, mod).Should().Be(expected);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void MulMod_handles_large_inputs()
    {
        UInt128 a = (UInt128.One << 80) + 12345;
        UInt128 b = (UInt128.One << 96) + 67890;
        UInt128 mod = (UInt128.One << 100) - 93;

        UInt128 expected = (UInt128)(((BigInteger)a * (BigInteger)b) % (BigInteger)mod);

        a.MulMod(b, mod).Should().Be(expected);
    }
}

