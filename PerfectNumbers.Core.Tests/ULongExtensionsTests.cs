using System.Numerics;
using FluentAssertions;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class ULongExtensionsTests
{
    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(0UL)]
    [InlineData(1UL)]
    [InlineData(9UL)]
    [InlineData(10UL)]
    [InlineData(1234567890UL)]
    [InlineData(ulong.MaxValue)]
    public void Mod10_returns_same_as_operator(ulong value)
    {
        ulong expected = value % 10UL;

        ULongExtensions.Mod10(value).Should().Be(expected);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(0UL)]
    [InlineData(ulong.MaxValue)]
    [InlineData(1234567890123456789UL)]
    public void Mod128_matches_operator(ulong value)
    {
        value.Mod128().Should().Be(value % 128UL);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(1UL)]
    [InlineData(123456789UL)]
    [InlineData(ulong.MaxValue)]
    public void Mod10_8_5_3Steps_matches_manual_formula(ulong value)
    {
        value.Mod10_8_5_3Steps(out ulong step10, out ulong step8, out ulong step5, out ulong step3);

        ulong expected10 = ((value % 10UL) << 1) % 10UL;
        ulong expected8 = ((value & 7UL) << 1) & 7UL;
        ulong expected5 = ((value % 5UL) << 1) % 5UL;
        ulong expected3 = ((value % 3UL) << 1) % 3UL;

        step10.Should().Be(expected10);
        step8.Should().Be(expected8);
        step5.Should().Be(expected5);
        step3.Should().Be(expected3);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(2UL, true)]
    [InlineData(9UL, false)]
    [InlineData(97UL, true)]
    public void IsPrimeCandidate_filters_composites(ulong candidate, bool expected)
    {
        candidate.IsPrimeCandidate().Should().Be(expected);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(3UL)]
    [InlineData(5UL)]
    [InlineData(7UL)]
    [InlineData(11UL)]
    public void CalculateOrder_matches_naive_algorithm(ulong q)
    {
        ulong expected = NaiveOrder(q);

        q.CalculateOrder().Should().Be(expected);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(2UL, 10UL, 17UL)]
    [InlineData(123456789UL, 3UL, 97UL)]
    public void ModPow64_matches_BigInteger(ulong value, ulong exponent, ulong modulus)
    {
        ulong expected = (ulong)BigInteger.ModPow(value, exponent, modulus);

        value.ModPow64(exponent, modulus).Should().Be(expected);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(10UL, 17UL)]
    [InlineData(123UL, 97UL)]
    public void PowMod_returns_same_as_BigInteger(ulong exponent, ulong modulus)
    {
        UInt128 expected = (UInt128)BigInteger.ModPow(2, exponent, modulus);

        exponent.PowMod((UInt128)modulus).Should().Be(expected);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(5UL, 7UL, 3UL, 4UL)]
    [InlineData(10UL, 7UL, 3UL, 2UL)]
    public void PowModWithCycle_uses_cycle_length(ulong exponent, ulong modulus, ulong cycle, ulong expected)
    {
        ((UInt128)exponent).PowModWithCycle((UInt128)modulus, cycle).Should().Be((UInt128)expected);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(15UL, true)]
    [InlineData(7UL, false)]
    [InlineData(9UL, false)]
    public void SharesFactorWithExponentMinusOne_detects_relationship(ulong exponent, bool expected)
    {
        exponent.SharesFactorWithExponentMinusOne().Should().Be(expected);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(1234567890123456789UL, 3UL)]
    [InlineData(ulong.MaxValue, 7UL)]
    public void FastDiv64_matches_division(ulong value, ulong divisor)
    {
        ulong mul = (ulong)(((UInt128)1 << 64) / divisor);

        value.FastDiv64(divisor, mul).Should().Be(value / divisor);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(0UL, 0UL)]
    [InlineData(ulong.MaxValue, 1UL)]
    [InlineData(123456789UL, 987654321UL)]
    public void MulHighCpu_matches_high_bits(ulong x, ulong y)
    {
        ulong expected = (ulong)(((UInt128)x * y) >> 64);

        x.MulHighCpu(y).Should().Be(expected);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(0UL, 0UL)]
    [InlineData(ulong.MaxValue, 1UL)]
    [InlineData(123456789UL, 987654321UL)]
    public void MulHighGpu_matches_cpu_baseline(ulong x, ulong y)
    {
        x.MulHighGpu(y).Should().Be(x.MulHighCpu(y));
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(0UL, 0UL)]
    [InlineData(ulong.MaxValue, 2UL)]
    [InlineData(123456789UL, 987654321UL)]
    public void Mul64_matches_UInt128_product(ulong a, ulong b)
    {
        UInt128 expected = (UInt128)a * b;

        a.Mul64(b).Should().Be(expected);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(123456789UL, 987654321UL, 97UL)]
    [InlineData(ulong.MaxValue, ulong.MaxValue, 1009UL)]
    public void MulMod64_matches_BigInteger(ulong a, ulong b, ulong modulus)
    {
        ulong expected = (ulong)BigInteger.Remainder((BigInteger)a * b, modulus);

        a.MulMod64(b, modulus).Should().Be(expected);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(100UL, 7UL, 3UL)]
    [InlineData(123UL, 9UL, 6UL)]
    public void PowModWithCycle_ulong_overload_uses_cycle(ulong exponent, ulong modulus, ulong cycle)
    {
        UInt128 expected = (UInt128)BigInteger.ModPow(2, exponent, modulus);

        exponent.PowModWithCycle((UInt128)modulus, cycle).Should().Be(expected);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(100UL, 7UL, 3UL)]
    [InlineData(123UL, 9UL, 6UL)]
    public void PowModWithCycle_ulong_uint128_cycle_uses_cycle(ulong exponent, ulong modulus, ulong cycle)
    {
        UInt128 expected = (UInt128)BigInteger.ModPow(2, exponent, modulus);

        exponent.PowModWithCycle((UInt128)modulus, (UInt128)cycle).Should().Be(expected);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(11UL, 13UL)]
    public void PowModCrt_matches_BigInteger(ulong exponent, ulong modulus)
    {
        MersenneDivisorCycles cycles = MersenneDivisorCycles.Shared;
        UInt128 expected = (UInt128)BigInteger.ModPow(2, exponent, modulus);

        exponent.PowModCrt((UInt128)modulus, cycles).Should().Be(expected);
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

}
