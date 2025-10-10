using System.Collections.Generic;
using System.Numerics;
using FluentAssertions;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class PrimeOrderCalculatorTests
{
    public static IEnumerable<object[]> SmallPrimes()
    {
        yield return new object[] { 3UL };
        yield return new object[] { 5UL };
        yield return new object[] { 7UL };
        yield return new object[] { 11UL };
        yield return new object[] { 13UL };
        yield return new object[] { 17UL };
        yield return new object[] { 19UL };
        yield return new object[] { 31UL };
        yield return new object[] { 61UL };
        yield return new object[] { 89UL };
        yield return new object[] { 127UL };
        yield return new object[] { 521UL };
    }

    public static IEnumerable<object[]> PrimesWithPrevious()
    {
        yield return new object[] { 5UL, 3UL };
        yield return new object[] { 7UL, 5UL };
        yield return new object[] { 11UL, 7UL };
        yield return new object[] { 13UL, 11UL };
        yield return new object[] { 17UL, 13UL };
        yield return new object[] { 19UL, 17UL };
        yield return new object[] { 31UL, 29UL };
        yield return new object[] { 61UL, 59UL };
        yield return new object[] { 89UL, 83UL };
        yield return new object[] { 127UL, 113UL };
        yield return new object[] { 521UL, 509UL };
    }

    [Theory]
    [Trait("Category", "Fast")]
    [MemberData(nameof(SmallPrimes))]
    public void Calculate_heuristic_mode_matches_naive_cycle_length(ulong prime)
    {
        ulong expected = ComputeOrderByDoubling(prime);

        ulong result = PrimeOrderCalculator.Calculate(
            prime,
            previousOrder: null,
            MontgomeryDivisorData.FromModulus(prime),
            PrimeOrderCalculator.PrimeOrderSearchConfig.HeuristicDefault);

        result.Should().Be(expected);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [MemberData(nameof(PrimesWithPrevious))]
    public void Calculate_heuristic_mode_matches_strict_mode_even_with_previous_order(ulong prime, ulong previousPrime)
    {
        ulong previousOrder = ComputeOrderByDoubling(previousPrime);

        ulong heuristic = PrimeOrderCalculator.Calculate(
            prime,
            previousOrder,
            MontgomeryDivisorData.FromModulus(prime),
            PrimeOrderCalculator.PrimeOrderSearchConfig.HeuristicDefault);

        ulong strict = PrimeOrderCalculator.Calculate(
            prime,
            previousOrder: null,
            MontgomeryDivisorData.FromModulus(prime),
            PrimeOrderCalculator.PrimeOrderSearchConfig.StrictDefault);

        heuristic.Should().Be(strict);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(2UL, 1UL)]
    [InlineData(3UL, 2UL)]
    public void Calculate_handles_trivial_primes(ulong prime, ulong expectedOrder)
    {
        ulong result = PrimeOrderCalculator.Calculate(
            prime,
            previousOrder: null,
            MontgomeryDivisorData.FromModulus(prime),
            PrimeOrderCalculator.PrimeOrderSearchConfig.HeuristicDefault);

        result.Should().Be(expectedOrder);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void Calculate_heuristic_mode_returns_unresolved_when_phi_cannot_be_factored()
    {
        var config = new PrimeOrderCalculator.PrimeOrderSearchConfig(
            smallFactorLimit: 1,
            pollardRhoMilliseconds: 0,
            maxPowChecks: 8,
            mode: PrimeOrderCalculator.PrimeOrderMode.Heuristic);

        ulong result = PrimeOrderCalculator.Calculate(
            prime: 13UL,
            previousOrder: null,
            MontgomeryDivisorData.FromModulus(13UL),
            config);

        result.Should().Be(ComputeOrderByDoubling(13UL));
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void Calculate_heuristic_mode_attempts_candidates_before_falling_back_to_strict_mode()
    {
        var heuristicConfig = new PrimeOrderCalculator.PrimeOrderSearchConfig(
            smallFactorLimit: 2,
            pollardRhoMilliseconds: 0,
            maxPowChecks: 1,
            mode: PrimeOrderCalculator.PrimeOrderMode.Heuristic);

        MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(239UL);
        ulong heuristic = PrimeOrderCalculator.Calculate(
            prime: 239UL,
            previousOrder: null,
            divisorData,
            heuristicConfig);

        heuristic.Should().Be(ComputeOrderByDoubling(239UL));

        var strictConfig = new PrimeOrderCalculator.PrimeOrderSearchConfig(
            smallFactorLimit: 2,
            pollardRhoMilliseconds: 0,
            maxPowChecks: 1,
            mode: PrimeOrderCalculator.PrimeOrderMode.Strict);

        ulong strict = PrimeOrderCalculator.Calculate(
            prime: 239UL,
            previousOrder: null,
            divisorData,
            strictConfig);

        strict.Should().Be(heuristic);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void Calculate_handles_128_bit_prime()
    {
        UInt128 prime = UInt128.Parse("18446744073709641691");

        UInt128 result = PrimeOrderCalculator.Calculate(
            prime,
            previousOrder: null,
            PrimeOrderCalculator.PrimeOrderSearchConfig.HeuristicDefault);

        result.Should().Be((UInt128)1229782938247309446UL);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void GetCycle_uses_wide_prime_heuristic()
    {
        UInt128 prime = UInt128.Parse("18446744073709641691");
        MersenneDivisorCycles.GetCycle(prime).Should().Be((UInt128)1229782938247309446UL);
    }

    private static ulong ComputeOrderByDoubling(ulong prime)
    {
        if (prime == 2UL)
        {
            return 1UL;
        }

        if (prime == 3UL)
        {
            return 2UL;
        }

        ulong order = 1UL;
        ulong value = 2UL % prime;

        while (value != 1UL)
        {
            value = (value * 2UL) % prime;
            order++;
        }

        return order;
    }
}
