using System.Numerics;
using FluentAssertions;
using PerfectNumbers.Core.Cpu;
using MersenneNumberDivisorByDivisorCpuTesterWithCpuOrder = PerfectNumbers.Core.Cpu.MersenneNumberDivisorByDivisorCpuTesterWithForOneByOneDivisorSetForCpuOrder;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class MersenneNumberDivisorCpuTesterTests
{
    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_tester_tracks_divisors_across_primes()
    {
        var tester = new MersenneNumberDivisorByDivisorCpuTesterWithCpuOrder();

        ulong[] primes = [5UL, 7UL, 11UL];
        ulong[] allowedMax = new ulong[primes.Length];
        tester.PrepareCandidates(maxPrime: 11UL, primes, allowedMax);

        tester.ResumeFromState(Path.Combine(Path.GetTempPath(), "bydivisor-cpu-tests.bin"), BigInteger.Zero, BigInteger.One);
        tester.ResetStateTracking();

        tester.IsPrime(5UL, out bool divisorsExhausted, out BigInteger divisor).Should().BeTrue();
        divisorsExhausted.Should().BeTrue();
        divisor.Should().Be(BigInteger.Zero);

        tester.IsPrime(7UL, out divisorsExhausted, out divisor).Should().BeTrue();
        divisorsExhausted.Should().BeTrue();
        divisor.Should().Be(BigInteger.Zero);

        tester.IsPrime(11UL, out divisorsExhausted, out divisor).Should().BeFalse();
        divisorsExhausted.Should().BeTrue();
        divisor.Should().BeGreaterThan(BigInteger.Zero);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_session_checks_divisors_across_primes()
    {
        var session = new MersenneCpuDivisorScanSessionWithCpuOrder();

        ulong[] primes = [5UL, 7UL, 11UL, 13UL];

        ulong cycle23 = MersenneDivisorCyclesCpu.CalculateCycleLength(23UL, MontgomeryDivisorData.FromModulus(23UL));
        session.CheckDivisor(23UL, MontgomeryDivisorData.FromModulus(23UL), cycle23, primes).Should().BeTrue();

        ulong[] primesWithoutHit = [5UL, 7UL, 13UL];
        session.CheckDivisor(23UL, MontgomeryDivisorData.FromModulus(23UL), cycle23, primesWithoutHit).Should().BeFalse();

        ulong cycle31 = MersenneDivisorCyclesCpu.CalculateCycleLength(31UL, MontgomeryDivisorData.FromModulus(31UL));
        session.CheckDivisor(31UL, MontgomeryDivisorData.FromModulus(31UL), cycle31, primes).Should().BeTrue();

        ulong[] primesBeforeHit = [7UL, 11UL, 13UL];
        session.CheckDivisor(31UL, MontgomeryDivisorData.FromModulus(31UL), cycle31, primesBeforeHit).Should().BeFalse();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_session_marks_mersenne_numbers_divisible_by_seven_as_composite()
    {
        var session = new MersenneCpuDivisorScanSessionWithCpuOrder();

        ulong[] exponents = [6UL, 7UL, 9UL, 10UL, 12UL];

        MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(7UL);
        ulong cycle = MersenneDivisorCyclesCpu.CalculateCycleLength(7UL, divisorData);
        session.CheckDivisor(7UL, divisorData, cycle, exponents).Should().BeTrue();

        ulong[] nonDivisibleExponents = [5UL, 11UL];
        session.CheckDivisor(7UL, divisorData, cycle, nonDivisibleExponents).Should().BeFalse();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_session_marks_mersenne_numbers_divisible_by_eleven_as_composite()
    {
        var session = new MersenneCpuDivisorScanSessionWithCpuOrder();

        ulong[] exponents = [10UL, 11UL, 20UL, 21UL, 30UL];

        MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(11UL);
        ulong cycle = MersenneDivisorCyclesCpu.CalculateCycleLength(11UL, divisorData);
        session.CheckDivisor(11UL, divisorData, cycle, exponents).Should().BeTrue();

        ulong[] nonDivisibleExponents = [5UL, 17UL];
        session.CheckDivisor(11UL, divisorData, cycle, nonDivisibleExponents).Should().BeFalse();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_does_not_mark_mersenne_prime_as_composite_via_self_divisor()
    {
        var tester = new MersenneNumberDivisorByDivisorCpuTesterWithCpuOrder();

        tester.DivisorLimit = (BigInteger.One << 30) - BigInteger.One;
        BigInteger kSelf = ((BigInteger.One << 31) - 2) / 62; // (2^p - 2) / (2p) for p=31.

        tester.ResumeFromState(Path.Combine(Path.GetTempPath(), "bydivisor-selfdivisor.bin"), BigInteger.Zero, kSelf);
        tester.ResetStateTracking();

        tester.IsPrime(31UL, out bool divisorsExhausted, out BigInteger divisor).Should().BeTrue();
        divisorsExhausted.Should().BeTrue();
        divisor.Should().Be(BigInteger.Zero);
    }

}

