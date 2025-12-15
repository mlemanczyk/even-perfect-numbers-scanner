using System.Numerics;
using FluentAssertions;
using PerfectNumbers.Core.Cpu;
using PerfectNumbers.Core.Gpu.Accelerators;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class MersenneNumberDivisorCpuTesterTests
{
	[Fact]
	[Trait("Category", "Fast")]
	public void ByDivisor_tester_tracks_divisors_across_primes()
	{
		var tester = new MersenneNumberDivisorByDivisorCpuTester();
		tester.ConfigureFromMaxPrime(11UL);

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
            tester.IsPrime(gpu, 5UL, out bool divisorsExhausted, out BigInteger divisor).Should().BeTrue();
            divisorsExhausted.Should().BeTrue();
            divisor.Should().Be(BigInteger.Zero);

            tester.IsPrime(gpu, 7UL, out divisorsExhausted, out divisor).Should().BeTrue();
            divisorsExhausted.Should().BeTrue();
            divisor.Should().Be(BigInteger.Zero);

            tester.IsPrime(gpu, 11UL, out divisorsExhausted, out divisor).Should().BeFalse();
            divisorsExhausted.Should().BeTrue();
            divisor.Should().BeGreaterThan(BigInteger.Zero);
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}

	[Fact]
	[Trait("Category", "Fast")]
	public void ByDivisor_session_checks_divisors_across_primes()
	{
		var tester = new MersenneNumberDivisorByDivisorCpuTester();
		tester.ConfigureFromMaxPrime(13UL);

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		var session = tester.CreateDivisorSession(gpu);
		try
		{
			ulong[] primes = [5UL, 7UL, 11UL, 13UL];

			ulong cycle23 = MersenneDivisorCycles.CalculateCycleLengthCpu(23UL, MontgomeryDivisorData.FromModulus(23UL));
			session.CheckDivisor(23UL, MontgomeryDivisorData.FromModulus(23UL), cycle23, primes).Should().BeTrue();

			ulong[] primesWithoutHit = [5UL, 7UL, 13UL];
			session.CheckDivisor(23UL, MontgomeryDivisorData.FromModulus(23UL), cycle23, primesWithoutHit).Should().BeFalse();

			ulong cycle31 = MersenneDivisorCycles.CalculateCycleLengthCpu(31UL, MontgomeryDivisorData.FromModulus(31UL));
			session.CheckDivisor(31UL, MontgomeryDivisorData.FromModulus(31UL), cycle31, primes).Should().BeTrue();

			ulong[] primesBeforeHit = [7UL, 11UL, 13UL];
			session.CheckDivisor(31UL, MontgomeryDivisorData.FromModulus(31UL), cycle31, primesBeforeHit).Should().BeFalse();
		}
		finally
		{
			session.Return();
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}

	[Fact]
	[Trait("Category", "Fast")]
	public void ByDivisor_session_marks_mersenne_numbers_divisible_by_seven_as_composite()
	{
		var tester = new MersenneNumberDivisorByDivisorCpuTester();
		tester.ConfigureFromMaxPrime(43UL);

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		var session = tester.CreateDivisorSession(gpu);
		try
		{
			ulong[] exponents = [6UL, 7UL, 9UL, 10UL, 12UL];

			MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(7UL);
			ulong cycle = MersenneDivisorCycles.CalculateCycleLengthCpu(7UL, divisorData);
			session.CheckDivisor(7UL, divisorData, cycle, exponents).Should().BeTrue();

			ulong[] nonDivisibleExponents = [5UL, 11UL];
			session.CheckDivisor(7UL, divisorData, cycle, nonDivisibleExponents).Should().BeFalse();
		}
		finally
		{
			session.Return();
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}

	[Fact]
	[Trait("Category", "Fast")]
	public void ByDivisor_session_marks_mersenne_numbers_divisible_by_eleven_as_composite()
	{
		var tester = new MersenneNumberDivisorByDivisorCpuTester();
		tester.ConfigureFromMaxPrime(61UL);

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		var session = tester.CreateDivisorSession(gpu);
		try
		{
			ulong[] exponents = [10UL, 11UL, 20UL, 21UL, 30UL];

			MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(11UL);
			ulong cycle = MersenneDivisorCycles.CalculateCycleLengthCpu(11UL, divisorData);
			session.CheckDivisor(11UL, divisorData, cycle, exponents).Should().BeTrue();

			ulong[] nonDivisibleExponents = [5UL, 17UL];
			session.CheckDivisor(11UL, divisorData, cycle, nonDivisibleExponents).Should().BeFalse();
		}
		finally
		{
			session.Return();
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}
}

