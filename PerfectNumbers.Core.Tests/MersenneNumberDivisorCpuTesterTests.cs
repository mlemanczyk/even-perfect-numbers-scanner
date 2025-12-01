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
			tester.IsPrime(gpu, 5UL, out bool divisorsExhausted, out ulong divisor).Should().BeTrue();
			divisorsExhausted.Should().BeTrue();
			divisor.Should().Be(0UL);

			tester.IsPrime(gpu, 7UL, out divisorsExhausted, out divisor).Should().BeTrue();
			divisorsExhausted.Should().BeTrue();
			divisor.Should().Be(0UL);

			tester.IsPrime(gpu, 11UL, out divisorsExhausted, out divisor).Should().BeFalse();
			divisorsExhausted.Should().BeTrue();
			divisor.Should().BeGreaterThan(0UL);
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
			byte[] hits = new byte[primes.Length];

			ulong cycle23 = MersenneDivisorCycles.CalculateCycleLengthCpu(23UL, MontgomeryDivisorDataPool.Shared.FromModulus(23UL));
			session.CheckDivisor(23UL, MontgomeryDivisorDataPool.Shared.FromModulus(23UL), cycle23, primes, hits);
			hits.Should().ContainInOrder([0, 0, 1, 0]);

			Array.Fill(hits, (byte)0);
			ulong cycle31 = MersenneDivisorCycles.CalculateCycleLengthCpu(31UL, MontgomeryDivisorDataPool.Shared.FromModulus(31UL));
			session.CheckDivisor(31UL, MontgomeryDivisorDataPool.Shared.FromModulus(31UL), cycle31, primes, hits);
			hits.Should().ContainInOrder([1, 0, 0, 0]);
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
			byte[] hits = new byte[exponents.Length];

			MontgomeryDivisorData divisorData = MontgomeryDivisorDataPool.Shared.FromModulus(7UL);
			ulong cycle = MersenneDivisorCycles.CalculateCycleLengthCpu(7UL, divisorData);
			session.CheckDivisor(7UL, divisorData, cycle, exponents, hits);

			hits.Should().Equal([1, 0, 1, 0, 1]);
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
			byte[] hits = new byte[exponents.Length];

			MontgomeryDivisorData divisorData = MontgomeryDivisorDataPool.Shared.FromModulus(11UL);
			ulong cycle = MersenneDivisorCycles.CalculateCycleLengthCpu(11UL, divisorData);
			session.CheckDivisor(11UL, divisorData, cycle, exponents, hits);

			hits.Should().Equal([1, 0, 1, 0, 1]);
		}
		finally
		{
			session.Return();
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}
}

