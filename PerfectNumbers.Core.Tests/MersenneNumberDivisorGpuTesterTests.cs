using System.Numerics;
using System.Reflection;
using FluentAssertions;
using PerfectNumbers.Core.Cpu;
using PerfectNumbers.Core.Gpu;
using PerfectNumbers.Core.Gpu.Accelerators;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class MersenneNumberDivisorGpuTesterTests
{
	[Theory]
	[InlineData(33UL, 7UL, true)]
	[InlineData(35UL, 31UL, true)]
	[InlineData(37UL, 223UL, true)]
	[InlineData(33UL, 13UL, false)]
	[InlineData(35UL, 73UL, false)]
	[InlineData(37UL, 227UL, false)]
	[Trait("Category", "Fast")]
	public void IsDivisible_returns_expected(ulong exponent, ulong divisor, bool expected)
	{
		var tester = new MersenneNumberDivisorGpuTester();
		ReadOnlyGpuUInt128 divisorValue = new ReadOnlyGpuUInt128(divisor);
		tester.IsDivisible(exponent, in divisorValue).Should().Be(expected);
	}

	[Fact]
	[Trait("Category", "Fast")]
	public void IsDivisible_handles_large_divisors()
	{
		var tester = new MersenneNumberDivisorGpuTester();
		UInt128 divisor = (UInt128.One << 65) - UInt128.One;
		ReadOnlyGpuUInt128 divisorValue = new ReadOnlyGpuUInt128(divisor);
		tester.IsDivisible(65UL, in divisorValue).Should().BeTrue();
		divisorValue = new ReadOnlyGpuUInt128(divisor + 2); // Reusing divisorValue for the shifted candidate.
		tester.IsDivisible(65UL, in divisorValue).Should().BeFalse();
	}

	[Fact]
	[Trait("Category", "Fast")]
	public void IsPrime_sets_divisorsExhausted_false_when_search_range_not_exhausted()
	{
		var tester = new MersenneNumberDivisorGpuTester();
		typeof(MersenneNumberDivisorGpuTester)
			.GetField("_divisorCandidates", BindingFlags.NonPublic | BindingFlags.Static)!
			.SetValue(null, Array.Empty<(ulong, uint)>());

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			tester.IsPrime(gpu, 31UL, UInt128.Zero, 0UL, out bool divisorsExhausted).Should().BeTrue();
			divisorsExhausted.Should().BeFalse();
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}

	[Fact]
	[Trait("Category", "Fast")]
	public void IsPrime_sets_divisorsExhausted_true_when_divisible_by_specified_divisor()
	{
		var tester = new MersenneNumberDivisorGpuTester();
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			tester.IsPrime(gpu, 35UL, 31UL, 0UL, out bool divisorsExhausted).Should().BeFalse();
			divisorsExhausted.Should().BeTrue();
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}

	[Fact]
	[Trait("Category", "Fast")]
	public void IsPrime_accepts_large_search_limits()
	{
		var tester = new MersenneNumberDivisorGpuTester();
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			tester.IsPrime(gpu, 35UL, 31UL, ulong.MaxValue, out bool exhausted).Should().BeFalse();
			exhausted.Should().BeTrue();
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}

	private static void ConfigureByDivisorTester(MersenneNumberDivisorByDivisorGpuTester tester, ulong maxPrime)
	{
		tester.PrepareCandidates(maxPrime, ReadOnlySpan<ulong>.Empty, Span<ulong>.Empty);
		tester.ResumeFromState(Path.Combine(Path.GetTempPath(), "bydivisor-gpu-tests.bin"), BigInteger.Zero, BigInteger.One);
		tester.ResetStateTracking();
	}

	[Fact]
	[Trait("Category", "Fast")]
	public void ByDivisor_tester_tracks_divisors_across_primes()
	{
		var tester = new MersenneNumberDivisorByDivisorGpuTester();
		ConfigureByDivisorTester(tester, maxPrime: 43UL);

		tester.IsPrime(31UL, out bool exhausted, out BigInteger divisor).Should().BeTrue();
		exhausted.Should().BeTrue();
		divisor.Should().Be(BigInteger.Zero);

		tester.IsPrime(37UL, out exhausted, out divisor).Should().BeFalse();
		exhausted.Should().BeTrue();
		divisor.Should().BeGreaterThan(BigInteger.Zero);

		tester.IsPrime(43UL, out exhausted, out divisor).Should().BeFalse();
		exhausted.Should().BeTrue();
		divisor.Should().BeGreaterThan(BigInteger.Zero);
	}

	[Fact]
	[Trait("Category", "Fast")]
	public void ByDivisor_tester_defaults_to_zero_limit_before_prepare_candidates()
	{
		var tester = new MersenneNumberDivisorByDivisorGpuTester();
		tester.DivisorLimit.Should().Be(0UL);

		ulong[] primes = [31UL, 37UL];
		ulong[] allowed = new ulong[primes.Length];
		tester.PrepareCandidates(maxPrime: 37UL, primes, allowed);

		tester.DivisorLimit.Should().BeGreaterThan(0UL);
		allowed.Should().OnlyContain(value => value > 0UL);
	}

	[Fact]
	[Trait("Category", "Fast")]
	public void ByDivisor_tester_respects_filter_based_limit()
	{
		var tester = new MersenneNumberDivisorByDivisorGpuTester();
		ConfigureByDivisorTester(tester, maxPrime: 41UL);

		tester.IsPrime(31UL, out _, out BigInteger divisor).Should().BeTrue();
		divisor.Should().Be(BigInteger.Zero);

		tester.IsPrime(37UL, out bool exhausted, out divisor).Should().BeFalse();
		exhausted.Should().BeTrue();
		divisor.Should().BeGreaterThan(BigInteger.Zero);

		tester.IsPrime(41UL, out bool divisorsExhausted, out divisor).Should().BeFalse();
		divisorsExhausted.Should().BeTrue();
		divisor.Should().BeGreaterThan(BigInteger.Zero);
	}

	[Theory]
	[InlineData(71UL, false)]
	[InlineData(89UL, true)]
	[InlineData(127UL, true)]
	[Trait("Category", "Fast")]
	public void ByDivisor_tester_matches_incremental_expectations(ulong exponent, bool expectedPrime)
	{
		var tester = new MersenneNumberDivisorByDivisorGpuTester();
		ConfigureByDivisorTester(tester, maxPrime: exponent);

		tester.IsPrime(exponent, out bool exhausted, out BigInteger divisor).Should().Be(expectedPrime);

		exhausted.Should().BeTrue();
		if (expectedPrime)
		{
			divisor.Should().Be(BigInteger.Zero);
		}
		else
		{
			divisor.Should().BeGreaterThan(BigInteger.Zero);
		}
	}

	[Fact]
	[Trait("Category", "Fast")]
	public void ByDivisor_gpu_tester_uses_cycle_remainders_per_divisor()
	{
		int originalBatchSize = EnvironmentConfiguration.GpuBatchSize;
		try
		{
			EnvironmentConfiguration.GpuBatchSize = 8;
			var tester = new MersenneNumberDivisorByDivisorGpuTester();
			ConfigureByDivisorTester(tester, maxPrime: 43UL);

			tester.IsPrime(41UL, out bool divisorsExhausted, out BigInteger divisor).Should().BeFalse();
			divisorsExhausted.Should().BeTrue();
			divisor.Should().BeGreaterThan(BigInteger.Zero);
		}
		finally
		{
			EnvironmentConfiguration.GpuBatchSize = originalBatchSize;
		}
	}

	[Fact]
	[Trait("Category", "Fast")]
	public void ByDivisor_session_computes_cycle_when_zero_is_provided()
	{
		var tester = new MersenneNumberDivisorByDivisorGpuTester();
		ConfigureByDivisorTester(tester, maxPrime: 43UL);

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			var session = tester.CreateDivisorSession(gpu);

			ulong[] primes = [31UL, 37UL, 41UL];
			session.CheckDivisor(223UL, MontgomeryDivisorData.FromModulus(223UL), 0UL, primes).Should().BeTrue();

			ulong[] primesWithoutHit = [31UL, 41UL];
			session.CheckDivisor(223UL, MontgomeryDivisorData.FromModulus(223UL), 0UL, primesWithoutHit).Should().BeFalse();
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
		var tester = new MersenneNumberDivisorByDivisorGpuTester();
		ConfigureByDivisorTester(tester, maxPrime: 43UL);

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			var session = tester.CreateDivisorSession(gpu);

			ulong[] primes = [31UL, 37UL, 41UL, 43UL];

			ulong cycle223 = MersenneDivisorCycles.CalculateCycleLengthGpu(gpu, 223UL, MontgomeryDivisorData.FromModulus(223UL));
			session.CheckDivisor(223UL, MontgomeryDivisorData.FromModulus(223UL), cycle223, primes).Should().BeTrue();

			ulong[] primesBeforeHit = [31UL];
			session.CheckDivisor(223UL, MontgomeryDivisorData.FromModulus(223UL), cycle223, primesBeforeHit).Should().BeFalse();

			ulong cycle13367 = MersenneDivisorCycles.CalculateCycleLengthGpu(gpu, 13367UL, MontgomeryDivisorData.FromModulus(13367UL));
			session.CheckDivisor(13367UL, MontgomeryDivisorData.FromModulus(13367UL), cycle13367, primes).Should().BeTrue();

			ulong[] primesWithoutMatchingExponent = [31UL, 37UL, 43UL];
			session.CheckDivisor(13367UL, MontgomeryDivisorData.FromModulus(13367UL), cycle13367, primesWithoutMatchingExponent).Should().BeFalse();
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}

	[Fact]
	[Trait("Category", "Fast")]
	public void ByDivisor_gpu_session_reuses_cycle_remainders_across_batches()
	{
		int originalBatchSize = EnvironmentConfiguration.GpuBatchSize;
		try
		{
			EnvironmentConfiguration.GpuBatchSize = 2;
			var tester = new MersenneNumberDivisorByDivisorGpuTester();
			ConfigureByDivisorTester(tester, maxPrime: 47UL);

			var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
			try
			{
				var session = tester.CreateDivisorSession(gpu);

				ulong[] primes = [31UL, 37UL, 41UL, 43UL, 47UL];

				ulong cycle223 = MersenneDivisorCycles.CalculateCycleLengthGpu(gpu, 223UL, MontgomeryDivisorData.FromModulus(223UL));
				session.CheckDivisor(223UL, MontgomeryDivisorData.FromModulus(223UL), cycle223, primes).Should().BeTrue();

				ulong[] primesBeforeHit = [31UL];
				session.CheckDivisor(223UL, MontgomeryDivisorData.FromModulus(223UL), cycle223, primesBeforeHit).Should().BeFalse();
			}
			finally
			{
				PrimeOrderCalculatorAccelerator.Return(gpu);
			}
		}
		finally
		{
			EnvironmentConfiguration.GpuBatchSize = originalBatchSize;
		}
	}

	[Fact]
	[Trait("Category", "Fast")]
	public void ByDivisor_session_marks_mersenne_numbers_divisible_by_seven_as_composite()
	{
		var tester = new MersenneNumberDivisorByDivisorGpuTester();
		ConfigureByDivisorTester(tester, maxPrime: 43UL);

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			var session = tester.CreateDivisorSession(gpu);
			ulong[] exponents = [6UL, 7UL, 9UL, 10UL, 12UL];

			MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(7UL);
			ulong cycle = MersenneDivisorCycles.CalculateCycleLengthGpu(gpu, 7UL, divisorData);

			session.CheckDivisor(7UL, divisorData, cycle, exponents).Should().BeTrue();

			ulong[] nonDivisibleExponents = [5UL, 11UL];
			session.CheckDivisor(7UL, divisorData, cycle, nonDivisibleExponents).Should().BeFalse();
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}

	[Fact]
	[Trait("Category", "Fast")]
	public void ByDivisor_session_marks_mersenne_numbers_divisible_by_eleven_as_composite()
	{
		var tester = new MersenneNumberDivisorByDivisorGpuTester();
		ConfigureByDivisorTester(tester, maxPrime: 61UL);

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			var session = tester.CreateDivisorSession(gpu);
			ulong[] exponents = [10UL, 11UL, 20UL, 21UL, 30UL];

			MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(11UL);
			ulong cycle = MersenneDivisorCycles.CalculateCycleLengthGpu(gpu, 11UL, divisorData);

			session.CheckDivisor(11UL, divisorData, cycle, exponents).Should().BeTrue();

			ulong[] nonDivisibleExponents = [5UL, 17UL];
			session.CheckDivisor(11UL, divisorData, cycle, nonDivisibleExponents).Should().BeFalse();
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}

	[Fact]
	[Trait("Category", "Fast")]
	public void ByDivisor_gpu_tester_skips_divisors_excluded_by_small_cycle_generation()
	{
		var cycles = MersenneDivisorCycles.Shared;
		var tableField = typeof(MersenneDivisorCycles).GetField("_table", BindingFlags.NonPublic | BindingFlags.Instance)!;
		var smallCyclesField = typeof(MersenneDivisorCycles).GetField("_smallCycles", BindingFlags.NonPublic | BindingFlags.Instance)!;

		var originalTable = (List<(ulong Divisor, ulong Cycle)>)tableField.GetValue(cycles)!;
		var originalSmall = (ulong[]?)smallCyclesField.GetValue(cycles);

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			var patchedTable = new List<(ulong Divisor, ulong Cycle)>();
			ulong[] patchedSmall = new ulong[PerfectNumberConstants.MaxQForDivisorCycles + 1];

			MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(191UL);
			patchedSmall[191] = MersenneDivisorCycles.CalculateCycleLengthGpu(gpu, 191UL, divisorData);

			tableField.SetValue(cycles, patchedTable);
			smallCyclesField.SetValue(cycles, patchedSmall);
			DivisorCycleCache.Shared.RefreshSnapshot();

			var tester = new MersenneNumberDivisorByDivisorGpuTester();
			ConfigureByDivisorTester(tester, maxPrime: 19UL);

			typeof(MersenneNumberDivisorByDivisorGpuTester)
				.GetField("_divisorLimit", BindingFlags.NonPublic | BindingFlags.Instance)!
				.SetValue(tester, 200UL);

			tester.IsPrime(19UL, out bool divisorsExhausted, out BigInteger divisor).Should().BeTrue();
			divisorsExhausted.Should().BeTrue();
			divisor.Should().Be(BigInteger.Zero);
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
			tableField.SetValue(cycles, originalTable);
			smallCyclesField.SetValue(cycles, originalSmall);
			DivisorCycleCache.Shared.RefreshSnapshot();
		}
	}

	[Fact]
	[Trait("Category", "Fast")]
	public void ByDivisor_cpu_session_checks_divisors_across_primes()
	{
		var session = new MersenneCpuDivisorScanSessionWithCpuOrder();

		ulong[] primes = [31UL, 37UL, 41UL, 43UL, 47UL];
		ulong cycle223 = MersenneDivisorCycles.CalculateCycleLengthCpu(223UL, MontgomeryDivisorData.FromModulus(223UL));
		session.CheckDivisor(223UL, MontgomeryDivisorData.FromModulus(223UL), cycle223, primes).Should().BeTrue();

		ulong[] primesBeforeHit = [31UL];
		session.CheckDivisor(223UL, MontgomeryDivisorData.FromModulus(223UL), cycle223, primesBeforeHit).Should().BeFalse();
	}
}
