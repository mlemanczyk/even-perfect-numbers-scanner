using FluentAssertions;
using PerfectNumbers.Core.Gpu;
using PerfectNumbers.Core.Gpu.Accelerators;
using Xunit;

namespace PerfectNumbers.Core.Tests.Gpu;

[Trait("Category", "Slow")]
public class LucasLehmerPrimeTesterTests
{
	[Theory]
	[InlineData(3UL, true)]
	[InlineData(5UL, true)]
	[InlineData(11UL, false)]
	[Trait("Category", "Fast")]
	public void IsMersennePrime_gpu_returns_expected_results(ulong p, bool expected)
	{
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			new MersenneNumberLucasLehmerGpuTester().IsPrime(gpu, p, true).Should().Be(expected);
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}

	[Fact]
	[Trait("Category", "Fast")]
	public void IsMersennePrime_gpu_large_exponent_without_cpu_fallback()
	{
		// Use a modest exponent that still exercises the NTT path while keeping
		// runtime short enough for the unit test environment.
		const ulong exponent = 128UL;
		var tester = new MersenneNumberLucasLehmerGpuTester();
		tester.WarmUpNttParameters(exponent);
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			tester.IsPrime(gpu, exponent, true).Should().BeFalse();
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}

	[Fact]
	[Trait("Category", "Fast")]
	public void IsMersennePrimeBatch_matches_individual()
	{
		ulong[] exponents = new ulong[] { 3UL, 5UL, 7UL, 11UL, 13UL, 17UL, 19UL, 23UL };
		bool[] expected = new bool[] { true, true, true, false, true, true, true, false };
		var tester = new MersenneNumberLucasLehmerGpuTester();
		bool[] results = new bool[exponents.Length];
		tester.IsMersennePrimeBatch(exponents, results);
		results.Should().Equal(expected);
	}

	[Theory]
	[InlineData(132UL)]
	[InlineData(1000UL)]
	[Trait("Category", "Slow")]
	public void WarmUpNttParameters_populates_cache_for_large_exponent(ulong exponent)
	{
		var tester = new MersenneNumberLucasLehmerGpuTester();
		tester.WarmUpNttParameters(exponent);
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			tester.IsPrime(gpu, exponent, true).Should().BeFalse();
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}

	[Theory]
	[InlineData(3UL, true)]
	[InlineData(5UL, true)]
	[InlineData(11UL, false)]
	[Trait("Category", "Fast")]
	public void IsMersennePrime_cpu_matches_expected_results(ulong p, bool expected)
	{
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			new MersenneNumberLucasLehmerCpuTester().IsPrime(gpu, p).Should().Be(expected);
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}
}

