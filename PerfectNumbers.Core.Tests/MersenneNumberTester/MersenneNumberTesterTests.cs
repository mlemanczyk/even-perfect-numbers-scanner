using FluentAssertions;
using PerfectNumbers.Core.Gpu.Accelerators;
using Xunit;

namespace PerfectNumbers.Core.Tests;

[Collection("GpuNtt")]
[Trait("Category", "Fast")]
public class MersenneNumberTesterTests
{
	[Theory]
	[InlineData(136_000_002UL)]
	[InlineData(136_000_005UL)]
	public void IsMersennePrime_detects_small_prime_divisors_for_large_exponents(ulong p)
	{
		var tester = new MersenneNumberTester();
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			tester.IsMersennePrime(gpu, p).Should().BeFalse();
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}

	[Theory]
	[InlineData(125UL)]
	public void WarmUpOrders_populates_cache_without_affecting_results(ulong p)
	{
		var tester = new MersenneNumberTester(useOrderCache: true, useGpuOrder: true);
		tester.WarmUpOrders(p, 1_000UL);
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			tester.IsMersennePrime(gpu, p).Should().BeFalse();
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}

	[Theory]
	[InlineData(125UL, true)]
	[InlineData(127UL, false)]
	public void SharesFactorWithExponentMinusOne_detects_common_factors(ulong p, bool expected)
	{
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			p.SharesFactorWithExponentMinusOne(gpu).Should().Be(expected);
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}
}

