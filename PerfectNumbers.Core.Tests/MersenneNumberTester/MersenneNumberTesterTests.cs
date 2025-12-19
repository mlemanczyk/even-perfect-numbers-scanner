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
		var tester = new MersenneNumberTesterCpu();
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
	public void WarmUpOrdersCpu_populates_cache_without_affecting_results(ulong p)
	{
		var tester = new MersenneNumberTesterCpu(useOrderCache: true, useGpuOrder: true);
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
	[InlineData(125UL)]
	public void WarmUpOrdersGpu_populates_cache_without_affecting_results(ulong p)
	{
		var tester = new MersenneNumberTesterGpu(useOrderCache: true, useGpuOrder: true);
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
	public void SharesFactorWithExponentMinusOneCpu_detects_common_factors(ulong p, bool expected)
	{
		p.SharesFactorWithExponentMinusOneCpu().Should().Be(expected);
	}

	[Theory]
	[InlineData(125UL, true)]
	[InlineData(127UL, false)]
	public void SharesFactorWithExponentMinusOneGpu_detects_common_factors(ulong p, bool expected)
	{
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			p.SharesFactorWithExponentMinusOneGpu(gpu).Should().Be(expected);
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}
}



