using FluentAssertions;
using PerfectNumbers.Core.Gpu.Accelerators;
using Xunit;

namespace PerfectNumbers.Core.Tests;

[Trait("Category", "Fast")]
public class MersenneResidueModeTests
{
	[Fact]
	public void IsMersennePrime_residue_mode_rejects_composite_exponent()
	{
		var tester = new MersenneNumberTesterForResidueCalculationMethodForCpu(

			useGpuScan: false,
			useGpuOrder: false);
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			tester.IsMersennePrime(gpu, 136_000_002UL).Should().BeFalse();
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}

	[Fact]
	public void IsMersennePrime_residue_mode_accepts_prime_exponent()
	{
		var tester = new MersenneNumberTesterForResidueCalculationMethodForCpu(

			useGpuScan: false,
			useGpuOrder: false,
			maxK: 1_000UL);
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			tester.IsMersennePrime(gpu, 31UL).Should().BeTrue();
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}
}



