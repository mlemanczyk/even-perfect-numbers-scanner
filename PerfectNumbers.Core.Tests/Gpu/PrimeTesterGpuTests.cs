using FluentAssertions;
using PerfectNumbers.Core.Gpu.Accelerators;
using Xunit;

namespace PerfectNumbers.Core.Tests.Gpu;

[Trait("Category", "Fast")]
public class PrimeTesterGpuTests
{
	[Fact]
	public void IsPrimeGpu_accepts_known_primes()
	{
		ulong[] primes = [31UL, 61UL, 89UL, 107UL, 127UL, 521UL];
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			foreach (ulong prime in primes)
			{
				PrimeTester.IsPrimeGpu(gpu, prime).Should().BeTrue();
			}
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}

	[Fact]
	public void IsPrimeGpu_rejects_composites()
	{
		ulong[] composites = [33UL, 39UL, 51UL, 77UL, 91UL];
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			foreach (ulong composite in composites)
			{
				PrimeTester.IsPrimeGpu(gpu, composite).Should().BeFalse();
			}
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}
}
