using FluentAssertions;
using PerfectNumbers.Core.Gpu.Accelerators;
using Xunit;

namespace PerfectNumbers.Core.Tests;

[Trait("Category", "Fast")]
public class ModResidueTrackerTests
{
	[Fact]
	public void IsDivisible_advances_residue_for_identity()
	{
		var tracker = new ModResidueTracker(ResidueModel.Identity);

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			tracker.IsDivisible(gpu, 5, 2).Should().BeFalse();
			tracker.IsDivisible(gpu, 6, 2).Should().BeTrue();
			tracker.IsDivisible(gpu, 7, 2).Should().BeFalse();
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}

	[Theory]
	[InlineData(3UL, 7UL, true)]
	[InlineData(5UL, 31UL, true)]
	[InlineData(5UL, 7UL, false)]
	[InlineData(11UL, 23UL, true)]
	[InlineData(11UL, 31UL, false)]
	public void IsDivisible_handles_mersenne_model(ulong p, ulong divisor, bool expected)
	{
		var tracker = new ModResidueTracker(ResidueModel.Mersenne);
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);

		try
		{
			tracker.IsDivisible(gpu, p, divisor).Should().Be(expected);
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}

	[Fact]
	public void MergeOrAppend_tracks_existing_and_new_divisors()
	{
		var tracker = new ModResidueTracker(ResidueModel.Identity);
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);

		try
		{
			tracker.BeginMerge(10);
			tracker.MergeOrAppend(gpu, 10, 3, out bool d3).Should().BeTrue();
			d3.Should().BeFalse();
			tracker.MergeOrAppend(gpu, 10, 5, out bool d5).Should().BeTrue();
			d5.Should().BeTrue();

			tracker.BeginMerge(10);
			tracker.MergeOrAppend(gpu, 10, 4, out bool d4).Should().BeTrue();
			d4.Should().BeFalse();

			tracker.BeginMerge(11);
			tracker.MergeOrAppend(gpu, 11, 3, out d3).Should().BeTrue();
			d3.Should().BeFalse();
			tracker.MergeOrAppend(gpu, 11, 4, out d4).Should().BeTrue();
			d4.Should().BeFalse();
			tracker.MergeOrAppend(gpu, 11, 5, out d5).Should().BeTrue();
			d5.Should().BeFalse();
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}

	[Fact]
	public void IsDivisible_requires_non_decreasing_numbers()
	{
		var tracker = new ModResidueTracker(ResidueModel.Identity);
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			tracker.IsDivisible(gpu, 5, 2);

			Action act = () => tracker.IsDivisible(gpu, 4, 2);
			act.Should().Throw<InvalidOperationException>();
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}
}

