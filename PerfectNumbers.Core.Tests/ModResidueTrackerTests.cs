using FluentAssertions;
using PerfectNumbers.Core.Gpu.Accelerators;
using Xunit;

namespace PerfectNumbers.Core.Tests;

[Trait("Category", "Fast")]
public class ModResidueTrackerTests
{
	[Fact]
	public void IsDivisibleCpu_advances_residue_for_identity()
	{
		var tracker = new ModResidueTracker(ResidueModel.Identity);

		tracker.IsDivisibleCpu(5, 2).Should().BeFalse();
		tracker.IsDivisibleCpu(6, 2).Should().BeTrue();
		tracker.IsDivisibleCpu(7, 2).Should().BeFalse();
	}

	[Fact]
	public void IsDivisibleGpu_advances_residue_for_identity()
	{
		var tracker = new ModResidueTracker(ResidueModel.Identity);

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			tracker.IsDivisibleGpu(gpu, 5, 2).Should().BeFalse();
			tracker.IsDivisibleGpu(gpu, 6, 2).Should().BeTrue();
			tracker.IsDivisibleGpu(gpu, 7, 2).Should().BeFalse();
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
	public void IsDivisibleCpu_handles_mersenne_model(ulong p, ulong divisor, bool expected)
	{
		var tracker = new ModResidueTracker(ResidueModel.Mersenne);
		tracker.IsDivisibleCpu(p, divisor).Should().Be(expected);
	}

	[Theory]
	[InlineData(3UL, 7UL, true)]
	[InlineData(5UL, 31UL, true)]
	[InlineData(5UL, 7UL, false)]
	[InlineData(11UL, 23UL, true)]
	[InlineData(11UL, 31UL, false)]
	public void IsDivisibleGpu_handles_mersenne_model(ulong p, ulong divisor, bool expected)
	{
		var tracker = new ModResidueTracker(ResidueModel.Mersenne);
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);

		try
		{
			tracker.IsDivisibleGpu(gpu, p, divisor).Should().Be(expected);
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}

	[Fact]
	public void MergeOrAppendCpu_tracks_existing_and_new_divisors()
	{
		var tracker = new ModResidueTracker(ResidueModel.Identity);

		tracker.BeginMerge(10);
		tracker.MergeOrAppendCpu(10, 3, out bool d3).Should().BeTrue();
		d3.Should().BeFalse();
		tracker.MergeOrAppendCpu(10, 5, out bool d5).Should().BeTrue();
		d5.Should().BeTrue();

		tracker.BeginMerge(10);
		tracker.MergeOrAppendCpu(10, 4, out bool d4).Should().BeTrue();
		d4.Should().BeFalse();

		tracker.BeginMerge(11);
		tracker.MergeOrAppendCpu(11, 3, out d3).Should().BeTrue();
		d3.Should().BeFalse();
		tracker.MergeOrAppendCpu(11, 4, out d4).Should().BeTrue();
		d4.Should().BeFalse();
		tracker.MergeOrAppendCpu(11, 5, out d5).Should().BeTrue();
		d5.Should().BeFalse();
	}

	[Fact]
	public void MergeOrAppendGpu_tracks_existing_and_new_divisors()
	{
		var tracker = new ModResidueTracker(ResidueModel.Identity);
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);

		try
		{
			tracker.BeginMerge(10);
			tracker.MergeOrAppendGpu(gpu, 10, 3, out bool d3).Should().BeTrue();
			d3.Should().BeFalse();
			tracker.MergeOrAppendGpu(gpu, 10, 5, out bool d5).Should().BeTrue();
			d5.Should().BeTrue();

			tracker.BeginMerge(10);
			tracker.MergeOrAppendGpu(gpu, 10, 4, out bool d4).Should().BeTrue();
			d4.Should().BeFalse();

			tracker.BeginMerge(11);
			tracker.MergeOrAppendGpu(gpu, 11, 3, out d3).Should().BeTrue();
			d3.Should().BeFalse();
			tracker.MergeOrAppendGpu(gpu, 11, 4, out d4).Should().BeTrue();
			d4.Should().BeFalse();
			tracker.MergeOrAppendGpu(gpu, 11, 5, out d5).Should().BeTrue();
			d5.Should().BeFalse();
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}

	[Fact]
	public void IsDivisibleCpu_requires_non_decreasing_numbers()
	{
		var tracker = new ModResidueTracker(ResidueModel.Identity);

		tracker.IsDivisibleCpu(5, 2);

		Action act = () => tracker.IsDivisibleCpu(4, 2);
		act.Should().Throw<InvalidOperationException>();
	}

	[Fact]
	public void IsDivisibleGpu_requires_non_decreasing_numbers()
	{
		var tracker = new ModResidueTracker(ResidueModel.Identity);
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			tracker.IsDivisibleGpu(gpu, 5, 2);

			Action act = () => tracker.IsDivisibleGpu(gpu, 4, 2);
			act.Should().Throw<InvalidOperationException>();
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}
}

