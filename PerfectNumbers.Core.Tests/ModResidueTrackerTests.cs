using FluentAssertions;
using Xunit;

namespace PerfectNumbers.Core.Tests;

[Trait("Category", "Fast")]
public class ModResidueTrackerTests
{
    [Fact]
    public void IsDivisible_advances_residue_for_identity()
    {
        var tracker = new ModResidueTracker(ResidueModel.Identity);

        tracker.IsDivisible(5, 2).Should().BeFalse();
        tracker.IsDivisible(6, 2).Should().BeTrue();
        tracker.IsDivisible(7, 2).Should().BeFalse();
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

        tracker.IsDivisible(p, divisor).Should().Be(expected);
    }

    [Fact]
    public void MergeOrAppend_tracks_existing_and_new_divisors()
    {
        var tracker = new ModResidueTracker(ResidueModel.Identity);

        tracker.BeginMerge(10);
        tracker.MergeOrAppend(10, 3, out bool d3).Should().BeTrue();
        d3.Should().BeFalse();
        tracker.MergeOrAppend(10, 5, out bool d5).Should().BeTrue();
        d5.Should().BeTrue();

        tracker.BeginMerge(10);
        tracker.MergeOrAppend(10, 4, out bool d4).Should().BeTrue();
        d4.Should().BeFalse();

        tracker.BeginMerge(11);
        tracker.MergeOrAppend(11, 3, out d3).Should().BeTrue();
        d3.Should().BeFalse();
        tracker.MergeOrAppend(11, 4, out d4).Should().BeTrue();
        d4.Should().BeFalse();
        tracker.MergeOrAppend(11, 5, out d5).Should().BeTrue();
        d5.Should().BeFalse();
    }

    [Fact]
    public void IsDivisible_requires_non_decreasing_numbers()
    {
        var tracker = new ModResidueTracker(ResidueModel.Identity);
        tracker.IsDivisible(5, 2);

        Action act = () => tracker.IsDivisible(4, 2);
        act.Should().Throw<InvalidOperationException>();
    }
}

