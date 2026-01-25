using FluentAssertions;
using PerfectNumbers.Core.ByDivisor;
using Xunit;
using static PerfectNumbers.Core.ByDivisor.BitContradictionSolverWithAMultiplier;

namespace PerfectNumbers.Core.Tests;

public class BitContradictionSolverTests
{
    [Fact]
    public void ComputeColumnBounds_TracksForcedAndPossibleOnes()
    {
        bool[] a = [true, false];
        bool[] q = [true, true];

        ColumnBounds bounds = ComputeColumnBounds(a, q, 1);

        bounds.ForcedOnes.Should().Be(1);
        bounds.PossibleOnes.Should().Be(2);
    }

    [Fact]
    public void TryPropagateCarry_FailsWhenParityUnreachable()
    {
		CarryRange carry = CarryRange.Single(0);
        bool success = TryPropagateCarry(
            currentCarry: ref carry,
            forcedOnes: 0,
            possibleOnes: 0,
            requiredResultBit: 1);

        success.Should().BeFalse();
        carry.Min.Should().Be(0);
        carry.Max.Should().Be(0);
    }

    [Fact]
    public void TryPropagateCarry_ComputesNextCarryRange()
    {
		CarryRange carry = new(0, 1);
        bool success = TryPropagateCarry(
            currentCarry: ref carry,
            forcedOnes: 1,
            possibleOnes: 3,
            requiredResultBit: 1);

        success.Should().BeTrue();
        carry.Min.Should().Be(0);
        carry.Max.Should().Be(1);
    }
}
