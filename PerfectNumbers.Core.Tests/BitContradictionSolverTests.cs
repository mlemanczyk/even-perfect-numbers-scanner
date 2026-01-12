using FluentAssertions;
using PerfectNumbers.Core.ByDivisor;
using Xunit;
using static PerfectNumbers.Core.ByDivisor.BitContradictionSolver;

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
        bounds.PossibleOnes.Should().Be(1);
    }

    [Fact]
    public void TryPropagateCarry_FailsWhenParityUnreachable()
    {
        bool success = TryPropagateCarry(
            currentCarry: CarryRange.Single(0),
            forcedOnes: 0,
            possibleOnes: 0,
            requiredResultBit: 1,
            out var nextCarry,
            out ContradictionReason reason);

        success.Should().BeFalse();
        reason.Should().Be(ContradictionReason.ParityUnreachable);
        nextCarry.Min.Should().Be(0);
        nextCarry.Max.Should().Be(0);
    }

    [Fact]
    public void TryPropagateCarry_ComputesNextCarryRange()
    {
        bool success = TryPropagateCarry(
            currentCarry: new CarryRange(0, 1),
            forcedOnes: 1,
            possibleOnes: 3,
            requiredResultBit: 1,
            out var nextCarry,
            out ContradictionReason reason);

        success.Should().BeTrue();
        reason.Should().Be(ContradictionReason.None);
        nextCarry.Min.Should().Be(0);
        nextCarry.Max.Should().Be(1);
    }
}
