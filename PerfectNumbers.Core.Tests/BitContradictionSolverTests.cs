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
        BitState[] a = [BitState.One, BitState.Unknown];
        BitState[] q = [BitState.One, BitState.One];

        ColumnBounds bounds = ComputeColumnBounds(a, q, 1);

        bounds.ForcedOnes.Should().Be(1);
        bounds.PossibleOnes.Should().Be(2);
    }

    [Fact]
    public void TryPropagateCarry_FailsWhenParityUnreachable()
    {
        CarryRange carry = CarryRange.Single(0);

        bool success = TryPropagateCarry(
            carry,
            forcedOnes: 0,
            possibleOnes: 0,
            requiredResultBit: 1,
            out CarryRange next,
            out ContradictionReason reason);

        success.Should().BeFalse();
        reason.Should().Be(ContradictionReason.ParityUnreachable);
        next.Min.Should().Be(0);
        next.Max.Should().Be(0);
    }

    [Fact]
    public void TryPropagateCarry_ComputesNextCarryRange()
    {
        CarryRange carry = new(0, 1);

        bool success = TryPropagateCarry(
            carry,
            forcedOnes: 1,
            possibleOnes: 3,
            requiredResultBit: 1,
            out CarryRange next,
            out ContradictionReason reason);

        success.Should().BeTrue();
        reason.Should().Be(ContradictionReason.None);
        next.Min.Should().Be(0);
        next.Max.Should().Be(1);
    }
}
