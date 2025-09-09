using FluentAssertions;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class MersenneResidueStepperTests
{
    [Fact]
    [Trait("Category", "Fast")]
    public void Step_advances_residue()
    {
        var stepper = new MersenneResidueStepper(13, 4, 2);

        stepper.State().Should().Be((4UL, 2UL));

        stepper.Step(3).Should().Be((7UL, 10UL));
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void To_moves_directly_to_target_exponent()
    {
        var stepper = new MersenneResidueStepper(13, 4, 2);

        stepper.To(10).Should().Be((10UL, 9UL));
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void Constructor_rejects_zero_modulus()
    {
        var act = () => new MersenneResidueStepper(0, 0, 0);

        act.Should().Throw<ArgumentException>();
    }
}

