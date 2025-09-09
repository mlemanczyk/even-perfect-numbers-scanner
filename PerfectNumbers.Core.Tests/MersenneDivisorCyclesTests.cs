using FluentAssertions;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class MersenneDivisorCyclesTests
{
    public static IEnumerable<object[]> Divisors() =>
        new[]
        {
            new object[] {3UL, 2UL},
            new object[] {7UL, 3UL},
            new object[] {11UL, 10UL},
        };

    [Theory]
    [Trait("Category", "Fast")]
    [MemberData(nameof(Divisors))]
    public void CalculateCycleLength_returns_expected_value(ulong divisor, ulong expected)
    {
        MersenneDivisorCycles.CalculateCycleLength(divisor).Should().Be(expected);
        MersenneDivisorCycles.CalculateCycleLengthGpu(divisor).Should().Be(expected);
    }
}

