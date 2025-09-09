using FluentAssertions;
using Xunit;

namespace PerfectNumbers.Core.Tests;

[Trait("Category", "Fast")]
public class EulerPrimeTesterTests
{
    private readonly EulerPrimeTester _tester = new(new PrimeTester());

    [Theory]
    [InlineData(13UL, true)]
    [InlineData(29UL, true)]
    [InlineData(3UL, false)]
    [InlineData(21UL, false)]
    [InlineData(1UL, false)]
    public void IsEulerPrime_returns_expected(ulong n, bool expected)
    {
        _tester.IsEulerPrime(n).Should().Be(expected);
    }
}

