using FluentAssertions;
using PerfectNumbers.Core.Gpu;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class PrimeTesterIsPrimeTests
{
    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(1UL, false)]
    [InlineData(2UL, true)]
    [InlineData(3UL, true)]
    [InlineData(4UL, false)]
    [InlineData(5UL, true)]
    [InlineData(7UL, true)]
    [InlineData(97UL, true)]
    [InlineData(121UL, false)]
    public void IsPrime_returns_expected_results(ulong n, bool expected)
    {
        var tester = new PrimeTester();

        tester.IsPrime(n, CancellationToken.None).Should().Be(expected);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void IsPrimeGpu_falls_back_to_cpu_when_forced()
    {
        var tester = new PrimeTester();
        GpuContextPool.ForceCpu = true;

        tester.IsPrimeGpu(11UL, CancellationToken.None).Should().BeTrue();

        tester.IsPrimeGpu(12UL, CancellationToken.None).Should().BeFalse();

        GpuContextPool.ForceCpu = false;
    }
}

