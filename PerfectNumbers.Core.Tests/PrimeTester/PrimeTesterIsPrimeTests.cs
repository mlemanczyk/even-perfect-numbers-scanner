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

        tester.IsPrimeGpu(137UL, CancellationToken.None).Should().BeTrue();

        tester.IsPrimeGpu(341UL, CancellationToken.None).Should().BeFalse();

        GpuContextPool.ForceCpu = false;
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void IsPrimeGpu_handles_parallel_calls_for_known_primes()
    {
        GpuContextPool.ForceCpu = false;
        GpuPrimeWorkLimiter.SetLimit(64);
        PrimeTester.GpuBatchSize = 256_000;

        ulong[] primes = [107UL, 127UL, 521UL, 607UL];
        var results = new bool[primes.Length];

        var testers = new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true);
        Parallel.For(0, primes.Length, i =>
        {
            results[i] = testers.Value!.IsPrimeGpu(primes[i], CancellationToken.None);
        });

        results.Should().AllBeEquivalentTo(true);

        Parallel.For(0, Environment.ProcessorCount * 16, _ =>
        {
            foreach (ulong prime in primes)
            {
                testers.Value!.IsPrimeGpu(prime, CancellationToken.None).Should().BeTrue();
            }
        });
    }
}

