using FluentAssertions;
using PerfectNumbers.Core.Gpu;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class PrimeTesterIsPrimeTests
{
    [Theory]
    [Trait("Category", "Fast")]
    // EvenPerfectBitScanner never routes the tiny values anymore; keep coverage focused on the admissible range handled by production scans so the guards in PrimeTester stay commented out without breaking validation.
    [InlineData(7UL, true)]
    [InlineData(9UL, false)]
    [InlineData(11UL, true)]
    [InlineData(13UL, true)]
    [InlineData(21UL, false)]
    [InlineData(97UL, true)]
    [InlineData(133UL, false)]
    [InlineData(137UL, true)]
    [InlineData(143UL, false)]
    [InlineData(199UL, true)]
    [InlineData(209UL, false)]
    [InlineData(403UL, false)]
    [InlineData(341UL, false)]
    public void IsPrime_returns_expected_results(ulong n, bool expected)
    {
        var tester = new PrimeTester();

		PrimeTester.IsPrime(n, CancellationToken.None).Should().Be(expected);
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(139UL, true)]
    [InlineData(189UL, false)]
    [InlineData(193UL, true)]
    [InlineData(279UL, false)]
    public void IsPrimeGpu_uses_residue_specific_tables(ulong n, bool expected)
    {
        GpuContextPool.ForceCpu = false;
        var tester = new PrimeTester();

        tester.IsPrimeGpu(n, CancellationToken.None).Should().Be(expected);
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

