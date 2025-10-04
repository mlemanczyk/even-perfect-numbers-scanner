using System;
using FluentAssertions;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;
using Xunit;

namespace PerfectNumbers.Core.Tests;

[Trait("Category", "Fast")]
public class PrimeOrderGpuHeuristicsTests
{
    [Fact]
    public void TryPow2Mod_returns_overflow_for_marked_prime()
    {
        const ulong prime = 97UL;
        PrimeOrderGpuHeuristics.ClearAllOverflowForTesting();
        try
        {
            PrimeOrderGpuHeuristics.MarkOverflow(prime);

            GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2Mod(1UL, prime, out ulong remainder);

            status.Should().Be(GpuPow2ModStatus.Overflow);
            remainder.Should().Be(0UL);
        }
        finally
        {
            PrimeOrderGpuHeuristics.ClearOverflow(prime);
        }
    }

    [Fact]
    public void TryPow2Mod_returns_unavailable_for_unmarked_prime()
    {
        PrimeOrderGpuHeuristics.ClearAllOverflowForTesting();

        GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2Mod(1UL, 193UL, out ulong remainder);

        status.Should().Be(GpuPow2ModStatus.Unavailable);
        remainder.Should().Be(0UL);
    }

    [Fact]
    public void TryPow2ModBatch_returns_overflow_for_marked_prime()
    {
        const ulong prime = 193UL;
        PrimeOrderGpuHeuristics.ClearAllOverflowForTesting();
        try
        {
            PrimeOrderGpuHeuristics.MarkOverflow(prime);

            ulong[] remainders = new ulong[2];
            ulong[] exponents = { 1UL, 2UL };
            GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2ModBatch(exponents, prime, remainders);

            status.Should().Be(GpuPow2ModStatus.Overflow);
            remainders[0].Should().Be(0UL);
            remainders[1].Should().Be(0UL);
        }
        finally
        {
            PrimeOrderGpuHeuristics.ClearOverflow(prime);
        }
    }

    [Fact]
    public void TryPow2ModBatch_returns_unavailable_and_clears_remainders_for_unmarked_prime()
    {
        PrimeOrderGpuHeuristics.ClearAllOverflowForTesting();

        ulong[] remainders = { 123UL, 456UL };
        ulong[] exponents = { 1UL, 2UL };
        GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2ModBatch(exponents, 257UL, remainders);

        status.Should().Be(GpuPow2ModStatus.Unavailable);
        remainders[0].Should().Be(0UL);
        remainders[1].Should().Be(0UL);
    }

    [Fact]
    public void TryPow2ModBatch_throws_when_remainder_span_is_too_small()
    {
        ulong[] remainders = new ulong[1];
        ulong[] exponents = { 1UL, 2UL };

        Action act = () => PrimeOrderGpuHeuristics.TryPow2ModBatch(exponents, 5UL, remainders);

        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Calculate_falls_back_to_cpu_when_gpu_overflow_is_marked()
    {
        const ulong prime = 7UL;
        PrimeOrderGpuHeuristics.ClearAllOverflowForTesting();
        try
        {
            PrimeOrderGpuHeuristics.MarkOverflow(prime);

            PrimeOrderCalculator.PrimeOrderResult result = PrimeOrderCalculator.Calculate(prime, null, PrimeOrderCalculator.PrimeOrderSearchConfig.HeuristicDefault);

            result.Status.Should().Be(PrimeOrderCalculator.PrimeOrderStatus.Found);
            result.Order.Should().Be(3UL);
        }
        finally
        {
            PrimeOrderGpuHeuristics.ClearOverflow(prime);
        }
    }
}
