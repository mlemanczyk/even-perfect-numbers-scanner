using System;
using FluentAssertions;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;
using Xunit;

namespace PerfectNumbers.Core.Tests;

[Trait("Category", "Fast")]
public class PrimeOrderGpuHeuristicsTests
{
    public PrimeOrderGpuHeuristicsTests()
    {
        PrimeOrderGpuHeuristics.OverflowRegistry.Clear();
        PrimeOrderGpuHeuristics.ResetCapabilitiesForTesting();
    }

    [Fact]
    public void TryPow2Mod_returns_overflow_for_marked_prime()
    {
        const ulong prime = 97UL;
        try
        {
            PrimeOrderGpuHeuristics.OverflowRegistry[prime] = 0;

            GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2Mod(1UL, prime, out ulong remainder);

            status.Should().Be(GpuPow2ModStatus.Overflow);
            remainder.Should().Be(0UL);
        }
        finally
        {
            PrimeOrderGpuHeuristics.OverflowRegistry.TryRemove(prime, out _);
        }
    }

    [Fact]
    public void TryPow2Mod_returns_success_for_supported_prime()
    {
        GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2Mod(1UL, 193UL, out ulong remainder);

        status.Should().Be(GpuPow2ModStatus.Success);
        remainder.Should().Be(2UL);
    }

    [Fact]
    public void TryPow2Mod_returns_overflow_when_modulus_exceeds_capability()
    {
        const ulong prime = 97UL;
        var capability = new PrimeOrderGpuHeuristics.PrimeOrderGpuCapability(6, 64);
        PrimeOrderGpuHeuristics.OverrideCapabilitiesForTesting(capability);
        try
        {
            GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2Mod(1UL, prime, out ulong remainder);

            status.Should().Be(GpuPow2ModStatus.Overflow);
            remainder.Should().Be(0UL);
        }
        finally
        {
            PrimeOrderGpuHeuristics.ResetCapabilitiesForTesting();
            PrimeOrderGpuHeuristics.OverflowRegistry.TryRemove(prime, out _);
        }
    }

    [Fact]
    public void TryPow2Mod_returns_overflow_when_exponent_exceeds_capability()
    {
        var capability = new PrimeOrderGpuHeuristics.PrimeOrderGpuCapability(64, 3);
        PrimeOrderGpuHeuristics.OverrideCapabilitiesForTesting(capability);
        try
        {
            GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2Mod(16UL, 193UL, out ulong remainder);

            status.Should().Be(GpuPow2ModStatus.Overflow);
            remainder.Should().Be(0UL);

            GpuPow2ModStatus followUp = PrimeOrderGpuHeuristics.TryPow2Mod(4UL, 193UL, out ulong followUpRemainder);
            followUp.Should().Be(GpuPow2ModStatus.Success);
            followUpRemainder.Should().Be(16UL);
        }
        finally
        {
            PrimeOrderGpuHeuristics.ResetCapabilitiesForTesting();
        }
    }

    [Fact]
    public void TryPow2ModBatch_returns_overflow_for_marked_prime()
    {
        const ulong prime = 193UL;
        try
        {
            PrimeOrderGpuHeuristics.OverflowRegistry[prime] = 0;

            ulong[] remainders = new ulong[2];
            ulong[] exponents = { 1UL, 2UL };
            GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2ModBatch(exponents, prime, remainders);

            status.Should().Be(GpuPow2ModStatus.Overflow);
            remainders[0].Should().Be(0UL);
            remainders[1].Should().Be(0UL);
        }
        finally
        {
            PrimeOrderGpuHeuristics.OverflowRegistry.TryRemove(prime, out _);
        }
    }

    [Fact]
    public void TryPow2ModBatch_returns_success_for_supported_prime()
    {
        ulong[] remainders = { 123UL, 456UL };
        ulong[] exponents = { 1UL, 2UL };
        GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2ModBatch(exponents, 257UL, remainders);

        status.Should().Be(GpuPow2ModStatus.Success);
        remainders[0].Should().Be(2UL);
        remainders[1].Should().Be(4UL);
    }

    [Fact]
    public void TryPow2ModBatch_returns_overflow_when_modulus_exceeds_capability()
    {
        const ulong prime = 97UL;
        var capability = new PrimeOrderGpuHeuristics.PrimeOrderGpuCapability(6, 64);
        PrimeOrderGpuHeuristics.OverrideCapabilitiesForTesting(capability);
        try
        {
            ulong[] exponents = { 1UL, 2UL };
            ulong[] remainders = { 5UL, 6UL };

            GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2ModBatch(exponents, prime, remainders);

            status.Should().Be(GpuPow2ModStatus.Overflow);
            remainders[0].Should().Be(0UL);
            remainders[1].Should().Be(0UL);
        }
        finally
        {
            PrimeOrderGpuHeuristics.ResetCapabilitiesForTesting();
            PrimeOrderGpuHeuristics.OverflowRegistry.TryRemove(prime, out _);
        }
    }

    [Fact]
    public void TryPow2ModBatch_returns_overflow_when_any_exponent_exceeds_capability()
    {
        var capability = new PrimeOrderGpuHeuristics.PrimeOrderGpuCapability(64, 3);
        PrimeOrderGpuHeuristics.OverrideCapabilitiesForTesting(capability);
        try
        {
            ulong[] exponents = { 2UL, 16UL };
            ulong[] remainders = { 9UL, 11UL };

            GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2ModBatch(exponents, 193UL, remainders);

            status.Should().Be(GpuPow2ModStatus.Overflow);
            remainders[0].Should().Be(0UL);
            remainders[1].Should().Be(0UL);
        }
        finally
        {
            PrimeOrderGpuHeuristics.ResetCapabilitiesForTesting();
        }
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
        try
        {
            PrimeOrderGpuHeuristics.OverflowRegistry[prime] = 0;

            PrimeOrderCalculator.PrimeOrderResult result = PrimeOrderCalculator.Calculate(prime, null, PrimeOrderCalculator.PrimeOrderSearchConfig.HeuristicDefault);

            result.Status.Should().Be(PrimeOrderCalculator.PrimeOrderStatus.Found);
            result.Order.Should().Be(3UL);
        }
        finally
        {
            PrimeOrderGpuHeuristics.OverflowRegistry.TryRemove(prime, out _);
        }
    }
}
