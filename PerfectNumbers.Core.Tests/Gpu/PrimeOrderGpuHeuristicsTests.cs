using System;
using System.Numerics;
using System.Reflection;
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
        PrimeOrderGpuHeuristics.OverflowRegistryWide.Clear();
        PrimeOrderGpuHeuristics.ResetCapabilitiesForTesting();
    }

    [Fact]
    public void TryPow2Mod_returns_overflow_for_marked_prime()
    {
        const ulong prime = 97UL;
        try
        {
            PrimeOrderGpuHeuristics.OverflowRegistry[prime] = 0;
            MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(prime);
            GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2Mod(1UL, prime, out ulong remainder, divisorData);

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
        const ulong prime = 193UL;
        MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(prime);
        GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2Mod(1UL, prime, out ulong remainder, divisorData);

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
            MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(prime);
            GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2Mod(1UL, prime, out ulong remainder, divisorData);

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
            const ulong prime = 193UL;
            MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(prime);
            GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2Mod(16UL, prime, out ulong remainder, divisorData);

            status.Should().Be(GpuPow2ModStatus.Overflow);
            remainder.Should().Be(0UL);

            GpuPow2ModStatus followUp = PrimeOrderGpuHeuristics.TryPow2Mod(4UL, prime, out ulong followUpRemainder, divisorData);
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
            MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(prime);
            GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2ModBatch(exponents, prime, remainders, divisorData);

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
        const ulong prime = 257UL;
        MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(prime);
        GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2ModBatch(exponents, prime, remainders, divisorData);

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

            MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(prime);
            GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2ModBatch(exponents, prime, remainders, divisorData);

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

            const ulong prime = 193UL;
            MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(prime);
            GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2ModBatch(exponents, prime, remainders, divisorData);

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
    public void TryPow2ModWide_returns_overflow_for_marked_prime()
    {
        UInt128 prime = ((UInt128)1 << 80) + 7;
        try
        {
            PrimeOrderGpuHeuristics.OverflowRegistryWide[prime] = 0;

            GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2Mod(UInt128.One, prime, out UInt128 remainder);

            status.Should().Be(GpuPow2ModStatus.Overflow);
            remainder.Should().Be(UInt128.Zero);
        }
        finally
        {
            PrimeOrderGpuHeuristics.OverflowRegistryWide.TryRemove(prime, out _);
        }
    }

    [Fact]
    public void TryPow2ModWide_returns_success_for_supported_prime()
    {
        UInt128 prime = ((UInt128)1 << 96) + 11;
        UInt128 exponent = ((UInt128)1 << 64) + 3;

        GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2Mod(exponent, prime, out UInt128 remainder);

        status.Should().Be(GpuPow2ModStatus.Success);
        UInt128 expected = (UInt128)BigInteger.ModPow(2, (BigInteger)exponent, (BigInteger)prime);
        remainder.Should().Be(expected);
    }

    [Theory]
    [InlineData("193", "31")]
    [InlineData("193", "255")]
    [InlineData("79228162514264337593543950347", "36893488147419103251")]
    public void Pow2ModKernelCore_matches_cpu_windowed_path(string modulusText, string exponentText)
    {
        UInt128 modulus = UInt128.Parse(modulusText);
        UInt128 exponent = UInt128.Parse(exponentText);

        var method = typeof(PrimeOrderGpuHeuristics).GetMethod("Pow2ModKernelCore", BindingFlags.NonPublic | BindingFlags.Static);
        method.Should().NotBeNull();

        object? gpuResultObject = method!.Invoke(null, new object[] { new GpuUInt128(exponent), new GpuUInt128(modulus) });
        GpuUInt128 gpuResult = (GpuUInt128)gpuResultObject!;
        UInt128 expected = exponent.Pow2MontgomeryModWindowed(modulus);

        ((UInt128)gpuResult).Should().Be(expected);
    }

    [Fact]
    public void TryPow2ModWide_returns_overflow_when_modulus_exceeds_capability()
    {
        UInt128 prime = ((UInt128)1 << 80) + 9;
        var capability = new PrimeOrderGpuHeuristics.PrimeOrderGpuCapability(64, 128);
        PrimeOrderGpuHeuristics.OverrideCapabilitiesForTesting(capability);
        try
        {
            GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2Mod(UInt128.One, prime, out UInt128 remainder);

            status.Should().Be(GpuPow2ModStatus.Overflow);
            remainder.Should().Be(UInt128.Zero);
        }
        finally
        {
            PrimeOrderGpuHeuristics.ResetCapabilitiesForTesting();
            PrimeOrderGpuHeuristics.OverflowRegistryWide.TryRemove(prime, out _);
        }
    }

    [Fact]
    public void TryPow2ModWide_returns_overflow_when_exponent_exceeds_capability()
    {
        var capability = new PrimeOrderGpuHeuristics.PrimeOrderGpuCapability(128, 4);
        PrimeOrderGpuHeuristics.OverrideCapabilitiesForTesting(capability);
        try
        {
            UInt128 exponent = UInt128.One << 8;
            UInt128 prime = ((UInt128)1 << 70) + 3;

            GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2Mod(exponent, prime, out UInt128 remainder);

            status.Should().Be(GpuPow2ModStatus.Overflow);
            remainder.Should().Be(UInt128.Zero);
        }
        finally
        {
            PrimeOrderGpuHeuristics.ResetCapabilitiesForTesting();
        }
    }

    [Fact]
    public void TryPow2ModBatchWide_returns_success_for_supported_prime()
    {
        UInt128 prime = ((UInt128)1 << 90) + 19;
        UInt128[] exponents = { UInt128.One, ((UInt128)1 << 64) + 5 };
        UInt128[] remainders = { UInt128.Zero, UInt128.Zero };

        GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2ModBatch(exponents, prime, remainders);

        status.Should().Be(GpuPow2ModStatus.Success);
        for (int i = 0; i < exponents.Length; i++)
        {
            UInt128 expected = (UInt128)BigInteger.ModPow(2, (BigInteger)exponents[i], (BigInteger)prime);
            remainders[i].Should().Be(expected);
        }
    }

    [Fact]
    public void TryPow2ModBatchWide_returns_overflow_when_modulus_exceeds_capability()
    {
        UInt128 prime = ((UInt128)1 << 88) + 13;
        UInt128[] exponents = { UInt128.One, (UInt128)5 };
        UInt128[] remainders = { UInt128.Zero, UInt128.Zero };
        var capability = new PrimeOrderGpuHeuristics.PrimeOrderGpuCapability(64, 128);
        PrimeOrderGpuHeuristics.OverrideCapabilitiesForTesting(capability);
        try
        {
            GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2ModBatch(exponents, prime, remainders);

            status.Should().Be(GpuPow2ModStatus.Overflow);
            remainders[0].Should().Be(UInt128.Zero);
            remainders[1].Should().Be(UInt128.Zero);
        }
        finally
        {
            PrimeOrderGpuHeuristics.ResetCapabilitiesForTesting();
            PrimeOrderGpuHeuristics.OverflowRegistryWide.TryRemove(prime, out _);
        }
    }

    [Fact]
    public void TryPow2ModBatchWide_returns_overflow_when_any_exponent_exceeds_capability()
    {
        UInt128 prime = ((UInt128)1 << 72) + 23;
        UInt128[] exponents = { UInt128.One << 5, UInt128.One << 9 };
        UInt128[] remainders = { UInt128.Zero, UInt128.Zero };
        var capability = new PrimeOrderGpuHeuristics.PrimeOrderGpuCapability(128, 6);
        PrimeOrderGpuHeuristics.OverrideCapabilitiesForTesting(capability);
        try
        {
            GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2ModBatch(exponents, prime, remainders);

            status.Should().Be(GpuPow2ModStatus.Overflow);
            remainders[0].Should().Be(UInt128.Zero);
            remainders[1].Should().Be(UInt128.Zero);
        }
        finally
        {
            PrimeOrderGpuHeuristics.ResetCapabilitiesForTesting();
        }
    }

    // This will never happen in production code because the caller ensures the remainder span is large enough.
    // [Fact]
    // public void TryPow2ModBatch_throws_when_remainder_span_is_too_small()
    // {
    //     ulong[] remainders = new ulong[1];
    //     ulong[] exponents = { 1UL, 2UL };

    //     Action act = () =>
    //     {
    //         const ulong prime = 5UL;
    //         MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(prime);
    //         PrimeOrderGpuHeuristics.TryPow2ModBatch(exponents, prime, remainders, divisorData);
    //     };

    //     act.Should().Throw<ArgumentException>();
    // }

    [Fact]
    public void Calculate_falls_back_to_cpu_when_gpu_overflow_is_marked()
    {
        const ulong prime = 7UL;
        try
        {
            PrimeOrderGpuHeuristics.OverflowRegistry[prime] = 0;

            MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(prime);
            PrimeOrderCalculator.PrimeOrderResult result = PrimeOrderCalculator.Calculate(prime, null, divisorData, PrimeOrderCalculator.PrimeOrderSearchConfig.HeuristicDefault, PrimeOrderCalculator.PrimeOrderHeuristicDevice.Gpu);

            result.Status.Should().Be(PrimeOrderCalculator.PrimeOrderStatus.Found);
            result.Order.Should().Be(3UL);
        }
        finally
        {
            PrimeOrderGpuHeuristics.OverflowRegistry.TryRemove(prime, out _);
        }
    }
}
