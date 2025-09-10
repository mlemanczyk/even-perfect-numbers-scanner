using FluentAssertions;
using PerfectNumbers.Core.Gpu;
using Xunit;

namespace PerfectNumbers.Core.Tests;

[Trait("Category", "Fast")]
public class GpuUInt128Pow2Minus1ModTests
{
    [Fact]
    public void Pow2Minus1Mod_matches_pow2mod_minus_one()
    {
        ulong exponent = 23UL;
        GpuUInt128 modulus = new(0UL, 97UL);
        var pow = GpuUInt128.Pow2Mod(exponent, modulus);
        var expected = pow.IsZero ? modulus.Sub(GpuUInt128.One) : pow.Sub(GpuUInt128.One);
        var actual = GpuUInt128.Pow2Minus1Mod(exponent, modulus);
        actual.Should().Be(expected);
    }

    [Fact]
    public void Pow2Minus1ModBatch_matches_individual()
    {
        ulong[] exponents = { 1UL, 2UL, 3UL, 4UL, 5UL, 6UL, 7UL, 8UL };
        Span<GpuUInt128> results = stackalloc GpuUInt128[exponents.Length];
        GpuUInt128 modulus = new(0UL, 97UL);
        GpuUInt128.Pow2Minus1ModBatch(modulus, exponents, results);
        for (int i = 0; i < exponents.Length; i++)
        {
            var single = GpuUInt128.Pow2Minus1Mod(exponents[i], modulus);
            results[i].Should().Be(single);
        }
    }
}
