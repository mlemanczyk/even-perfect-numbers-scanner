using FluentAssertions;
using PerfectNumbers.Core;
using Xunit;

namespace PerfectNumbers.Core.Tests.Gpu;

public class ULongExtensionsGpuTests
{
    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(1UL, 3UL)]
    [InlineData(2UL, 5UL)]
    [InlineData(3UL, 7UL)]
    [InlineData(63UL, 97UL)]
    [InlineData(512UL, 12289UL)]
    [InlineData(1024UL, 196613UL)]
    [InlineData(4096UL, 999983UL)]
    [InlineData(65535UL, 2147483647UL)]
    public void Pow2ModWindowedGpu_matches_montgomery_cpu_for_known_values(ulong exponent, ulong modulus)
    {
        MontgomeryDivisorData divisor = MontgomeryDivisorData.FromModulus(modulus);
        ulong expected = exponent.Pow2MontgomeryModWindowedCpu(divisor, keepMontgomery: false);

        exponent.Pow2ModWindowedGpu(modulus).Should().Be(expected);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void Pow2ModWindowedGpu_matches_cpu_for_sampled_inputs()
    {
        ulong[] moduli =
        {
            3UL,
            5UL,
            97UL,
            12289UL,
            196613UL,
            4294967291UL,
            9223372036854775783UL,
        };

        ulong[] exponents =
        {
            1UL,
            2UL,
            7UL,
            32UL,
            63UL,
            2048UL,
            65535UL,
        };

        foreach (ulong modulus in moduli)
        {
            MontgomeryDivisorData divisor = MontgomeryDivisorData.FromModulus(modulus);
            foreach (ulong exponent in exponents)
            {
                ulong expected = exponent.Pow2MontgomeryModWindowedCpu(divisor, keepMontgomery: false);
                exponent.Pow2ModWindowedGpu(modulus).Should().Be(expected);
            }
        }
    }

}
