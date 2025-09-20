using System.Numerics;
using System.Runtime.CompilerServices;
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
        GpuUInt128 expected;
        if (pow.IsZero)
        {
            expected = modulus;
            expected.Sub(GpuUInt128.One);
        }
        else
        {
            expected = pow;
            expected.Sub(GpuUInt128.One);
        }
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

    [Fact]
    public void Pow2Minus1Mod_matches_bigint_for_large_mersenne_residue_candidates()
    {
        const ulong exponent = 107UL;
        const int limit = 500_000;
        UInt128 twoP = (UInt128)(2UL * exponent);

        for (int k = 1; k <= limit; k++)
        {
            UInt128 q = checked(twoP * (UInt128)(ulong)k + UInt128.One);
            GpuUInt128 gpuRemainder = GpuUInt128.Pow2Minus1Mod(exponent, (GpuUInt128)q);

            BigInteger bigQ = ToBigInteger(q);
            BigInteger bigPow = BigInteger.ModPow(new BigInteger(2), checked((int)exponent), bigQ);
            BigInteger bigRemainder = (bigPow - BigInteger.One) % bigQ;
            if (bigRemainder.Sign < 0)
            {
                bigRemainder += bigQ;
            }

            gpuRemainder.Should().Be(ToGpuUInt128(bigRemainder));
        }

        UInt128[] largeKs =
        {
            UInt128.Parse("1000000000000000"),
            UInt128.Parse("1000000000000000000"),
            UInt128.Parse("1000000000000000000000"),
            UInt128.Parse("99999999998192000000000")
        };

        foreach (UInt128 kValue in largeKs)
        {
            UInt128 q = checked(twoP * kValue + UInt128.One);
            GpuUInt128 gpuRemainder = GpuUInt128.Pow2Minus1Mod(exponent, (GpuUInt128)q);

            BigInteger bigQ = ToBigInteger(q);
            BigInteger bigPow = BigInteger.ModPow(new BigInteger(2), checked((int)exponent), bigQ);
            BigInteger bigRemainder = (bigPow - BigInteger.One) % bigQ;
            if (bigRemainder.Sign < 0)
            {
                bigRemainder += bigQ;
            }

            gpuRemainder.Should().Be(ToGpuUInt128(bigRemainder));
        }
    }

    private static BigInteger ToBigInteger(UInt128 value)
    {
        Span<byte> buffer = stackalloc byte[16];
        buffer.Clear();
        Unsafe.WriteUnaligned(ref buffer[0], value);
        return new BigInteger(buffer, isUnsigned: true, isBigEndian: false);
    }

    private static GpuUInt128 ToGpuUInt128(BigInteger value)
    {
        Span<byte> buffer = stackalloc byte[16];
        buffer.Clear();
        value.TryWriteBytes(buffer, out _, isUnsigned: true, isBigEndian: false);
        UInt128 converted = Unsafe.ReadUnaligned<UInt128>(ref buffer[0]);
        return (GpuUInt128)converted;
    }
}
