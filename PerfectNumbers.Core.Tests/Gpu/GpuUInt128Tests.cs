using FluentAssertions;
using PerfectNumbers.Core.Gpu;
using Xunit;

[Trait("Category", "Fast")]
public class GpuUInt128Tests
{
    [Fact]
    public void Mul_matches_UInt128()
    {
        var random = new Random(0);
        for (var i = 0; i < 100; i++)
        {
            var high1 = ((ulong)random.Next() << 32) | (uint)random.Next();
            var low1 = ((ulong)random.Next() << 32) | (uint)random.Next();
            var high2 = ((ulong)random.Next() << 32) | (uint)random.Next();
            var low2 = ((ulong)random.Next() << 32) | (uint)random.Next();
            var a = new GpuUInt128(high1, low1);
            var b = new GpuUInt128(high2, low2);
            var expected = (UInt128)new GpuUInt128(high1, low1) * (UInt128)new GpuUInt128(high2, low2);
            a.Mul(b);
            ((UInt128)a).Should().Be(expected);
        }
    }

    [Fact]
    public void MulMod_matches_UInt128()
    {
        var random = new Random(1);
        for (var i = 0; i < 100; i++)
        {
            var high1 = ((ulong)random.Next() << 32) | (uint)random.Next();
            var low1 = ((ulong)random.Next() << 32) | (uint)random.Next();
            var high2 = ((ulong)random.Next() << 32) | (uint)random.Next();
            var low2 = ((ulong)random.Next() << 32) | (uint)random.Next();
            var modHigh = ((ulong)random.Next() << 32) | (uint)random.Next();
            var modLow = ((ulong)random.Next() << 32) | (uint)random.Next() | 1UL;
            var modulus = new GpuUInt128(modHigh, modLow);
            var a = new GpuUInt128((UInt128)new GpuUInt128(high1, low1) % (UInt128)modulus);
            var b = new GpuUInt128((UInt128)new GpuUInt128(high2, low2) % (UInt128)modulus);
            var expected = MulModReference((UInt128)a, (UInt128)b, (UInt128)modulus);
            a.MulMod(b, modulus);
            ((UInt128)a).Should().Be(expected);
        }
    }

    [Fact]
    public void MulMod_with_ulong_matches_UInt128()
    {
        var random = new Random(5);
        for (var i = 0; i < 100; i++)
        {
            var high1 = ((ulong)random.Next() << 32) | (uint)random.Next();
            var low1 = ((ulong)random.Next() << 32) | (uint)random.Next();
            var modHigh = ((ulong)random.Next() << 32) | (uint)random.Next();
            var modLow = ((ulong)random.Next() << 32) | (uint)random.Next() | 1UL;
            var modulus = new GpuUInt128(modHigh, modLow);
            var a = new GpuUInt128((UInt128)new GpuUInt128(high1, low1) % (UInt128)modulus);
            ulong bRaw = ((ulong)random.Next() << 32) | (uint)random.Next();
            UInt128 bBig = (UInt128)bRaw % (UInt128)modulus;
            ulong b = (ulong)bBig;
            var expected = MulModReference((UInt128)a, bBig, (UInt128)modulus);
            a.MulMod(b, modulus);
            ((UInt128)a).Should().Be(expected);
        }
    }

    [Fact]
    public void MulMod_with_ulong_modulus_matches_UInt128()
    {
        var random = new Random(11);
        for (var i = 0; i < 100; i++)
        {
            ulong modulus = ((ulong)random.Next() << 32) | (uint)random.Next() | 1UL;
            ulong aRaw = ((ulong)random.Next() << 32) | (uint)random.Next();
            ulong bRaw = ((ulong)random.Next() << 32) | (uint)random.Next();
            ulong aVal = aRaw % modulus;
            ulong bVal = bRaw % modulus;
            var a = new GpuUInt128(aVal);
            var b = new GpuUInt128(bVal);
            UInt128 expected = ((UInt128)aVal * bVal) % modulus;
            ulong result = a.MulMod(b, modulus);
            ((UInt128)result).Should().Be(expected);
        }
    }

    [Fact]
    public void SquareMod_matches_UInt128()
    {
        var random = new Random(8);
        for (var i = 0; i < 100; i++)
        {
            var valueHigh = ((ulong)random.Next() << 32) | (uint)random.Next();
            var valueLow = ((ulong)random.Next() << 32) | (uint)random.Next();
            var modHigh = ((ulong)random.Next() << 32) | (uint)random.Next();
            var modLow = ((ulong)random.Next() << 32) | (uint)random.Next() | 1UL;
            var modulus = new GpuUInt128(modHigh, modLow);
            var a = new GpuUInt128((UInt128)new GpuUInt128(valueHigh, valueLow) % (UInt128)modulus);
            var expected = MulModReference((UInt128)a, (UInt128)a, (UInt128)modulus);
            a.SquareMod(modulus);
            ((UInt128)a).Should().Be(expected);
        }
    }

    [Fact]
    public void ModPow_matches_BigInteger()
    {
        var random = new Random(2);
        for (var i = 0; i < 20; i++)
        {
            var baseHigh = ((ulong)random.Next() << 32) | (uint)random.Next();
            var baseLow = ((ulong)random.Next() << 32) | (uint)random.Next();
            var expHigh = 0UL;
            var expLow = (ulong)random.Next(1, int.MaxValue);
            var modHigh = ((ulong)random.Next() << 32) | (uint)random.Next();
            var modLow = ((ulong)random.Next() << 32) | (uint)random.Next() | 1UL;

            var modulus = new GpuUInt128(modHigh, modLow);
            var value = new GpuUInt128((UInt128)new GpuUInt128(baseHigh, baseLow) % (UInt128)modulus);
            var exponent = new GpuUInt128(expHigh, expLow);

            var expected = (UInt128)System.Numerics.BigInteger.ModPow((System.Numerics.BigInteger)(UInt128)value, (System.Numerics.BigInteger)(UInt128)exponent, (System.Numerics.BigInteger)(UInt128)modulus);
            value.ModPow(exponent, modulus);
            ((UInt128)value).Should().Be(expected);
        }
    }

    [Fact]
    public void ModPow_with_ulong_matches_BigInteger()
    {
        var random = new Random(9);
        for (var i = 0; i < 20; i++)
        {
            var baseHigh = ((ulong)random.Next() << 32) | (uint)random.Next();
            var baseLow = ((ulong)random.Next() << 32) | (uint)random.Next();
            ulong exponent = (ulong)random.Next(1, int.MaxValue);
            var modHigh = ((ulong)random.Next() << 32) | (uint)random.Next();
            var modLow = ((ulong)random.Next() << 32) | (uint)random.Next() | 1UL;

            var modulus = new GpuUInt128(modHigh, modLow);
            var value = new GpuUInt128((UInt128)new GpuUInt128(baseHigh, baseLow) % (UInt128)modulus);

            var expected = (UInt128)System.Numerics.BigInteger.ModPow((System.Numerics.BigInteger)(UInt128)value, exponent, (System.Numerics.BigInteger)(UInt128)modulus);
            value.ModPow(exponent, modulus);
            ((UInt128)value).Should().Be(expected);
        }
    }

    [Fact]
    public void SubMod_matches_UInt128()
    {
        var random = new Random(3);
        for (var i = 0; i < 100; i++)
        {
            var modHigh = ((ulong)random.Next() << 32) | (uint)random.Next();
            var modLow = ((ulong)random.Next() << 32) | (uint)random.Next() | 1UL;
            var modulus = new GpuUInt128(modHigh, modLow);
            var valueHigh = ((ulong)random.Next() << 32) | (uint)random.Next();
            var valueLow = ((ulong)random.Next() << 32) | (uint)random.Next();
            var subHigh = ((ulong)random.Next() << 32) | (uint)random.Next();
            var subLow = ((ulong)random.Next() << 32) | (uint)random.Next();
            var a = new GpuUInt128((UInt128)new GpuUInt128(valueHigh, valueLow) % (UInt128)modulus);
            var b = new GpuUInt128((UInt128)new GpuUInt128(subHigh, subLow) % (UInt128)modulus);
            UInt128 expected = ((UInt128)a + (UInt128)modulus - (UInt128)b) % (UInt128)modulus;
            a.SubMod(b, modulus);
            ((UInt128)a).Should().Be(expected);
        }
    }

    [Fact]
    public void SubMod_with_ulong_matches_UInt128()
    {
        var random = new Random(6);
        for (var i = 0; i < 100; i++)
        {
            var modHigh = ((ulong)random.Next() << 32) | (uint)random.Next();
            var modLow = ((ulong)random.Next() << 32) | (uint)random.Next() | 1UL;
            var modulus = new GpuUInt128(modHigh, modLow);
            var valueHigh = ((ulong)random.Next() << 32) | (uint)random.Next();
            var valueLow = ((ulong)random.Next() << 32) | (uint)random.Next();
            var a = new GpuUInt128((UInt128)new GpuUInt128(valueHigh, valueLow) % (UInt128)modulus);
            ulong subRaw = ((ulong)random.Next() << 32) | (uint)random.Next();
            UInt128 bBig = (UInt128)subRaw % (UInt128)modulus;
            ulong b = (ulong)bBig;
            UInt128 expected = ((UInt128)a + (UInt128)modulus - bBig) % (UInt128)modulus;
            a.SubMod(b, modulus);
            ((UInt128)a).Should().Be(expected);
        }
    }

    [Fact]
    public void AddMod_matches_UInt128()
    {
        var random = new Random(4);
        for (var i = 0; i < 100; i++)
        {
            var modHigh = ((ulong)random.Next() << 32) | (uint)random.Next();
            var modLow = ((ulong)random.Next() << 32) | (uint)random.Next() | 1UL;
            var modulus = new GpuUInt128(modHigh, modLow);
            var valueHigh = ((ulong)random.Next() << 32) | (uint)random.Next();
            var valueLow = ((ulong)random.Next() << 32) | (uint)random.Next();
            var addHigh = ((ulong)random.Next() << 32) | (uint)random.Next();
            var addLow = ((ulong)random.Next() << 32) | (uint)random.Next();
            var a = new GpuUInt128((UInt128)new GpuUInt128(valueHigh, valueLow) % (UInt128)modulus);
            var b = new GpuUInt128((UInt128)new GpuUInt128(addHigh, addLow) % (UInt128)modulus);
            UInt128 expected = ((UInt128)a + (UInt128)b) % (UInt128)modulus;
            a.AddMod(b, modulus);
            ((UInt128)a).Should().Be(expected);
        }
    }

    [Fact]
    public void AddMod_with_ulong_modulus_matches_UInt128()
    {
        var random = new Random(12);
        for (var i = 0; i < 100; i++)
        {
            ulong modulus = ((ulong)random.Next() << 32) | (uint)random.Next() | 1UL;
            ulong aRaw = ((ulong)random.Next() << 32) | (uint)random.Next();
            ulong bRaw = ((ulong)random.Next() << 32) | (uint)random.Next();
            ulong aVal = aRaw % modulus;
            ulong bVal = bRaw % modulus;
            var a = new GpuUInt128(aVal);
            var b = new GpuUInt128(bVal);
            UInt128 expected = ((UInt128)aVal + bVal) % modulus;
            a.AddMod(b, modulus);
            ((UInt128)a).Should().Be(expected);
        }
    }

    [Fact]
    public void AddMod_with_ulong_operand_and_modulus_matches_UInt128()
    {
        var random = new Random(13);
        for (var i = 0; i < 100; i++)
        {
            ulong modulus = ((ulong)random.Next() << 32) | (uint)random.Next() | 1UL;
            ulong aRaw = ((ulong)random.Next() << 32) | (uint)random.Next();
            ulong bRaw = ((ulong)random.Next() << 32) | (uint)random.Next();
            ulong aVal = aRaw % modulus;
            ulong bVal = bRaw % modulus;
            var a = new GpuUInt128(aVal);
            UInt128 expected = ((UInt128)aVal + bVal) % modulus;
            a.AddMod(bVal, modulus);
            ((UInt128)a).Should().Be(expected);
        }
    }

    [Fact]
    public void ModInv_matches_BigInteger()
    {
        var random = new Random(7);
        for (var i = 0; i < 20; i++)
        {
            var modHigh = ((ulong)random.Next() << 32) | (uint)random.Next();
            var modLow = ((ulong)random.Next() << 32) | (uint)random.Next() | 1UL;
            var modulus = new GpuUInt128(modHigh, modLow);
            var valueHigh = ((ulong)random.Next() << 32) | (uint)random.Next();
            var valueLow = ((ulong)random.Next() << 32) | (uint)random.Next();
            var value = new GpuUInt128((UInt128)new GpuUInt128(valueHigh, valueLow) % (UInt128)modulus);
            if (value.IsZero)
            {
                value.Add(1UL);
            }

            var expected = (UInt128)System.Numerics.BigInteger.ModPow((System.Numerics.BigInteger)(UInt128)value, (System.Numerics.BigInteger)(UInt128)modulus - 2, (System.Numerics.BigInteger)(UInt128)modulus);
            value.ModInv(modulus);
            ((UInt128)value).Should().Be(expected);
        }
    }

    [Fact]
    public void ModInv_with_ulong_modulus_matches_BigInteger()
    {
        var random = new Random(8);
        for (var i = 0; i < 20; i++)
        {
            ulong modulus = ((ulong)random.Next() << 32) | (uint)random.Next() | 1UL;
            ulong raw = ((ulong)random.Next() << 32) | (uint)random.Next();
            var value = new GpuUInt128((UInt128)(raw % modulus));
            if (value.IsZero)
            {
                value.Add(1UL);
            }

            var expected = (UInt128)System.Numerics.BigInteger.ModPow((System.Numerics.BigInteger)(UInt128)value, (System.Numerics.BigInteger)modulus - 2, (System.Numerics.BigInteger)modulus);
            value.ModInv(modulus);
            ((UInt128)value).Should().Be(expected);
        }
    }

    private static UInt128 MulModReference(UInt128 a, UInt128 b, UInt128 modulus)
    {
        UInt128 result = 0;
        UInt128 x = a;
        UInt128 y = b;
        while (y != 0)
        {
            if ((y & 1) != 0)
            {
                result += x;
                if (result >= modulus)
                {
                    result -= modulus;
                }
            }

            x <<= 1;
            if (x >= modulus)
            {
                x -= modulus;
            }

            y >>= 1;
        }

        return result;
    }
}

