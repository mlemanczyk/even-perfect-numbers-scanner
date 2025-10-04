using System.Numerics;
using FluentAssertions;
using PerfectNumbers.Core;
using Xunit;

namespace PerfectNumbers.Core.Tests;

[Trait("Category", "Fast")]
public class ULongExtensionsMultiplyShiftTests
{
    [Theory]
    [InlineData(0UL, 0UL, 1)]
    [InlineData(1UL, 3UL, 1)]
    [InlineData(ulong.MaxValue, 3UL, 3)]
    [InlineData(0xFFFF_FFFF_FFFF_FFFFUL, 5UL, 4)]
    [InlineData(0x8000_0000_0000_0000UL, 7UL, 5)]
    public void MultiplyShiftRight_matches_big_integer(ulong value, ulong multiplier, int shift)
    {
        BigInteger expected = ((BigInteger)value * multiplier) >> shift;

        ulong result = ULongExtensions.MultiplyShiftRight(value, multiplier, shift);

        result.Should().Be((ulong)expected);
    }

    [Theory]
    [InlineData(0UL, 0UL, 1)]
    [InlineData(1UL, 3UL, 1)]
    [InlineData(ulong.MaxValue, 3UL, 3)]
    [InlineData(0xFFFF_FFFF_FFFF_FFFFUL, 5UL, 4)]
    [InlineData(0x8000_0000_0000_0000UL, 7UL, 5)]
    [InlineData(0xDEAD_BEEF_DEAD_BEEFUL, 11UL, 63)]
    public void MultiplyShiftRightShiftFirst_matches_big_integer(ulong value, ulong multiplier, int shift)
    {
        BigInteger expected = ((BigInteger)value * multiplier) >> shift;

        ulong result = ULongExtensions.MultiplyShiftRightShiftFirst(value, multiplier, shift);

        result.Should().Be((ulong)expected);
    }

    [Fact]
    public void MultiplyShiftRightShiftFirst_returns_zero_when_shift_is_large()
    {
        ULongExtensions.MultiplyShiftRightShiftFirst(ulong.MaxValue, 123UL, 64).Should().Be(0UL);
    }

    [Theory]
    [InlineData(0UL, ulong.MaxValue, -1)]
    [InlineData(1UL, 1UL, -3)]
    public void MultiplyShiftRight_throws_when_shift_is_negative(ulong value, ulong multiplier, int shift)
    {
        Action act = () => ULongExtensions.MultiplyShiftRight(value, multiplier, shift);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Theory]
    [InlineData(0UL, ulong.MaxValue, -1)]
    [InlineData(1UL, 1UL, -3)]
    public void MultiplyShiftRightShiftFirst_throws_when_shift_is_negative(ulong value, ulong multiplier, int shift)
    {
        Action act = () => ULongExtensions.MultiplyShiftRightShiftFirst(value, multiplier, shift);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}
