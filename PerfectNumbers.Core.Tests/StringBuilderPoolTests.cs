using FluentAssertions;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class StringBuilderPoolTests
{
    [Fact]
    [Trait("Category", "Fast")]
    public void Rent_ReturnsClearedBuilder()
    {
        var sb = StringBuilderPool.Rent();

        sb.Length.Should().Be(0);

        StringBuilderPool.Return(sb);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void Return_AllowsReuseOfBuilder()
    {
        var sb = StringBuilderPool.Rent();
        sb.Append("data");

        StringBuilderPool.Return(sb);

        var reused = StringBuilderPool.Rent();

        reused.Should().BeSameAs(sb);
        reused.Length.Should().Be(0);

        StringBuilderPool.Return(reused);
    }
}

