using FluentAssertions;
using PerfectNumbers.Core.Gpu;
using Xunit;

namespace PerfectNumbers.Core.Tests;

[Collection("GpuNtt")]
public class GpuKernelPoolTests
{
    [Fact]
    [Trait("Category", "Fast")]
    public void Run_provides_accelerator_and_stream()
    {
        bool executed = false;

        GpuKernelPool.Run((acc, stream) =>
        {
            acc.Should().NotBeNull();
            stream.Should().NotBeNull();
            executed = true;
        });

        executed.Should().BeTrue();
    }
}
