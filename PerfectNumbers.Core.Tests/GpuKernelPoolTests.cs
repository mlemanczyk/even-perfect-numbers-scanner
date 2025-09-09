using FluentAssertions;
using PerfectNumbers.Core.Gpu;
using Xunit;

namespace PerfectNumbers.Core.Tests;

[Collection("GpuNtt")]
public class GpuKernelPoolTests
{
    [Fact]
    [Trait("Category", "Fast")]
    public void EnsureSmallCyclesOnDevice_returns_same_view_on_subsequent_calls()
    {
        using var lease = GpuContextPool.RentPreferred(preferCpu: true);
        var accelerator = lease.Accelerator;

        var first = GpuKernelPool.EnsureSmallCyclesOnDevice(accelerator);
        var second = GpuKernelPool.EnsureSmallCyclesOnDevice(accelerator);

        second.Equals(first).Should().BeTrue();
        second.Length.Should().Be(MersenneDivisorCycles.SmallDivisorsMax + 1);
    }

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
