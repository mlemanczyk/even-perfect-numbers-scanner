using FluentAssertions;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;
using Xunit;

namespace PerfectNumbers.Core.Tests.Gpu;

[Collection("GpuNtt")]
public class GpuContextPoolTests
{
    [Fact]
    [Trait("Category", "Fast")]
    public void RentPreferred_cpu_returns_cpu_accelerator()
    {
        GpuContextPool.DisposeAll();

        using var lease = GpuContextPool.RentPreferred(preferCpu: true);

        lease.Accelerator.AcceleratorType.Should().Be(AcceleratorType.CPU);

        GpuContextPool.DisposeAll();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void RentPreferred_reuses_cpu_context()
    {
        GpuContextPool.DisposeAll();

        Context firstContext;
        Accelerator firstAccelerator;
        using (var lease = GpuContextPool.RentPreferred(preferCpu: true))
        {
            firstContext = lease.Context;
            firstAccelerator = lease.Accelerator;
        }

        using (var lease = GpuContextPool.RentPreferred(preferCpu: true))
        {
            lease.Context.Should().BeSameAs(firstContext);
            lease.Accelerator.Should().BeSameAs(firstAccelerator);
        }

        GpuContextPool.DisposeAll();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void DisposeAll_clears_pooled_contexts()
    {
        GpuContextPool.DisposeAll();

        Context firstContext;
        using (var lease = GpuContextPool.RentPreferred(preferCpu: true))
        {
            firstContext = lease.Context;
        }

        GpuContextPool.DisposeAll();

        using (var lease = GpuContextPool.RentPreferred(preferCpu: true))
        {
            lease.Context.Should().NotBeSameAs(firstContext);
        }

        GpuContextPool.DisposeAll();
    }
}

