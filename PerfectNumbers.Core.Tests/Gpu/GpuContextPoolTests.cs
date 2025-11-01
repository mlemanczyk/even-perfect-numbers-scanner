using FluentAssertions;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;
using Xunit;

namespace PerfectNumbers.Core.Tests.Gpu;

[Collection("GpuNtt")]
public class GpuContextPoolTests
{
    // [Fact]
    // [Trait("Category", "Fast")]
    // public void RentPreferred_cpu_returns_cpu_accelerator()
    // {
    //     GpuContextPool.DisposeAll();

    //     GpuContextPool.GpuContextLease lease = GpuContextPool.RentPreferred(preferCpu: true);
    //     try
    //     {
    //         lease.Accelerator.AcceleratorType.Should().Be(AcceleratorType.CPU);
    //     }
    //     finally
    //     {
    //         lease.Dispose();
    //     }

    //     GpuContextPool.DisposeAll();
    // }

    // [Fact]
    // [Trait("Category", "Fast")]
    // public void RentPreferred_reuses_cpu_context()
    // {
    //     GpuContextPool.DisposeAll();

    //     Context firstContext;
    //     Accelerator firstAccelerator;
    //     GpuContextPool.GpuContextLease lease = GpuContextPool.RentPreferred(preferCpu: true);
    //     try
    //     {
    //         firstContext = lease.Context;
    //         firstAccelerator = lease.Accelerator;
    //     }
    //     finally
    //     {
    //         lease.Dispose();
    //     }

    //     lease = GpuContextPool.RentPreferred(preferCpu: true);
    //     try
    //     {
    //         lease.Context.Should().BeSameAs(firstContext);
    //         lease.Accelerator.Should().BeSameAs(firstAccelerator);
    //     }
    //     finally
    //     {
    //         lease.Dispose();
    //     }

    //     GpuContextPool.DisposeAll();
    // }

    // [Fact]
    // [Trait("Category", "Fast")]
    // public void DisposeAll_clears_pooled_contexts()
    // {
    //     GpuContextPool.DisposeAll();

    //     Context firstContext;
    //     GpuContextPool.GpuContextLease lease = GpuContextPool.RentPreferred(preferCpu: true);
    //     try
    //     {
    //         firstContext = lease.Context;
    //     }
    //     finally
    //     {
    //         lease.Dispose();
    //     }

    //     GpuContextPool.DisposeAll();

    //     lease = GpuContextPool.RentPreferred(preferCpu: true);
    //     try
    //     {
    //         lease.Context.Should().NotBeSameAs(firstContext);
    //     }
    //     finally
    //     {
    //         lease.Dispose();
    //     }

    //     GpuContextPool.DisposeAll();
    // }
}

