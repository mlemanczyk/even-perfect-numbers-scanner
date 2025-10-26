using FluentAssertions;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;
using Xunit;

namespace PerfectNumbers.Core.Tests.Gpu;

[Trait("Category", "Fast")]
public class PrimeTesterGpuTests
{
    [Fact]
    public void IsPrimeGpu_accepts_known_primes()
    {
        GpuContextPool.ForceCpu = false;
        try
        {
            var tester = new PrimeTester();
            ulong[] primes = [31UL, 61UL, 89UL, 107UL, 127UL, 521UL];
            foreach (ulong prime in primes)
            {
                tester.IsPrimeGpu(prime, CancellationToken.None).Should().BeTrue();
            }
        }
        finally
        {
            GpuContextPool.ForceCpu = false;
            GpuContextPool.DisposeAll();
        }
    }

    [Fact]
    public void IsPrimeBatchGpu_with_preallocated_resources_marks_expected_values()
    {
        GpuContextPool.ForceCpu = false;
        using var context = Context.CreateDefault();
        using var accelerator = context.GetPreferredDevice(false).CreateAccelerator(context);

        try
        {
            ulong[] values = [31UL, 33UL, 37UL, 39UL];
            byte[] results = new byte[values.Length];

            using var inputBuffer = accelerator.Allocate1D<ulong>(values.Length);
            using var outputBuffer = accelerator.Allocate1D<byte>(values.Length);
            ulong[] staging = new ulong[values.Length];

            PrimeTester.IsPrimeBatchGpu(values, results, accelerator, inputBuffer, outputBuffer, staging);

            results.Should().Equal(new byte[] { 1, 0, 1, 0 });
        }
        finally
        {
            PrimeTester.ClearGpuCaches(accelerator);
            GpuContextPool.ForceCpu = false;
            GpuContextPool.DisposeAll();
        }
    }

    [Fact]
    public void IsPrimeGpu_rejects_composites()
    {
        GpuContextPool.ForceCpu = false;
        try
        {
            var tester = new PrimeTester();
            ulong[] composites = [33UL, 39UL, 51UL, 77UL, 91UL];
            foreach (ulong composite in composites)
            {
                tester.IsPrimeGpu(composite, CancellationToken.None).Should().BeFalse();
            }
        }
        finally
        {
            GpuContextPool.ForceCpu = false;
            GpuContextPool.DisposeAll();
        }
    }

}
