using FluentAssertions;
using PerfectNumbers.Core.Gpu;
using Xunit;

namespace PerfectNumbers.Core.Tests.Gpu;

[Collection("GpuNtt")]
public class NttBackendConfigTests
{
    [Fact]
    [Trait("Category", "Fast")]
    public void ForwardInverseGpu_reference_backend_matches_cpu_on_small_input()
    {
        var original = NttGpuMath.GpuTransformBackend;
        try
        {
            Span<GpuUInt128> input = stackalloc GpuUInt128[8];
            for (ulong i = 0; i < (ulong)input.Length; i++)
            {
                input[(int)i] = new GpuUInt128(0UL, i + 1UL);
            }

            var modulus = new GpuUInt128(0UL, 17UL);
            var primitiveRoot = new GpuUInt128(0UL, 3UL);

            NttGpuMath.GpuTransformBackend = NttBackend.Reference;
            Span<GpuUInt128> gpu = stackalloc GpuUInt128[input.Length];
            input.CopyTo(gpu);
            NttGpuMath.ForwardGpu(gpu, modulus, primitiveRoot);
            NttGpuMath.InverseGpu(gpu, modulus, primitiveRoot);

            Span<GpuUInt128> cpu = stackalloc GpuUInt128[input.Length];
            input.CopyTo(cpu);
            NttGpuMath.Forward(cpu, modulus, primitiveRoot);
            NttGpuMath.Inverse(cpu, modulus, primitiveRoot);
            gpu.ToArray().Should().Equal(cpu.ToArray());
        }
        finally
        {
            NttGpuMath.GpuTransformBackend = original;
        }
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ForwardInverseGpu_staged_backend_matches_cpu_on_small_input()
    {
        var original = NttGpuMath.GpuTransformBackend;
        try
        {
            Span<GpuUInt128> input = stackalloc GpuUInt128[8];
            for (ulong i = 0; i < (ulong)input.Length; i++)
            {
                input[(int)i] = new GpuUInt128(0UL, i + 1UL);
            }

            var modulus = new GpuUInt128(0UL, 17UL);
            var primitiveRoot = new GpuUInt128(0UL, 3UL);

            NttGpuMath.GpuTransformBackend = NttBackend.Staged;
            Span<GpuUInt128> gpu = stackalloc GpuUInt128[input.Length];
            input.CopyTo(gpu);
            NttGpuMath.ForwardGpu(gpu, modulus, primitiveRoot);
            NttGpuMath.InverseGpu(gpu, modulus, primitiveRoot);

            Span<GpuUInt128> cpu = stackalloc GpuUInt128[input.Length];
            input.CopyTo(cpu);
            NttGpuMath.Forward(cpu, modulus, primitiveRoot);
            NttGpuMath.Inverse(cpu, modulus, primitiveRoot);
            gpu.ToArray().Should().Equal(cpu.ToArray());
        }
        finally
        {
            NttGpuMath.GpuTransformBackend = original;
        }
    }
}

