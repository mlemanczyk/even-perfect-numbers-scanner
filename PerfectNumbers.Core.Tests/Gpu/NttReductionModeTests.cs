using FluentAssertions;
using PerfectNumbers.Core.Gpu;
using Xunit;

namespace PerfectNumbers.Core.Tests.Gpu;

[Collection("GpuNtt")]
public class NttReductionModeTests
{
    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(ModReductionMode.Auto)]
    [InlineData(ModReductionMode.GpuUInt128)]
    public void ForwardInverse_small64bit_modulus_reference_backend_works_with_mode(ModReductionMode mode)
    {
        var originalBackend = NttGpuMath.GpuTransformBackend;
        var originalMode = NttGpuMath.ReductionMode;
        try
        {
            NttGpuMath.GpuTransformBackend = NttBackend.Reference;
            NttGpuMath.ReductionMode = mode;

            var modulus = new GpuUInt128(0UL, 17UL);
            var primitiveRoot = new GpuUInt128(0UL, 3UL);
            Span<GpuUInt128> input = stackalloc GpuUInt128[8];
            for (ulong i = 0; i < (ulong)input.Length; i++)
            {
                input[(int)i] = new GpuUInt128(0UL, i + 1UL);
            }

            Span<GpuUInt128> cpu = stackalloc GpuUInt128[input.Length];
            input.CopyTo(cpu);
            NttGpuMath.Forward(cpu, modulus, primitiveRoot);
            NttGpuMath.Inverse(cpu, modulus, primitiveRoot);

            Span<GpuUInt128> gpu = stackalloc GpuUInt128[input.Length];
            input.CopyTo(gpu);
            NttGpuMath.ForwardGpu(gpu, modulus, primitiveRoot);
            NttGpuMath.InverseGpu(gpu, modulus, primitiveRoot);

            gpu.ToArray().Should().Equal(cpu.ToArray());
        }
        finally
        {
            NttGpuMath.GpuTransformBackend = originalBackend;
            NttGpuMath.ReductionMode = originalMode;
        }
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(ModReductionMode.Auto)]
    [InlineData(ModReductionMode.GpuUInt128)]
    public void ForwardInverse_small64bit_modulus_staged_backend_works_with_mode(ModReductionMode mode)
    {
        var originalBackend = NttGpuMath.GpuTransformBackend;
        var originalMode = NttGpuMath.ReductionMode;
        try
        {
            NttGpuMath.GpuTransformBackend = NttBackend.Staged;
            NttGpuMath.ReductionMode = mode;

            var modulus = new GpuUInt128(0UL, 17UL);
            var primitiveRoot = new GpuUInt128(0UL, 3UL);
            Span<GpuUInt128> input = stackalloc GpuUInt128[8];
            for (ulong i = 0; i < (ulong)input.Length; i++)
            {
                input[(int)i] = new GpuUInt128(0UL, i + 1UL);
            }

            Span<GpuUInt128> cpu = stackalloc GpuUInt128[input.Length];
            input.CopyTo(cpu);
            NttGpuMath.Forward(cpu, modulus, primitiveRoot);
            NttGpuMath.Inverse(cpu, modulus, primitiveRoot);

            Span<GpuUInt128> gpu = stackalloc GpuUInt128[input.Length];
            input.CopyTo(gpu);
            NttGpuMath.ForwardGpu(gpu, modulus, primitiveRoot);
            NttGpuMath.InverseGpu(gpu, modulus, primitiveRoot);

            gpu.ToArray().Should().Equal(cpu.ToArray());
        }
        finally
        {
            NttGpuMath.GpuTransformBackend = originalBackend;
            NttGpuMath.ReductionMode = originalMode;
        }
    }
}

