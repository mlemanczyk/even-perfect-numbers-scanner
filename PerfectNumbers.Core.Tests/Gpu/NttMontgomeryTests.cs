using FluentAssertions;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;
using Xunit;

namespace PerfectNumbers.Core.Tests.Gpu;

[Collection("GpuNtt")]
[Trait("Category", "Fast")]
public class NttMontgomeryTests
{
    [Fact]
    [Trait("Category", "Fast")]
    public void SquareDevice_montgomery64_matches_cpu_square_on_small_input()
    {
        // Small 64-bit modulus and tiny input to keep runtime trivial.
        var modulus = new GpuUInt128(0UL, 17UL);
        var primitiveRoot = new GpuUInt128(0UL, 3UL);

        Span<GpuUInt128> cpu = stackalloc GpuUInt128[8];
        for (ulong i = 0; i < (ulong)cpu.Length; i++)
        {
            cpu[(int)i] = new GpuUInt128(0UL, i + 1UL);
        }
        Span<GpuUInt128> gpu = stackalloc GpuUInt128[16];
        cpu.CopyTo(gpu);
        for (int i = cpu.Length; i < gpu.Length; i++)
        {
            gpu[i] = new GpuUInt128(0UL, 0UL);
        }

        // CPU reference
        NttGpuMath.Square(cpu, modulus, primitiveRoot);

        // Device execution
        using var context = GpuContextPool.Rent();
        var accelerator = context.Accelerator;
        using var buffer = accelerator.Allocate1D<GpuUInt128>(gpu.Length);
        buffer.View.CopyFromCPU(ref gpu[0], gpu.Length);
        NttGpuMath.SquareDevice(accelerator, buffer.View, modulus, primitiveRoot);
        buffer.View.CopyToCPU(ref gpu[0], gpu.Length);

        for (int i = 0; i < cpu.Length; i++)
        {
            gpu[i].Should().Be(cpu[i]);
        }
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void InverseDevice_montgomery64_matches_cpu_inverse_on_small_input()
    {
        var modulus = new GpuUInt128(0UL, 17UL);
        var primitiveRoot = new GpuUInt128(0UL, 3UL);

        Span<GpuUInt128> input = stackalloc GpuUInt128[8];
        for (ulong i = 0; i < (ulong)input.Length; i++)
        {
            input[(int)i] = new GpuUInt128(0UL, i + 1UL);
        }

        Span<GpuUInt128> transformed = stackalloc GpuUInt128[input.Length];
        input.CopyTo(transformed);
        NttGpuMath.Forward(transformed, modulus, primitiveRoot);

        Span<GpuUInt128> cpu = stackalloc GpuUInt128[input.Length];
        transformed.CopyTo(cpu);
        NttGpuMath.Inverse(cpu, modulus, primitiveRoot);

        var original = NttGpuMath.GpuTransformBackend;
        try
        {
            NttGpuMath.GpuTransformBackend = NttBackend.Staged;
            Span<GpuUInt128> gpu = stackalloc GpuUInt128[input.Length];
            transformed.CopyTo(gpu);
            NttGpuMath.InverseGpu(gpu, modulus, primitiveRoot);
            for (int i = 0; i < input.Length; i++)
            {
                gpu[i].Should().Be(cpu[i]);
            }
        }
        finally
        {
            NttGpuMath.GpuTransformBackend = original;
        }
    }
}
