using System.Reflection;
using FluentAssertions;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;
using System.Numerics;
using Xunit;

namespace PerfectNumbers.Core.Tests.Gpu;

// GPU-related tests are organized per-backend so both reference and staged
// implementations are exercised under the fast test suite.
[Collection("GpuNtt")]
public class NttGpuMathTests
{
    [Fact]
    [Trait("Category", "Fast")]
    public void ForwardGpu_reference_matches_cpu_forward()
    {
        var original = NttGpuMath.GpuTransformBackend;
        try
        {
            NttGpuMath.GpuTransformBackend = NttBackend.Reference;
            Span<GpuUInt128> cpu = stackalloc GpuUInt128[4];
            cpu[0] = new GpuUInt128(0UL, 1UL);
            cpu[1] = new GpuUInt128(0UL, 2UL);
            cpu[2] = new GpuUInt128(0UL, 3UL);
            cpu[3] = new GpuUInt128(0UL, 4UL);
            Span<GpuUInt128> gpu = stackalloc GpuUInt128[4];
            cpu.CopyTo(gpu);
            var modulus = new GpuUInt128(0UL, 17UL);
            var primitiveRoot = new GpuUInt128(0UL, 3UL);
            NttGpuMath.Forward(cpu, modulus, primitiveRoot);
            NttGpuMath.ForwardGpu(gpu, modulus, primitiveRoot);
            for (int i = 0; i < cpu.Length; i++)
            {
                gpu[i].Should().Be(cpu[i]);
            }
        }
        finally
        {
            NttGpuMath.GpuTransformBackend = original;
        }
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ForwardGpu_staged_matches_cpu_forward()
    {
        var original = NttGpuMath.GpuTransformBackend;
        try
        {
            NttGpuMath.GpuTransformBackend = NttBackend.Staged;
            Span<GpuUInt128> cpu = stackalloc GpuUInt128[4];
            cpu[0] = new GpuUInt128(0UL, 1UL);
            cpu[1] = new GpuUInt128(0UL, 2UL);
            cpu[2] = new GpuUInt128(0UL, 3UL);
            cpu[3] = new GpuUInt128(0UL, 4UL);
            Span<GpuUInt128> gpu = stackalloc GpuUInt128[4];
            cpu.CopyTo(gpu);
            var modulus = new GpuUInt128(0UL, 17UL);
            var primitiveRoot = new GpuUInt128(0UL, 3UL);
            NttGpuMath.Forward(cpu, modulus, primitiveRoot);
            NttGpuMath.ForwardGpu(gpu, modulus, primitiveRoot);
            for (int i = 0; i < cpu.Length; i++)
            {
                gpu[i].Should().Be(cpu[i]);
            }
        }
        finally
        {
            NttGpuMath.GpuTransformBackend = original;
        }
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void InverseGpu_reference_matches_cpu_inverse()
    {
        var original = NttGpuMath.GpuTransformBackend;
        try
        {
            NttGpuMath.GpuTransformBackend = NttBackend.Reference;
            Span<GpuUInt128> input = stackalloc GpuUInt128[4];
            input[0] = new GpuUInt128(0UL, 5UL);
            input[1] = new GpuUInt128(0UL, 7UL);
            input[2] = new GpuUInt128(0UL, 9UL);
            input[3] = new GpuUInt128(0UL, 11UL);
            var modulus = new GpuUInt128(0UL, 17UL);
            var primitiveRoot = new GpuUInt128(0UL, 3UL);
            Span<GpuUInt128> transformed = stackalloc GpuUInt128[4];
            input.CopyTo(transformed);
            NttGpuMath.Forward(transformed, modulus, primitiveRoot);
            Span<GpuUInt128> cpu = stackalloc GpuUInt128[4];
            transformed.CopyTo(cpu);
            NttGpuMath.Inverse(cpu, modulus, primitiveRoot);
            Span<GpuUInt128> gpu = stackalloc GpuUInt128[4];
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

    [Fact]
    [Trait("Category", "Fast")]
    public void InverseGpu_staged_matches_cpu_inverse()
    {
        var original = NttGpuMath.GpuTransformBackend;
        try
        {
            NttGpuMath.GpuTransformBackend = NttBackend.Staged;
            Span<GpuUInt128> input = stackalloc GpuUInt128[4];
            input[0] = new GpuUInt128(0UL, 5UL);
            input[1] = new GpuUInt128(0UL, 7UL);
            input[2] = new GpuUInt128(0UL, 9UL);
            input[3] = new GpuUInt128(0UL, 11UL);
            var modulus = new GpuUInt128(0UL, 17UL);
            var primitiveRoot = new GpuUInt128(0UL, 3UL);
            Span<GpuUInt128> transformed = stackalloc GpuUInt128[4];
            input.CopyTo(transformed);
            NttGpuMath.Forward(transformed, modulus, primitiveRoot);
            Span<GpuUInt128> cpu = stackalloc GpuUInt128[4];
            transformed.CopyTo(cpu);
            NttGpuMath.Inverse(cpu, modulus, primitiveRoot);
            Span<GpuUInt128> gpu = stackalloc GpuUInt128[4];
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

    [Fact]
    [Trait("Category", "Fast")]
    public void Montgomery_twiddle_inverses_multiply_to_r2()
    {
        var modulus = new GpuUInt128(0UL, 17UL);
        var primitiveRoot = new GpuUInt128(0UL, 3UL);

        GpuContextLease gpu = GpuContextPool.Rent();
        try
        {
            var accelerator = gpu.Accelerator;

            var method = typeof(NttGpuMath).GetMethod("GetSquareCache", BindingFlags.NonPublic | BindingFlags.Static);
            method.Should().NotBeNull();
            var cache = method!.Invoke(null, new object[] { accelerator, 8, modulus, primitiveRoot })!;
            var cacheType = cache.GetType();

            var twiddlesMont = (MemoryBuffer1D<GpuUInt128, Stride1D.Dense>)cacheType.GetProperty("TwiddlesMont")!.GetValue(cache)!;
            var twiddlesInvMont = (MemoryBuffer1D<GpuUInt128, Stride1D.Dense>)cacheType.GetProperty("TwiddlesInvMont")!.GetValue(cache)!;
            var r2 = (ulong)cacheType.GetProperty("MontR2Mod64")!.GetValue(cache)!;
            var mod = (ulong)cacheType.GetProperty("ModulusLow")!.GetValue(cache)!;

            int count = 7; // length 8 -> 7 twiddles
            var forward = new GpuUInt128[count];
            var inverse = new GpuUInt128[count];
            twiddlesMont.CopyToCPU(forward);
            twiddlesInvMont.CopyToCPU(inverse);

            for (int i = 0; i < count; i++)
            {
                ulong prod = (ulong)((((BigInteger)forward[i].Low) * inverse[i].Low) % mod);
                prod.Should().Be(r2);
            }
        }
        finally
        {
            gpu.Dispose();
        }
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ConvolveGpu_reference_matches_cpu_convolve()
    {
        var original = NttGpuMath.GpuTransformBackend;
        try
        {
            NttGpuMath.GpuTransformBackend = NttBackend.Reference;
            Span<GpuUInt128> leftCpu = stackalloc GpuUInt128[4];
            Span<GpuUInt128> rightCpu = stackalloc GpuUInt128[4];
            leftCpu[0] = new GpuUInt128(0UL, 1UL);
            leftCpu[1] = new GpuUInt128(0UL, 2UL);
            leftCpu[2] = new GpuUInt128(0UL, 3UL);
            leftCpu[3] = new GpuUInt128(0UL, 4UL);
            rightCpu[0] = new GpuUInt128(0UL, 5UL);
            rightCpu[1] = new GpuUInt128(0UL, 6UL);
            rightCpu[2] = new GpuUInt128(0UL, 7UL);
            rightCpu[3] = new GpuUInt128(0UL, 8UL);
            Span<GpuUInt128> leftGpu = stackalloc GpuUInt128[4];
            Span<GpuUInt128> rightGpu = stackalloc GpuUInt128[4];
            leftCpu.CopyTo(leftGpu);
            rightCpu.CopyTo(rightGpu);
            var modulus = new GpuUInt128(0UL, 17UL);
            var primitiveRoot = new GpuUInt128(0UL, 3UL);
            NttGpuMath.Convolve(leftCpu, rightCpu, modulus, primitiveRoot);
            NttGpuMath.ConvolveGpu(leftGpu, rightGpu, modulus, primitiveRoot);
            for (int i = 0; i < leftCpu.Length; i++)
            {
                leftGpu[i].Should().Be(leftCpu[i]);
            }
        }
        finally
        {
            NttGpuMath.GpuTransformBackend = original;
        }
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ConvolveGpu_staged_matches_cpu_convolve()
    {
        var original = NttGpuMath.GpuTransformBackend;
        try
        {
            NttGpuMath.GpuTransformBackend = NttBackend.Staged;
            Span<GpuUInt128> leftCpu = stackalloc GpuUInt128[4];
            Span<GpuUInt128> rightCpu = stackalloc GpuUInt128[4];
            leftCpu[0] = new GpuUInt128(0UL, 1UL);
            leftCpu[1] = new GpuUInt128(0UL, 2UL);
            leftCpu[2] = new GpuUInt128(0UL, 3UL);
            leftCpu[3] = new GpuUInt128(0UL, 4UL);
            rightCpu[0] = new GpuUInt128(0UL, 5UL);
            rightCpu[1] = new GpuUInt128(0UL, 6UL);
            rightCpu[2] = new GpuUInt128(0UL, 7UL);
            rightCpu[3] = new GpuUInt128(0UL, 8UL);
            Span<GpuUInt128> leftGpu = stackalloc GpuUInt128[4];
            Span<GpuUInt128> rightGpu = stackalloc GpuUInt128[4];
            leftCpu.CopyTo(leftGpu);
            rightCpu.CopyTo(rightGpu);
            var modulus = new GpuUInt128(0UL, 17UL);
            var primitiveRoot = new GpuUInt128(0UL, 3UL);
            NttGpuMath.Convolve(leftCpu, rightCpu, modulus, primitiveRoot);
            NttGpuMath.ConvolveGpu(leftGpu, rightGpu, modulus, primitiveRoot);
            for (int i = 0; i < leftCpu.Length; i++)
            {
                leftGpu[i].Should().Be(leftCpu[i]);
            }
        }
        finally
        {
            NttGpuMath.GpuTransformBackend = original;
        }
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void SquareGpu_reference_matches_cpu_square()
    {
        var original = NttGpuMath.GpuTransformBackend;
        try
        {
            NttGpuMath.GpuTransformBackend = NttBackend.Reference;
            Span<GpuUInt128> cpu = stackalloc GpuUInt128[4];
            cpu[0] = new GpuUInt128(0UL, 1UL);
            cpu[1] = new GpuUInt128(0UL, 2UL);
            cpu[2] = new GpuUInt128(0UL, 3UL);
            cpu[3] = new GpuUInt128(0UL, 4UL);
            Span<GpuUInt128> gpu = stackalloc GpuUInt128[4];
            cpu.CopyTo(gpu);
            var modulus = new GpuUInt128(0UL, 17UL);
            var primitiveRoot = new GpuUInt128(0UL, 3UL);
            NttGpuMath.Square(cpu, modulus, primitiveRoot);
            NttGpuMath.SquareGpu(gpu, modulus, primitiveRoot);
            for (int i = 0; i < cpu.Length; i++)
            {
                gpu[i].Should().Be(cpu[i]);
            }
        }
        finally
        {
            NttGpuMath.GpuTransformBackend = original;
        }
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void SquareGpu_staged_matches_cpu_square()
    {
        var original = NttGpuMath.GpuTransformBackend;
        try
        {
            NttGpuMath.GpuTransformBackend = NttBackend.Staged;
            Span<GpuUInt128> cpu = stackalloc GpuUInt128[4];
            cpu[0] = new GpuUInt128(0UL, 1UL);
            cpu[1] = new GpuUInt128(0UL, 2UL);
            cpu[2] = new GpuUInt128(0UL, 3UL);
            cpu[3] = new GpuUInt128(0UL, 4UL);
            Span<GpuUInt128> gpu = stackalloc GpuUInt128[4];
            cpu.CopyTo(gpu);
            var modulus = new GpuUInt128(0UL, 17UL);
            var primitiveRoot = new GpuUInt128(0UL, 3UL);
            NttGpuMath.Square(cpu, modulus, primitiveRoot);
            NttGpuMath.SquareGpu(gpu, modulus, primitiveRoot);
            for (int i = 0; i < cpu.Length; i++)
            {
                gpu[i].Should().Be(cpu[i]);
            }
        }
        finally
        {
            NttGpuMath.GpuTransformBackend = original;
        }
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void BitReverseGpu_matches_cpu()
    {
        GpuUInt128[] cpu = new GpuUInt128[8];
        for (ulong i = 0; i < (ulong)cpu.Length; i++)
        {
            cpu[i] = new GpuUInt128(0UL, i);
        }

        GpuUInt128[] gpu = (GpuUInt128[])cpu.Clone();
        NttGpuMath.BitReverse(cpu);
        NttGpuMath.BitReverseGpu(gpu);
        gpu.Should().Equal(cpu);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void DisposeAll_clears_ntt_caches()
    {
        // Ensure the cache starts empty.
        NttGpuMath.ClearCaches();

        Span<GpuUInt128> values = stackalloc GpuUInt128[2];
        values[0] = new GpuUInt128(0UL, 1UL);
        values[1] = new GpuUInt128(0UL, 2UL);
        var modulus = new GpuUInt128(0UL, 17UL);
        var primitiveRoot = new GpuUInt128(0UL, 3UL);
        GpuContextLease gpu = GpuContextPool.Rent();
        try
        {
            var buffer = gpu.Accelerator.Allocate1D<GpuUInt128>(values.Length);
            try
            {
                buffer.View.CopyFromCPU(ref values[0], values.Length);
                NttGpuMath.SquareDevice(gpu.Accelerator, buffer.View, modulus, primitiveRoot);
            }
            finally
            {
                buffer.Dispose();
            }
        }
        finally
        {
            gpu.Dispose();
        }

        // Verify that the square cache now holds entries for the accelerator.
        var field = typeof(NttGpuMath).GetField("SquareCache", BindingFlags.NonPublic | BindingFlags.Static)!;
        var cacheObj = field.GetValue(null)!;
        int count = (int)cacheObj.GetType().GetProperty("Count")!.GetValue(cacheObj)!;
        count.Should().BeGreaterThan(0);

        GpuContextPool.DisposeAll();

        count = (int)cacheObj.GetType().GetProperty("Count")!.GetValue(cacheObj)!;
        count.Should().Be(0);
    }
}

