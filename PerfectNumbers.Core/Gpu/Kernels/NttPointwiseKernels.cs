using ILGPU;

namespace PerfectNumbers.Core.Gpu;

internal static class NttPointwiseKernels
{
    public static void MulKernel(Index1D index, ArrayView<GpuUInt128> a, ArrayView<GpuUInt128> b, GpuUInt128 modulus)
    {
        var val = a[index];
        val.MulMod(b[index], modulus);
        a[index] = val;
    }

    public static void ScaleKernel(Index1D index, ArrayView<GpuUInt128> data, GpuUInt128 scale, GpuUInt128 modulus)
    {
        var v = data[index];
        v.MulMod(scale, modulus);
        data[index] = v;
    }
}
