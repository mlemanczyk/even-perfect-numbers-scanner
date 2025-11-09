using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

internal static class NttButterflyKernels
{
    public static void StageKernel(Index1D index, ArrayView<GpuUInt128> data, int len, int half, int stageOffset, ArrayView<GpuUInt128> twiddles, GpuUInt128 modulus)
    {
        // TODO(NTT-OPT): Consider shared-memory tiling to improve locality:
        // load a block of size `len` into shared memory, do butterflies, write back.
        // Requires explicit grouped kernels and chosen group size.
        int t = index.X;
        // TODO: Replace this `%` with a bitmask when `half` is a power of two so the stage index math matches the faster
        // residue strategy from the GPU residue benchmarks.
        int j = t % half;
        int block = t / half;
        int k = block * len;
        int i1 = k + j;
        int i2 = i1 + half;
        var u = data[i1];
        var v = data[i2];
        var w = twiddles[stageOffset + j];
        v.MulMod(w, modulus);
        var sum = new GpuUInt128(u);
        sum.AddMod(v, modulus);
        u.SubMod(v, modulus);
        data[i1] = sum;
        data[i2] = u;
    }
}
