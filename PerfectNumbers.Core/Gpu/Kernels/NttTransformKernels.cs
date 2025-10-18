using ILGPU;

namespace PerfectNumbers.Core.Gpu;

internal static class NttTransformKernels
{
    public static void ForwardKernel(Index1D index, ArrayView<GpuUInt128> input, ArrayView<GpuUInt128> output, int length, GpuUInt128 modulus, GpuUInt128 root)
    {
        // TODO(NTT-OPT): Replace this reference O(n^2) kernel with a stage-wise
        // Cooley–Tukey butterfly kernel (O(n log n)). This version uses a per-
        // element ModPow in the hot loop which is far too slow and causes long
        // single-kernel runtimes. After twiddle precomputation, each butterfly
        // should perform: (u, v) -> (u+v*w, u-v*w) with a single MulMod.
        var sum = new GpuUInt128(0UL, 0UL);
        for (int i = 0; i < length; i++)
        {
            var term = input[i];
            var power = new GpuUInt128(root);
            power.ModPow((ulong)index.X * (ulong)i, modulus);
            term.MulMod(power, modulus);
            sum.AddMod(term, modulus);
        }

        output[index] = sum;
    }

    public static void InverseKernel(Index1D index, ArrayView<GpuUInt128> input, ArrayView<GpuUInt128> output, int length, GpuUInt128 modulus, GpuUInt128 rootInv, GpuUInt128 nInv)
    {
        // TODO(NTT-OPT): Replace this O(n^2) inverse with stage-wise butterflies
        // using precomputed inverse twiddles. Apply final scaling by nInv after
        // all stages, preferably in a small dedicated kernel.
        var sum = new GpuUInt128(0UL, 0UL);
        for (int i = 0; i < length; i++)
        {
            var term = input[i];
            var power = new GpuUInt128(rootInv);
            power.ModPow((ulong)index.X * (ulong)i, modulus);
            term.MulMod(power, modulus);
            sum.AddMod(term, modulus);
        }

        sum.MulMod(nInv, modulus);
        output[index] = sum;
    }

    public static void BitReverseKernel(Index1D index, ArrayView<GpuUInt128> data, int bits)
    {
        int i = index.X;
        int j = ReverseBits(i, bits);
        if (i < j)
        {
            var tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }

    internal static int ReverseBits(int value, int bits)
    {
        int result = 0;
        for (int i = 0; i < bits; i++)
        {
            result = (result << 1) | (value & 1);
            value >>= 1;
        }

        return result;
    }
}
