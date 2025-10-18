using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

internal static class OrderKernels
{
    public static void OrderKernelScan(
        Index1D index,
        ulong exponent,
        ulong divMul,
        ArrayView<GpuUInt128> qs,
        ArrayView<ulong> orders)
    {
        GpuUInt128 q = qs[index];
        ReadOnlyGpuUInt128 readOnlyQ = q.AsReadOnly();
        GpuUInt128 phi = q - GpuUInt128.One;
        if (phi.High != 0UL)
        {
            orders[index] = 0UL;
            return;
        }

        ulong phi64 = phi.Low;
        // TODO: Once the ProcessEightBitWindows helper is available, switch this order kernel to that faster
        // Pow2Mod variant so cycle checks inherit the same gains observed in GpuPow2ModBenchmarks.
        GpuUInt128 pow = GpuUInt128.Pow2Mod(phi64, in readOnlyQ);
        if (pow != GpuUInt128.One)
        {
            orders[index] = 0UL;
            return;
        }

        GpuUInt128 halfPow = GpuUInt128.Pow2Mod(phi64 >> 1, in readOnlyQ) - GpuUInt128.One;
        if (GpuUInt128.BinaryGcd(halfPow, q) != GpuUInt128.One)
        {
            orders[index] = 0UL;
            return;
        }

        ulong div = KernelMathHelpers.FastDiv64Gpu(phi64, exponent, divMul);
        GpuUInt128 divPow = GpuUInt128.Pow2Mod(div, in readOnlyQ) - GpuUInt128.One;
        if (GpuUInt128.BinaryGcd(divPow, q) != GpuUInt128.One)
        {
            orders[index] = 0UL;
            return;
        }

        orders[index] = exponent;
    }
}
