using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

internal static class DivisorKernels
{
    public static void Kernel(Index1D _, ulong exponent, ReadOnlyGpuUInt128 divisor, ArrayView<byte> result)
    {
        if (divisor.High == 0UL && divisor.Low <= 1UL)
        {
            result[0] = 0;
            return;
        }

        GpuUInt128 baseVal;
        int x = divisor.High == 0UL
            ? 63 - XMath.LeadingZeroCount(divisor.Low)
            : 127 - XMath.LeadingZeroCount(divisor.High);
        ReadOnlyGpuUInt128 readOnlyMod = divisor;
        if (x == 0)
        {
            // TODO: Swap Pow2Mod for the ProcessEightBitWindows helper once Pow2Minus1Mod adopts it;
            // GpuPow2ModBenchmarks showed the windowed kernel cutting large divisors from ~51 µs to ~21 µs.
            baseVal = GpuUInt128.Pow2Mod(exponent, in readOnlyMod);
            result[0] = baseVal.High == 0UL && baseVal.Low == 1UL ? (byte)1 : (byte)0;
            return;
        }

        ulong ux = (ulong)x;
        if (x >= 64)
        {
            baseVal = new GpuUInt128(1UL << (x - 64), 0UL);
        }
        else
        {
            baseVal = new GpuUInt128(0UL, 1UL << x);
        }

        GpuUInt128 pow = baseVal;
        pow.ModPow(exponent / ux, in readOnlyMod);
        // TODO: Replace this trailing Pow2Mod call with the ProcessEightBitWindows variant once the shared
        // windowed helper lands so mixed-radix decompositions benefit from the same GPU speedup.
        var part2 = GpuUInt128.Pow2Mod(exponent % ux, in readOnlyMod);
        GpuUInt128 product = pow;
        ReadOnlyGpuUInt128 part2ReadOnly = part2.AsReadOnly();
        product.MulMod(in part2ReadOnly, in readOnlyMod);
        result[0] = product.High == 0UL && product.Low == 1UL ? (byte)1 : (byte)0;
    }
}
