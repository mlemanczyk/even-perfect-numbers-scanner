using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core;

namespace PerfectNumbers.Core.Gpu;

internal static class Pow2MontgomeryKernels
{
    public static void Pow2MontgomeryKernel(
        Index1D index,
        ArrayView1D<ulong, Stride1D.Dense> exponents,
        MontgomeryDivisorData divisor,
        byte keepMontgomery,
        ArrayView1D<ulong, Stride1D.Dense> results)
    {
        results[index] = ULongExtensions.Pow2MontgomeryModWindowedGpu(divisor, exponents[index], keepMontgomery != 0);
    }
}
