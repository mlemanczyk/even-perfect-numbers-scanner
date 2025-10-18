using ILGPU;
using ILGPU.Runtime;

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
        results[index] = ULongExtensions.Pow2MontgomeryModWindowedKernel(exponents[index], divisor, keepMontgomery != 0);
    }
}
