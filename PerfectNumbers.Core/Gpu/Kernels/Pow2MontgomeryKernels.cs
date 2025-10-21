using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core;

namespace PerfectNumbers.Core.Gpu;

internal static class Pow2MontgomeryKernels
{
    public static void Pow2MontgomeryKernelKeepMontgomery(
        Index1D index,
        ArrayView1D<ulong, Stride1D.Dense> exponents,
        MontgomeryDivisorData divisor,
        ArrayView1D<ulong, Stride1D.Dense> results)
    {
        results[index] = ULongExtensions.Pow2MontgomeryModWindowedGpu(divisor, exponents[index], keepMontgomery: true);
    }

    public static void Pow2MontgomeryKernelConvertToStandard(
        Index1D index,
        ArrayView1D<ulong, Stride1D.Dense> exponents,
        MontgomeryDivisorData divisor,
        ArrayView1D<ulong, Stride1D.Dense> results)
    {
        results[index] = ULongExtensions.Pow2MontgomeryModWindowedGpu(divisor, exponents[index], keepMontgomery: false);
    }
}
