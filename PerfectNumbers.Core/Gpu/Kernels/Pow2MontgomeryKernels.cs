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
        results[index] = ULongExtensions.Pow2MontgomeryModWindowedGpuKeepMontgomery(divisor, exponents[index]);
    }

    public static void Pow2MontgomeryKernelConvertToStandard(
        Index1D index,
        ArrayView1D<ulong, Stride1D.Dense> exponents,
        MontgomeryDivisorData divisor,
        ArrayView1D<ulong, Stride1D.Dense> results)
    {
        results[index] = ULongExtensions.Pow2MontgomeryModWindowedGpuConvertToStandard(divisor, exponents[index]);
    }
}
