using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core.Gpu;

internal readonly struct GpuDivisorPartialData
{
    public readonly ulong Modulus;
    public readonly ReadOnlyGpuUInt128 ModulusWide;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuDivisorPartialData(ulong modulus)
    {
        Modulus = modulus;
        ModulusWide = new ReadOnlyGpuUInt128(modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuDivisorPartialData(ulong modulus, in ReadOnlyGpuUInt128 modulusWide)
    {
        Modulus = modulus;
        ModulusWide = modulusWide;
    }
}
