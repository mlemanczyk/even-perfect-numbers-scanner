using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core.Gpu;

internal readonly struct GpuDivisorPartialData
{
    public readonly ulong Modulus;
    public readonly ReadOnlyGpuUInt128 ModulusWide;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private GpuDivisorPartialData(ulong modulus, in ReadOnlyGpuUInt128 modulusWide)
    {
        Modulus = modulus;
        ModulusWide = modulusWide;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuDivisorPartialData Create(ulong modulus)
    {
        ReadOnlyGpuUInt128 modulusWide = new(modulus);
        return new GpuDivisorPartialData(modulus, in modulusWide);
    }
}
