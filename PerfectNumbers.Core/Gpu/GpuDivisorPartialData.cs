using System.Runtime.CompilerServices;
using PerfectNumbers.Core;

namespace PerfectNumbers.Core.Gpu;

internal readonly struct GpuDivisorPartialData
{
    public readonly ulong Modulus;
    public readonly ReadOnlyGpuUInt128 ModulusWide;
    private readonly MontgomeryDivisorData _montgomeryData;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private GpuDivisorPartialData(ulong modulus, ReadOnlyGpuUInt128 modulusWide, in MontgomeryDivisorData montgomeryData)
    {
        Modulus = modulus;
        ModulusWide = modulusWide;
        _montgomeryData = montgomeryData;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuDivisorPartialData Create(ulong modulus)
    {
        MontgomeryDivisorData montgomery = MontgomeryDivisorData.FromModulus(modulus);
        return new GpuDivisorPartialData(modulus, new ReadOnlyGpuUInt128(modulus), in montgomery);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuDivisorPartialData FromMontgomery(in MontgomeryDivisorData divisor)
    {
        return new GpuDivisorPartialData(divisor.Modulus, new ReadOnlyGpuUInt128(divisor.Modulus), in divisor);
    }

    public MontgomeryDivisorData MontgomeryData
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _montgomeryData;
    }
}
