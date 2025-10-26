using System.Runtime.CompilerServices;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

internal struct MersenneDivisorResidueStepper
{
    private const ushort DecimalMaskWhenLastIsSeven = (1 << 3) | (1 << 7) | (1 << 9);
    private const ushort DecimalMaskOtherwise = (1 << 1) | (1 << 3) | (1 << 9);

    private readonly ushort decimalMask;
    private readonly byte step10;
    private readonly byte step8;
    private readonly byte step5;
    private readonly byte step3;
    private readonly byte step7;
    private readonly byte step11;

    public byte Remainder10 { get; private set; }

    public byte Remainder8 { get; private set; }

    public byte Remainder5 { get; private set; }

    public byte Remainder3 { get; private set; }

    public byte Remainder7 { get; private set; }

    public byte Remainder11 { get; private set; }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public MersenneDivisorResidueStepper(ulong prime, in GpuUInt128 step, in GpuUInt128 firstDivisor)
    {
        decimalMask = (prime & 3UL) == 3UL ? DecimalMaskWhenLastIsSeven : DecimalMaskOtherwise;

        step10 = ComputeModulo(step, 10, multiplier: 6);
        step8 = (byte)(step.Low % 8UL);
        step5 = ComputeModulo(step, 5, multiplier: 1);
        step3 = ComputeModulo(step, 3, multiplier: 1);
        step7 = ComputeModulo(step, 7, multiplier: 2);
        step11 = ComputeModulo(step, 11, multiplier: 5);

        Remainder10 = ComputeModulo(firstDivisor, 10, multiplier: 6);
        Remainder8 = (byte)(firstDivisor.Low % 8UL);
        Remainder5 = ComputeModulo(firstDivisor, 5, multiplier: 1);
        Remainder3 = ComputeModulo(firstDivisor, 3, multiplier: 1);
        Remainder7 = ComputeModulo(firstDivisor, 7, multiplier: 2);
        Remainder11 = ComputeModulo(firstDivisor, 11, multiplier: 5);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly bool IsAdmissible()
    {
        if (((decimalMask >> Remainder10) & 1) == 0)
        {
            return false;
        }

        if (Remainder8 != 1 && Remainder8 != 7)
        {
            return false;
        }

        if (Remainder3 == 0 || Remainder5 == 0 || Remainder7 == 0 || Remainder11 == 0)
        {
            return false;
        }

        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Advance()
    {
        Remainder10 = AddMod(Remainder10, step10, 10);
        Remainder8 = AddMod(Remainder8, step8, 8);
        Remainder5 = AddMod(Remainder5, step5, 5);
        Remainder3 = AddMod(Remainder3, step3, 3);
        Remainder7 = AddMod(Remainder7, step7, 7);
        Remainder11 = AddMod(Remainder11, step11, 11);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte ComputeModulo(in GpuUInt128 value, byte modulus, byte multiplier)
    {
        ulong high = value.High % modulus;
        ulong low = value.Low % modulus;
        int result = (int)((high * multiplier) + low);
        result %= modulus;
        return (byte)result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte AddMod(byte value, byte delta, byte modulus)
    {
        int sum = value + delta;
        if (sum >= modulus)
        {
            sum -= modulus;
        }

        return (byte)sum;
    }
}
