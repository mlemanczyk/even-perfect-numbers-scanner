using System.Runtime.CompilerServices;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

[method: MethodImpl(MethodImplOptions.AggressiveInlining)]
internal struct MersenneDivisorResidueStepper(ulong prime, in GpuUInt128 step, in GpuUInt128 firstDivisor)
{
    private const ushort DecimalMaskWhenLastIsSeven = (1 << 3) | (1 << 7) | (1 << 9);
    private const ushort DecimalMaskOtherwise = (1 << 1) | (1 << 3) | (1 << 9);

    private readonly ushort decimalMask = (prime & 3UL) == 3UL ? DecimalMaskWhenLastIsSeven : DecimalMaskOtherwise;
    private readonly byte step10 = ComputeModulo(step, 10, multiplier: 6);
    private readonly byte step8 = (byte)(step.Low % 8UL);
    private readonly byte step5 = ComputeModulo(step, 5, multiplier: 1);
    private readonly byte step3 = ComputeModulo(step, 3, multiplier: 1);
    private readonly byte step7 = ComputeModulo(step, 7, multiplier: 2);
    private readonly byte step11 = ComputeModulo(step, 11, multiplier: 5);
    private readonly byte step13 = ComputeModulo(step, 13, multiplier: 3);
    private readonly byte step17 = ComputeModulo(step, 17, multiplier: 1);
    private readonly byte step19 = ComputeModulo(step, 19, multiplier: 17);

    private byte Remainder10 = ComputeModulo(firstDivisor, 10, multiplier: 6);
    private byte Remainder8 = (byte)(firstDivisor.Low % 8UL);
    private byte Remainder5 = ComputeModulo(firstDivisor, 5, multiplier: 1);
    private byte Remainder3 = ComputeModulo(firstDivisor, 3, multiplier: 1);
    private byte Remainder7 = ComputeModulo(firstDivisor, 7, multiplier: 2);
    private byte Remainder11 = ComputeModulo(firstDivisor, 11, multiplier: 5);
    private byte Remainder13 = ComputeModulo(firstDivisor, 13, multiplier: 3);
    private byte Remainder17 = ComputeModulo(firstDivisor, 17, multiplier: 1);
    private byte Remainder19 = ComputeModulo(firstDivisor, 19, multiplier: 17);

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

        if (Remainder3 == 0 || Remainder5 == 0 || Remainder7 == 0 || Remainder11 == 0 || Remainder13 == 0 || Remainder17 == 0 || Remainder19 == 0)
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
        Remainder13 = AddMod(Remainder13, step13, 13);
        Remainder17 = AddMod(Remainder17, step17, 17);
        Remainder19 = AddMod(Remainder19, step19, 19);
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

[method: MethodImpl(MethodImplOptions.AggressiveInlining)]
internal struct MersenneDivisorResidueStepperDescending(ulong prime, in GpuUInt128 step, in GpuUInt128 firstDivisor)
{
    private const ushort DecimalMaskWhenLastIsSeven = (1 << 3) | (1 << 7) | (1 << 9);
    private const ushort DecimalMaskOtherwise = (1 << 1) | (1 << 3) | (1 << 9);

    private readonly ushort decimalMask = (prime & 3UL) == 3UL ? DecimalMaskWhenLastIsSeven : DecimalMaskOtherwise;
    private readonly byte step10 = ComputeModulo(step, 10, multiplier: 6);
    private readonly byte step8 = (byte)(step.Low % 8UL);
    private readonly byte step5 = ComputeModulo(step, 5, multiplier: 1);
    private readonly byte step3 = ComputeModulo(step, 3, multiplier: 1);
    private readonly byte step7 = ComputeModulo(step, 7, multiplier: 2);
    private readonly byte step11 = ComputeModulo(step, 11, multiplier: 5);
    private readonly byte step13 = ComputeModulo(step, 13, multiplier: 3);
    private readonly byte step17 = ComputeModulo(step, 17, multiplier: 1);
    private readonly byte step19 = ComputeModulo(step, 19, multiplier: 17);

    private byte Remainder10 = ComputeModulo(firstDivisor, 10, multiplier: 6);
    private byte Remainder8 = (byte)(firstDivisor.Low % 8UL);
    private byte Remainder5 = ComputeModulo(firstDivisor, 5, multiplier: 1);
    private byte Remainder3 = ComputeModulo(firstDivisor, 3, multiplier: 1);
    private byte Remainder7 = ComputeModulo(firstDivisor, 7, multiplier: 2);
    private byte Remainder11 = ComputeModulo(firstDivisor, 11, multiplier: 5);
    private byte Remainder13 = ComputeModulo(firstDivisor, 13, multiplier: 3);
    private byte Remainder17 = ComputeModulo(firstDivisor, 17, multiplier: 1);
    private byte Remainder19 = ComputeModulo(firstDivisor, 19, multiplier: 17);

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

        if (Remainder3 == 0 || Remainder5 == 0 || Remainder7 == 0 || Remainder11 == 0 || Remainder13 == 0 || Remainder17 == 0 || Remainder19 == 0)
        {
            return false;
        }

        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Retreat()
    {
        Remainder10 = SubMod(Remainder10, step10, 10);
        Remainder8 = SubMod(Remainder8, step8, 8);
        Remainder5 = SubMod(Remainder5, step5, 5);
        Remainder3 = SubMod(Remainder3, step3, 3);
        Remainder7 = SubMod(Remainder7, step7, 7);
        Remainder11 = SubMod(Remainder11, step11, 11);
        Remainder13 = SubMod(Remainder13, step13, 13);
        Remainder17 = SubMod(Remainder17, step17, 17);
        Remainder19 = SubMod(Remainder19, step19, 19);
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
    private static byte SubMod(byte value, byte delta, byte modulus)
    {
        int diff = value - delta;
        if (diff < 0)
        {
            diff += modulus;
        }

        return (byte)diff;
    }
}
