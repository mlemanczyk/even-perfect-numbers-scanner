using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public readonly struct MontgomeryDivisorData(ulong modulus, ulong nPrime, ulong montgomeryOne, ulong montgomeryTwo, ulong montgomeryTwoSquared)
{
    public readonly ulong Modulus = modulus;

    public readonly ulong NPrime = nPrime;

    public readonly ulong MontgomeryOne = montgomeryOne;

    public readonly ulong MontgomeryTwo = montgomeryTwo;
    public readonly ulong MontgomeryTwoSquared = montgomeryTwoSquared;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong ComputeMontgomeryResidue(UInt128 value, ulong modulus) => (ulong)(value % modulus);

    private static ulong ComputeMontgomeryNPrime(ulong modulus)
    {
        ulong inv = modulus;
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        return unchecked(0UL - inv);
    }

    public static MontgomeryDivisorData FromModulus(ulong modulus)
    {
        ulong nPrime = ComputeMontgomeryNPrime(modulus);
        ulong montgomeryOne = ComputeMontgomeryResidue(UInt128Numbers.OneShiftedLeft64, modulus);
        ulong montgomeryTwo = ComputeMontgomeryResidue(UInt128Numbers.OneShiftedLeft64x2, modulus);
        ulong montgomeryTwoSquared = ULongExtensions.MontgomeryMultiply(montgomeryTwo, montgomeryTwo, modulus, nPrime);

        return new MontgomeryDivisorData(
            modulus,
            nPrime,
            montgomeryOne,
            montgomeryTwo,
            montgomeryTwoSquared);
    }

}

internal static class MontgomeryDivisorDataCache
{
    [ThreadStatic]
    private static bool s_hasCachedValue;

    [ThreadStatic]
    private static ulong s_cachedModulus;

    [ThreadStatic]
    private static MontgomeryDivisorData s_cachedData;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static MontgomeryDivisorData Get(ulong modulus)
    {
        if (s_hasCachedValue && s_cachedModulus == modulus)
        {
            return s_cachedData;
        }

        MontgomeryDivisorData computed = MontgomeryDivisorData.FromModulus(modulus);
        s_cachedModulus = modulus;
        s_cachedData = computed;
        s_hasCachedValue = true;
        return computed;
    }
}
