using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public readonly struct MontgomeryDivisorData(ulong modulus, ulong nPrime, ulong montgomeryOne, ulong montgomeryTwo)
{
    public ulong Modulus { get; } = modulus;

    public ulong NPrime { get; } = nPrime;

    public ulong MontgomeryOne { get; } = montgomeryOne;

    public ulong MontgomeryTwo { get; } = montgomeryTwo;
}

internal static class MontgomeryDivisorDataCache
{
    private static readonly ConcurrentDictionary<ulong, MontgomeryDivisorData> Cache = new();

    public static MontgomeryDivisorData Get(ulong modulus)
    {
        if (modulus <= 1UL || (modulus & 1UL) == 0UL)
        {
            return new MontgomeryDivisorData(modulus, 0UL, 0UL, 0UL);
        }

        return Cache.GetOrAdd(modulus, static m => Create(m));
    }

    private static MontgomeryDivisorData Create(ulong modulus)
    {
        return new MontgomeryDivisorData(
            modulus,
            ComputeMontgomeryNPrime(modulus),
            ComputeMontgomeryResidue(1UL, modulus),
            ComputeMontgomeryResidue(2UL, modulus));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong ComputeMontgomeryResidue(ulong value, ulong modulus) => (ulong)((UInt128)value * (UInt128.One << 64) % modulus);

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
}
