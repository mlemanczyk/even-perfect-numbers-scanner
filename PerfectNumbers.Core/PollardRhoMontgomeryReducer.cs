using System;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

internal readonly struct PollardRhoMontgomeryReducer
{
    private readonly MontgomeryDivisorData _divisorData;
    private readonly ulong _reciprocal;

    [ThreadStatic]
    private static bool s_hasCachedReciprocal;

    [ThreadStatic]
    private static ulong s_cachedReciprocal;

    public PollardRhoMontgomeryReducer(in MontgomeryDivisorData divisorData, ulong reciprocal)
    {
        _divisorData = divisorData;
        _reciprocal = reciprocal == 0UL || reciprocal == ulong.MaxValue
            ? ReciprocalMath.ComputeReciprocalEstimate(divisorData.Modulus)
            : reciprocal;
    }

    public ulong Modulus => _divisorData.Modulus;
    private ulong NPrime => _divisorData.NPrime;
    private ulong MontgomeryTwoSquared => _divisorData.MontgomeryTwoSquared;
    private ulong MontgomeryOne => _divisorData.MontgomeryOne;
    private ulong MontgomeryTwo => _divisorData.MontgomeryTwo;
    public ulong Reciprocal => _reciprocal;

    public static PollardRhoMontgomeryReducer Create(ulong modulus)
    {
        MontgomeryDivisorData divisorData = MontgomeryDivisorDataCache.Get(modulus);
        ulong reciprocal;
        if (s_hasCachedReciprocal)
        {
            reciprocal = ReciprocalMath.RefineReciprocalEstimate(s_cachedReciprocal, modulus);
            if (reciprocal == 0UL || reciprocal == ulong.MaxValue)
            {
                reciprocal = ReciprocalMath.ComputeReciprocalEstimate(modulus);
            }
        }
        else
        {
            reciprocal = ReciprocalMath.ComputeReciprocalEstimate(modulus);
        }

        s_cachedReciprocal = reciprocal;
        s_hasCachedReciprocal = true;
        return new PollardRhoMontgomeryReducer(divisorData, reciprocal);
    }

    public static PollardRhoMontgomeryReducer Create(in MontgomeryDivisorData divisorData, ulong previousReciprocal)
    {
        ulong reciprocal = ReciprocalMath.RefineReciprocalEstimate(previousReciprocal, divisorData.Modulus);
        if (reciprocal == 0UL || reciprocal == ulong.MaxValue)
        {
            reciprocal = ReciprocalMath.ComputeReciprocalEstimate(divisorData.Modulus);
        }

        s_cachedReciprocal = reciprocal;
        s_hasCachedReciprocal = true;
        return new PollardRhoMontgomeryReducer(divisorData, reciprocal);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ulong ConvertToMontgomery(ulong value)
    {
        if (value == 0UL)
        {
            return 0UL;
        }

        if (value == 1UL)
        {
            return MontgomeryOne;
        }

        if (value == 2UL)
        {
            return MontgomeryTwo;
        }

        return value.MontgomeryMultiply(MontgomeryTwoSquared, Modulus, NPrime);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ulong ReduceFromMontgomery(ulong montgomeryValue)
    {
        return montgomeryValue.MontgomeryMultiply(1UL, Modulus, NPrime);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ulong Advance(ulong montgomeryValue, ulong cMontgomery)
    {
        ulong squared = montgomeryValue.MontgomeryMultiply(montgomeryValue, Modulus, NPrime);
        return MontgomeryAdd(squared, cMontgomery, Modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ulong AdvanceStandard(ulong standardValue, ulong c)
    {
        UInt128 polynomial = (UInt128)standardValue * standardValue + c;
        return ReciprocalMath.Reduce(polynomial, Modulus, _reciprocal);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MontgomeryAdd(ulong a, ulong b, ulong modulus)
    {
        UInt128 total = (UInt128)a + b;
        if (total >= modulus)
        {
            total -= modulus;
        }

        return (ulong)total;
    }
}
