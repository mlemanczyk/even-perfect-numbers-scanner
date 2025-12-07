using System.Numerics;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

/// <summary>
/// Tracks successive residues of <c>baseValue^exponent mod modulus</c> for ascending exponents using <see cref="BigInteger"/> inputs.
/// This avoids recomputing full modular exponentiation when stepping forward by a known delta.
/// </summary>
public struct BigIntegerExponentStepper(BigInteger modulus, BigInteger baseValue)
{
    private readonly BigInteger _modulus = modulus;
    private readonly BigInteger _baseValue = baseValue % modulus;
    private BigInteger PreviousExponent = BigInteger.Zero;
    private BigInteger CurrentResidue = BigInteger.One % modulus;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
    public BigInteger Initialize(BigInteger exponent)
    {
        CurrentResidue = BigInteger.ModPow(_baseValue, exponent, _modulus);
        PreviousExponent = exponent;
        return CurrentResidue;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool InitializeIsUnity(BigInteger exponent)
    {
        Initialize(exponent);
        return CurrentResidue.IsOne;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public BigInteger ComputeNext(BigInteger exponent)
    {
        if (exponent == PreviousExponent)
        {
            return CurrentResidue;
        }

        BigInteger delta = exponent - PreviousExponent;
        BigInteger deltaResidue = BigInteger.ModPow(_baseValue, delta, _modulus);
        CurrentResidue = (CurrentResidue * deltaResidue) % _modulus;
        PreviousExponent = exponent;
        return CurrentResidue;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool ComputeNextIsUnity(BigInteger exponent)
    {
        ComputeNext(exponent);
        return CurrentResidue.IsOne;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Reset()
    {
        PreviousExponent = BigInteger.Zero;
        CurrentResidue = BigInteger.One % _modulus;
    }
}
