using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public readonly struct MontgomeryDivisorData(ulong modulus, ulong nPrime, ulong montgomeryOne, ulong montgomeryTwo, ulong montgomeryTwoSquared)
{
	public static readonly MontgomeryDivisorData Empty = new();

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static MontgomeryDivisorData FromModulus(ulong modulus)
	{
		ulong nPrime = ComputeMontgomeryNPrime(modulus);
		ulong montgomeryOne = ComputeMontgomeryResidue(UInt128Numbers.OneShiftedLeft64, modulus);
		ulong montgomeryTwo = ComputeMontgomeryResidue(UInt128Numbers.OneShiftedLeft64x2, modulus);
		ulong montgomeryTwoSquared = ULongExtensions.MontgomeryMultiplyCpu(montgomeryTwo, montgomeryTwo, modulus, nPrime);

		return new(
			modulus,
			nPrime,
			montgomeryOne,
			montgomeryTwo,
			montgomeryTwoSquared);
	}

    public readonly ulong Modulus = modulus;
	public readonly ulong NPrime = nPrime;
	public readonly ulong MontgomeryOne = montgomeryOne;
	public readonly ulong MontgomeryTwo = montgomeryTwo;
	public readonly ulong MontgomeryTwoSquared = montgomeryTwoSquared;
}
