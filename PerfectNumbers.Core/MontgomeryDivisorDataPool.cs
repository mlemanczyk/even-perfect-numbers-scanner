using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public static class MontgomeryDivisorDataPool
{
	[ThreadStatic]
	private static FixedCapacityStack<MontgomeryDivisorData>? _pool;
	public static FixedCapacityStack<MontgomeryDivisorData> Shared
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		get => _pool ??= new(PerfectNumberConstants.DefaultPoolCapacity);
	}

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
	public static MontgomeryDivisorData Rent(ulong modulus, ulong nPrime, ulong montgomeryOne, ulong montgomeryTwo, ulong montgomeryTwoSquared)
	{
		var pool = _pool ??= new(PerfectNumberConstants.DefaultPoolCapacity);
		if (pool.Pop() is { } pooled)
		{
			pooled.Modulus = modulus;
			pooled.NPrime = nPrime;
			pooled.MontgomeryOne = montgomeryOne;
			pooled.MontgomeryTwo = montgomeryTwo;
			pooled.MontgomeryTwoSquared = montgomeryTwoSquared;
			return pooled;
		}

		return new(modulus, nPrime, montgomeryOne, montgomeryTwo, montgomeryTwoSquared);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static void Return(MontgomeryDivisorData divisorData) => _pool!.Push(divisorData);


	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static MontgomeryDivisorData FromModulus(this FixedCapacityStack<MontgomeryDivisorData> queue, ulong modulus)
	{
		ulong nPrime = ComputeMontgomeryNPrime(modulus);
		ulong montgomeryOne = ComputeMontgomeryResidue(UInt128Numbers.OneShiftedLeft64, modulus);
		ulong montgomeryTwo = ComputeMontgomeryResidue(UInt128Numbers.OneShiftedLeft64x2, modulus);
		ulong montgomeryTwoSquared = ULongExtensions.MontgomeryMultiplyCpu(montgomeryTwo, montgomeryTwo, modulus, nPrime);

		return queue.Rent(
			modulus,
			nPrime,
			montgomeryOne,
			montgomeryTwo,
			montgomeryTwoSquared);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static MontgomeryDivisorData Rent(this FixedCapacityStack<MontgomeryDivisorData> queue, ulong modulus, ulong nPrime, ulong montgomeryOne, ulong montgomeryTwo, ulong montgomeryTwoSquared)
	{
		if (queue.Pop() is { } pooled)
		{
			pooled.Modulus = modulus;
			pooled.NPrime = nPrime;
			pooled.MontgomeryOne = montgomeryOne;
			pooled.MontgomeryTwo = montgomeryTwo;
			pooled.MontgomeryTwoSquared = montgomeryTwoSquared;
			return pooled;
		}

		return new(modulus, nPrime, montgomeryOne, montgomeryTwo, montgomeryTwoSquared);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static void Return(this FixedCapacityStack<MontgomeryDivisorData> queue, MontgomeryDivisorData divisorData) => queue.Push(divisorData);
}
