using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public static class MontgomeryDivisorDataPool
{
	[ThreadStatic]
	private static Queue<MontgomeryDivisorData>? _pool;
	public static Queue<MontgomeryDivisorData> Shared => _pool ??= new();

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
	
	public static MontgomeryDivisorData Rent(ulong modulus, ulong nPrime, ulong montgomeryOne, ulong montgomeryTwo, ulong montgomeryTwoSquared)
	{
		var pool = _pool ??= new();
		if (pool.TryDequeue(out var pooled))
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

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static void Return(MontgomeryDivisorData divisorData) => _pool!.Enqueue(divisorData);
	

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static MontgomeryDivisorData FromModulus(this Queue<MontgomeryDivisorData> queue, ulong modulus)
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

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static MontgomeryDivisorData Rent(this Queue<MontgomeryDivisorData> queue, ulong modulus, ulong nPrime, ulong montgomeryOne, ulong montgomeryTwo, ulong montgomeryTwoSquared)
	{
		if (queue.TryDequeue(out var pooled))
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

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static void Return(this Queue<MontgomeryDivisorData> queue, MontgomeryDivisorData divisorData) => queue.Enqueue(divisorData);
}
