using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public struct PartialFactorPendingEntry(ulong value, bool knownComposite)
{
	public readonly ulong Value = value;
	public bool KnownComposite = knownComposite;
	public bool HasKnownPrimality = knownComposite;
	public bool IsPrime = false;

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public void WithPrimality(bool isPrime)
	{
		KnownComposite = KnownComposite || !isPrime;
		HasKnownPrimality = true;
		IsPrime = isPrime;
	}
}



