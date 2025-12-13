using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

internal sealed class PartialFactorPendingEntry(ulong value, bool knownComposite)
{
	[ThreadStatic]
	private static PartialFactorPendingEntry? _poolHead;
	private PartialFactorPendingEntry? _next;

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static PartialFactorPendingEntry Rent(ulong value, bool knownComposite)
	{
		var poolHead = _poolHead;
		if (poolHead == null)
		{
			return new(value, knownComposite);
		}

		_poolHead = poolHead._next;
		poolHead.Value = value;
		poolHead.KnownComposite = knownComposite;
		poolHead.HasKnownPrimality = knownComposite;
		poolHead.IsPrime = false;
		return poolHead;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static void Return(PartialFactorPendingEntry entry)
	{
		entry._next = _poolHead;
		_poolHead = entry;
	}

	public ulong Value = value;
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



