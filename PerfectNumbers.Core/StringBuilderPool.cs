using System.Runtime.CompilerServices;
using System.Text;

namespace PerfectNumbers.Core;

public static class StringBuilderPool
{
	private static readonly ConcurrentFixedCapacityStack<StringBuilder> _stringBuilderPool = new(PerfectNumberConstants.DefaultPoolCapacity);

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static StringBuilder Rent() => _stringBuilderPool.Pop() is { } sb ? sb : new(PerfectNumberConstants.DefaultStringBuilderCapacity);

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static void Return(StringBuilder sb)
	{
		_ = sb.Clear();
		// TODO: Preserve the builder's capacity when returning it so the pool hands the same buffer back without shrinkage.
		_stringBuilderPool.Push(sb);
	}
}

