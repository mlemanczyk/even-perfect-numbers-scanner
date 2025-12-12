using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public sealed class ConcurrentFixedCapacityStack<T>
{
	[ThreadStatic]
	private static FixedCapacityArrayPool<T>? _arrayPool;

	private readonly T[] _items;
	private readonly int _capacity;

	private int _count = 0;

	public ConcurrentFixedCapacityStack(int capacity)
	{
		_arrayPool ??= FixedCapacityPools<T>.ExclusiveArray;
		_items = _arrayPool.Value.Rent(capacity);
		_capacity = capacity;
	}

	public int Count
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		get => Volatile.Read(ref _count);
	}

	public int Capacity
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		get => _capacity;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public void Clear() => Interlocked.Exchange(ref _count, 0);

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public T? Pop() => TryPop(out T? item) ? item : default;

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	/// <summary>
	/// Tries popping the first element from the stack. It doesn't clear the internal memory buffer from unused references.
	/// </summary>
	/// <returns>Returns false when the stack is empty. Returns true otherwise. If the stack was not empty, item contains the first element from the stack.
	public bool TryPop([NotNullWhen(true)] out T? item)
	{
		while (true)
		{
			int observed = Volatile.Read(ref _count);
			if (observed == 0)
			{
				item = default;
				return false;
			}

			int next = observed - 1;
			if (Interlocked.CompareExchange(ref _count, next, observed) == observed)
			{
				item = _items[next]!;
				return true;
			}
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public void Push(T item)
	{
		int next = Interlocked.Increment(ref _count);
		if (next <= _capacity)
		{
			_items[next - 1] = item;
			return;
		}

		ArgumentOutOfRangeException.ThrowIfGreaterThan(next, _capacity);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static void Return(in T[] items) => _arrayPool!.Value.Return(items);

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public T[] ToArray()
	{
		int count = Volatile.Read(ref _count);
		T[] result = _arrayPool!.Value.Rent(count);
		Array.Copy(_items, result, count);
		return result;
	}
}
