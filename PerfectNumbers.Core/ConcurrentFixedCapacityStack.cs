using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public sealed class ConcurrentFixedCapacityStack<T>(int capacity)
{
	private static readonly FixedCapacityArrayPool<T> _arrayPool = FixedCapacityPools<T>.ExclusiveArray;

	private readonly T[] _items = _arrayPool.Rent(capacity);
	private readonly int _capacity = capacity;

	private int _count = 0;
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
	public bool TryPop(out T? item)
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
				item = _items[next];
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

		Interlocked.Decrement(ref _count);
		throw new InvalidOperationException("Stack is full.");
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static void Return(in T[] items) => _arrayPool.Return(items);

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public T[] ToArray()
	{
		int count = Volatile.Read(ref _count);
		T[] result = _arrayPool.Rent(count);
		Array.Copy(_items, result, count);
		return result;
	}
}
