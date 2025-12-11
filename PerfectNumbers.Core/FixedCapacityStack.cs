using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public sealed class FixedCapacityStack<T>(int capacity)
{
	private static readonly FixedCapacityArrayPool<T> _arrayPool = FixedCapacityPools<T>.ExclusiveArray;

	public int Count = 0;
	private T[] _items = _arrayPool.Rent(capacity);

	private int _capacity = capacity;
	public int Capacity
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		get => _capacity;

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		set
		{
			if (value < _capacity)
			{
				_arrayPool.Return(_items);
				_items = _arrayPool.Rent(value);
				_capacity = value;
			}
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public void Clear() => Count = 0;

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public T? Pop() => Count > 0 ? _items[Count--] : default;

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public void Push(T item)
	{
		int newCount = Count + 1;
		if (newCount < _items.Length)
		{
			_items[newCount] = item;
			Count = newCount;
		}
	}
}
