using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public sealed class FixedCapacityStack<T>
{
	[ThreadStatic]
	private static FixedCapacityArrayPool<T>? _arrayPool;

	public int Count = 0;
	private T[] _items;

	private int _capacity;

	public FixedCapacityStack(int capacity)
	{
		_arrayPool ??= new(PerfectNumberConstants.DefaultPoolCapacity);
		_items = _arrayPool.Value.Rent(capacity);
		_capacity = capacity;
	}

	public int Capacity
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		get => _capacity;

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		set
		{
			if (value < _capacity)
			{
				_arrayPool!.Value.Return(_items);
				_items = _arrayPool.Value.Rent(value);
				_capacity = value;
			}
		}
	}

	/// <summary>
	/// Sets the no. of elements on the stack to 0. It doesn't clear the internal memory buffer from unused references.
	/// </summary>
	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public void Clear() => Count = 0;

	/// <summary>
	/// Pops and returns the first element from the stack. It doesn't clear the internal memory buffer from unused references.
	/// </summary>
	/// <returns>The first element on the stack or null / default when the stack is empty.
	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public T? Pop() => Count > 0 ? _items[Count--] : default;

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public void Push(T item)
	{
		int newCount = Count + 1;
		T[] items = _items;
		if (newCount < items.Length)
		{
			items[newCount] = item;
			Count = newCount;
		}
	}
}
