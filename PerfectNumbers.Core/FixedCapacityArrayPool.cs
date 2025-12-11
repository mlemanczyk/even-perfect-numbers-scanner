using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core
{
	public struct FixedCapacityArrayPool<T>(int poolCapacity)
	{
        private readonly T[][] _arrays = new T[poolCapacity][];
        private int _count = 0;

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public T[] Rent(int length)
        {
            int available = _count;
            if (available == 0)
            {
                return new T[length];
            }

            T[][] arrays = _arrays;
            available--;
            _count = available;

            T[] candidate = arrays[available];
			return candidate.Length >= length ? candidate : new T[length];
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Return(T[] array)
        {
            T[][] arrays = _arrays;
            int available = _count;

            if (available < arrays.Length)
            {
				arrays[available] = array;
				_count = available + 1;
            }
        }
    }
}
