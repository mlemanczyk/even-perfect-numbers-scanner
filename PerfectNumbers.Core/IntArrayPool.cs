using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core
{
	public struct IntArrayPool(int poolCapacity)
	{
        private readonly int[][] _arrays = new int[poolCapacity][];
        private int _count = 0;

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public int[] Rent(int length)
        {
            int available = _count;
            if (available == 0)
            {
                return new int[length];
            }

            int[][] arrays = _arrays;
            available--;
            _count = available;

            int[] candidate = arrays[available];
			return candidate.Length >= length ? candidate : new int[length];
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Return(int[] array)
        {
            int[][] arrays = _arrays;
            int available = _count;

            if (available < arrays.Length)
            {
				arrays[available] = array;
				_count = available + 1;
            }
        }
    }
}
