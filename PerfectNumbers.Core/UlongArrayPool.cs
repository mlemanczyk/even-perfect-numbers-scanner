using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core
{
	public struct UlongArrayPool(int poolCapacity)
	{
        private readonly ulong[][] _arrays = new ulong[poolCapacity][];
        private int _count = 0;

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ulong[] Rent(int length)
        {
            int available = _count;
            if (available == 0)
            {
                return new ulong[length];
            }

            ulong[][] arrays = _arrays;
            available--;
            _count = available;

            ulong[] candidate = arrays[available];
			return candidate.Length >= length ? candidate : new ulong[length];
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Return(ulong[] array)
        {
            ulong[][] arrays = _arrays;
            int available = _count;

            if (available < arrays.Length)
            {
				arrays[available] = array;
				_count = available + 1;
            }
        }
    }
}
