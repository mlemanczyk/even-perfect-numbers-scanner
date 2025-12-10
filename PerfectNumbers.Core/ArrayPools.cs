using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core
{
	public static class Pools
	{
		private const int PoolCapacity = 10240; //PerfectNumberConstants.DefaultSmallPrimeFactorSlotCount << 8;

		[ThreadStatic]
		private static IntArrayPool? _intArrayPool;
		internal static IntArrayPool ExclusiveIntArray
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining)]
			get
			{
				return _intArrayPool ??= new(PoolCapacity);
			}
		}


		[ThreadStatic]
		private static UlongArrayPool? _ulongArrayPool;
		internal static UlongArrayPool ExclusiveUlongArray
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining)]
			get
			{
				return _ulongArrayPool ??= new(PoolCapacity);
			}
		}
	}

    public struct IntArrayPool(int poolCapacity)
	{
        private readonly int[][] _arrays = new int[poolCapacity][];
        private int _count = 0;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
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

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
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

    public struct UlongArrayPool(int poolCapacity)
	{
        private readonly ulong[][] _arrays = new ulong[poolCapacity][];
        private int _count = 0;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
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

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
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
