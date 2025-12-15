using System.Numerics;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core
{
    public struct UlongArrayPool(int poolCapacity)
    {
        private const int MinBucketPower = 1;
        private static readonly int MaxBucketPower = CalculateMaxBucketPower();
        private static readonly int MaxBucketLength = 1 << MaxBucketPower;
        private static readonly int BucketCount = MaxBucketPower - MinBucketPower + 1;

        private readonly ulong[][] _arrays = new ulong[poolCapacity * BucketCount][];
        private readonly int[] _counts = new int[BucketCount];
        private readonly int _bucketCapacity = poolCapacity;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public readonly ulong[] Rent(int length)
        {
            int bucketIndex = GetBucketIndex(length);
            if (bucketIndex < 0)
            {
                return new ulong[length];
            }

            int[] counts = _counts;
            int available = counts[bucketIndex];
            int bucketLength = GetBucketLength(bucketIndex);

            if (available == 0)
            {
                return new ulong[bucketLength];
            }

            int offset = (bucketIndex * _bucketCapacity) + (available - 1);
            counts[bucketIndex] = available - 1;
            return _arrays[offset];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public readonly void Return(ulong[] array)
        {
			int capacity = array.Length;
			int bucketIndex = GetBucketIndex(capacity);
            if (bucketIndex < 0)
            {
                return;
            }

            int bucketLength = GetBucketLength(bucketIndex);
            if (capacity < bucketLength)
            {
                return;
            }

            int[] counts = _counts;
            int available = counts[bucketIndex];
			int bucketCapacity = _bucketCapacity;
			if (available >= bucketCapacity)
            {
                return;
            }

            int offset = (bucketIndex * bucketCapacity) + available;
            _arrays[offset] = array;
            counts[bucketIndex] = available + 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static int GetBucketIndex(int length)
        {
            if (length <= 0 || length > MaxBucketLength)
            {
                return -1;
            }

			if (length <= (1 << MinBucketPower))
            {
                return 0;
            }

            int power = 32 - BitOperations.LeadingZeroCount((uint)(length - 1));
            if (power > MaxBucketPower)
            {
                return -1;
            }

            return power - MinBucketPower;
        }

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		private static int GetBucketLength(int bucketIndex) => 1 << (bucketIndex + MinBucketPower);

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		private static int CalculateMaxBucketPower()
        {
            int maxLength = Math.Min(int.MaxValue, Array.MaxLength);
            int leadingZeros = BitOperations.LeadingZeroCount((uint)maxLength);
            return 31 - leadingZeros;
        }
    }
}
