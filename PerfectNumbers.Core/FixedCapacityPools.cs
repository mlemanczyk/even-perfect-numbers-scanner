using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core
{
	public static class FixedCapacityPools<T>
	{
		[ThreadStatic]
		private static FixedCapacityArrayPool<T>? _arrayPool;
		public static FixedCapacityArrayPool<T> ExclusiveArray
		{

			[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
			get => _arrayPool ??= new(PerfectNumberConstants.DefaultPoolCapacity);
		}
	}

	public static class FixedCapacityPools
	{
		[ThreadStatic]
		private static IntArrayPool? _intArrayPool;
		public static IntArrayPool ExclusiveIntArray
		{

			[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
			get => _intArrayPool ??= new(PerfectNumberConstants.DefaultPoolCapacity);
		}

		[ThreadStatic]
		private static UlongArrayPool? _ulongArrayPool;
		public static UlongArrayPool ExclusiveUlongArray
		{

			[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
			get => _ulongArrayPool ??= new(PerfectNumberConstants.DefaultPoolCapacity);
		}
	}
}
