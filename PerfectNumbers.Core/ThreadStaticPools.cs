using System.Buffers;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core
{
    public static class ThreadStaticPools
    {
		[ThreadStatic]
		private static ArrayPool<FactorEntry>? _factorEntryPool;
		public static ArrayPool<FactorEntry> FactorEntryPool
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining)]
			get
			{
				return _factorEntryPool ??= ArrayPool<FactorEntry>.Create();
			}
		}

		[ThreadStatic]
		private static ArrayPool<FactorEntry128>? _factorEntry128Pool;
		public static ArrayPool<FactorEntry128> FactorEntry128Pool
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining)]
			get
			{
				return _factorEntry128Pool ??= ArrayPool<FactorEntry128>.Create();
			}
		}

		[ThreadStatic]
		private static ArrayPool<ulong>? _ulongPool;
		public static ArrayPool<ulong> UlongPool
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining)]
			get
			{
				return _ulongPool ??= ArrayPool<ulong>.Create();
			}
		}
	}
}