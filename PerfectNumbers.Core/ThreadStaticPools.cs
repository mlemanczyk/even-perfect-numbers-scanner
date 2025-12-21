using System.Buffers;
using System.Runtime.CompilerServices;
using PerfectNumbers.Core.Cpu;
using PerfectNumbers.Core.Gpu;
using PeterO.Numbers;

namespace PerfectNumbers.Core
{
	public static class ThreadStaticPools
	{
		[ThreadStatic]
		private static ArrayPool<EInteger>? _eintegerPool;

		public static ArrayPool<EInteger> EIntegerPool
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
			get => _eintegerPool ??= ArrayPool<EInteger>.Create();
		}

		[ThreadStatic]
		private static FixedCapacityStack<List<KeyValuePair<UInt128, int>>>? _keyValuePairUInt128IntegerPool;

		public static FixedCapacityStack<List<KeyValuePair<UInt128, int>>> KeyValuePairUInt128IntegerPool
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
			get => _keyValuePairUInt128IntegerPool ??= new(PerfectNumberConstants.DefaultPoolCapacity);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public static List<KeyValuePair<UInt128, int>> Rent(this FixedCapacityStack<List<KeyValuePair<UInt128, int>>> pool, int capacity)
		{
			if (pool.Pop() is { } pooled)
			{
				if (capacity > pooled.Capacity)
				{
					pooled.Capacity = capacity;
				}

				pooled.Clear();
				return pooled;
			}

			return new(capacity);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public static void Return(this FixedCapacityStack<List<KeyValuePair<UInt128, int>>> pool, List<KeyValuePair<UInt128, int>> list) => pool.Push(list);

		[ThreadStatic]
		private static FixedCapacityStack<List<UInt128>>? _uint128ListPool;

		public static FixedCapacityStack<List<UInt128>> UInt128ListPool
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
			get => _uint128ListPool ??= new(PerfectNumberConstants.DefaultPoolCapacity);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public static List<UInt128> Rent(this FixedCapacityStack<List<UInt128>> pool, int capacity)
		{
			if (pool.Pop() is { } pooled)
			{
				if (capacity > pooled.Capacity)
				{
					pooled.Capacity = capacity;
				}

				pooled.Clear();
				return pooled;
			}

			return new(capacity);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public static void Return(this FixedCapacityStack<List<UInt128>> pool, List<UInt128> list) => pool.Push(list);

		[ThreadStatic]
		private static FixedCapacityStack<Dictionary<UInt128, int>>? _uint128IntDictionaryPool;

		public static FixedCapacityStack<Dictionary<UInt128, int>> UInt128IntDictionaryPool
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
			get => _uint128IntDictionaryPool ??= new(PerfectNumberConstants.DefaultPoolCapacity);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public static Dictionary<UInt128, int> Rent(this FixedCapacityStack<Dictionary<UInt128, int>> pool, int capacity)
		{
			if (pool.Pop() is { } pooled)
			{
				pooled.EnsureCapacity(capacity);
				pooled.Clear();
				return pooled;
			}

			return new(capacity);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public static void Return(this FixedCapacityStack<Dictionary<UInt128, int>> pool, Dictionary<UInt128, int> dictionary) => pool.Push(dictionary);

		[ThreadStatic]
		private static FixedCapacityStack<Dictionary<ulong, int>>? _ulongIntDictionaryPool;

		public static FixedCapacityStack<Dictionary<ulong, int>> UlongIntDictionaryPool
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
			get => _ulongIntDictionaryPool ??= new(PerfectNumberConstants.DefaultPoolCapacity);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public static Dictionary<ulong, int> Rent(this FixedCapacityStack<Dictionary<ulong, int>> pool, int capacity)
		{
			if (pool.Pop() is { } pooled)
			{
				pooled.EnsureCapacity(capacity);
				pooled.Clear();
				return pooled;
			}

			return new(capacity);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public static void Return(this FixedCapacityStack<Dictionary<ulong, int>> pool, Dictionary<ulong, int> dictionary) => pool.Push(dictionary);

		[ThreadStatic]
		private static ArrayPool<FactorEntry>? _factorEntryPool;

		public static ArrayPool<FactorEntry> FactorEntryPool
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
			get => _factorEntryPool ??= ArrayPool<FactorEntry>.Create();
		}

		[ThreadStatic]
		private static ArrayPool<FactorEntry128>? _factorEntry128Pool;

		public static ArrayPool<FactorEntry128> FactorEntry128Pool
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
			get => _factorEntry128Pool ??= ArrayPool<FactorEntry128>.Create();
		}

		[ThreadStatic]
		private static ArrayPool<GpuUInt128>? _gpuUInt128Pool;

		public static ArrayPool<GpuUInt128> GpuUInt128Pool
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
			get => _gpuUInt128Pool ??= ArrayPool<GpuUInt128>.Create();
		}

		// [ThreadStatic]
		// private static MersenneCpuDivisorScanSession? _mersenneCpuDivisorSession;

		// [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		// internal static MersenneCpuDivisorScanSession RentMersenneCpuDivisorSession(Gpu.Accelerators.PrimeOrderCalculatorAccelerator gpu, ComputationDevice orderDevice)
		// {
		// 	if (_mersenneCpuDivisorSession is { } session)
		// 	{
		// 		session.Configure(gpu, orderDevice);
		// 		_mersenneCpuDivisorSession = null;
		// 		return session;
		// 	}

		// 	return new MersenneCpuDivisorScanSession(gpu, orderDevice);
		// }

		// [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		// internal static void ReturnMersenneCpuDivisorSession(MersenneCpuDivisorScanSession session) => _mersenneCpuDivisorSession = session;

		[ThreadStatic]
		private static ArrayPool<UInt128>? _uint128Pool;

		public static ArrayPool<UInt128> UInt128Pool
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
			get => _uint128Pool ??= ArrayPool<UInt128>.Create();
		}

		[ThreadStatic]
		private static ArrayPool<ulong>? _ulongPool;

		public static ArrayPool<ulong> UlongPool
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
			get => _ulongPool ??= ArrayPool<ulong>.Create();
		}

		[ThreadStatic]
		private static ArrayPool<bool>? _boolPool;

		public static ArrayPool<bool> BoolPool
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
			get => _boolPool ??= ArrayPool<bool>.Create();
		}

		[ThreadStatic]
		private static ArrayPool<byte>? _bytePool;

		public static ArrayPool<byte> BytePool
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
			get => _bytePool ??= ArrayPool<byte>.Create();
		}

		[ThreadStatic]
		private static ArrayPool<GpuDivisorPartialData>? _gpuDivisorPartialDataPool;

		internal static ArrayPool<GpuDivisorPartialData> GpuDivisorPartialDataPool
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
			get => _gpuDivisorPartialDataPool ??= ArrayPool<GpuDivisorPartialData>.Create();
		}

		[ThreadStatic]
		private static ArrayPool<int>? _intPool;

		public static ArrayPool<int> IntPool
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
			get => _intPool ??= ArrayPool<int>.Create();
		}

		[ThreadStatic]
		private static ArrayPool<char>? _charPool;

		public static ArrayPool<char> CharPool
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
			get => _charPool ??= ArrayPool<char>.Create();
		}

		[ThreadStatic]
		private static FixedCapacityStack<List<ulong>>? _ulongListPool;

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public static List<ulong> RentUlongList(int capacityHint)
		{
			FixedCapacityStack<List<ulong>>? pool = _ulongListPool;
			if (pool is not null)
			{
				if (pool.Pop() is { } list)
				{
					if (list.Capacity < capacityHint)
					{
						list.Capacity = capacityHint;
					}

					return list;
				}
			}
			else
			{
				_ulongListPool = new(capacityHint);
			}

			return new List<ulong>(capacityHint);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public static void ReturnUlongList(List<ulong> list) => _ulongListPool!.Push(list);

		[ThreadStatic]
		private static ArrayPool<(ulong, ulong)>? _ulongUlongTupleArrayPool;

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public static (ulong, ulong)[] RentUlongUlongArray(int capacityHint)
		{
			_ulongUlongTupleArrayPool ??= ArrayPool<(ulong, ulong)>.Create();
			ArrayPool<(ulong, ulong)>? pool = _ulongUlongTupleArrayPool;
			return pool.Rent(capacityHint);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public static void Return((ulong, ulong)[] array) => _ulongUlongTupleArrayPool!.Return(array, clearArray: false);

		[ThreadStatic]
		private static FixedCapacityStack<List<PartialFactorPendingEntry>>? _primeOrderPendingEntryListPool;

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		internal static List<PartialFactorPendingEntry> RentPrimeOrderPendingEntryList(int capacityHint)
		{
			FixedCapacityStack<List<PartialFactorPendingEntry>>? pool = _primeOrderPendingEntryListPool;
			if (pool is not null)
			{
				if (pool.Pop() is { } list)
				{
					if (list.Capacity < capacityHint)
					{
						list.Capacity = capacityHint;
					}

					return list;
				}
			}
			else
			{
				_primeOrderPendingEntryListPool = new(PerfectNumberConstants.DefaultPoolCapacity);
			}

			return new List<PartialFactorPendingEntry>(capacityHint);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		internal static void ReturnPrimeOrderPendingEntryList(List<PartialFactorPendingEntry> list) => _primeOrderPendingEntryListPool!.Push(list);

		[ThreadStatic]
		private static FixedCapacityStack<FixedCapacityStack<ulong>>? _ulongStackPool;

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public static FixedCapacityStack<ulong> RentUlongStack(int capacityHint)
		{
			FixedCapacityStack<FixedCapacityStack<ulong>>? pool = _ulongStackPool;
			if (pool is not null)
			{
				if (pool.Pop() is { } stack)
				{
					stack.Capacity = capacityHint;
					return stack;
				}
			}
			else
			{
				_ulongStackPool = new(PrimeOrderConstants.GpuSmallPrimeFactorSlots);
			}

			return new FixedCapacityStack<ulong>(capacityHint);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public static void ReturnUlongStack(FixedCapacityStack<ulong> stack) => _ulongStackPool!.Push(stack);

		[ThreadStatic]
		private static Dictionary<ulong, int>? _factorCountDictionary;

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public static Dictionary<ulong, int> RentFactorCountDictionary()
		{
			Dictionary<ulong, int>? dictionary = _factorCountDictionary;
			if (dictionary is null)
			{
				return new Dictionary<ulong, int>(8);
			}

			_factorCountDictionary = null;
			return dictionary;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public static void ReturnFactorCountDictionary(Dictionary<ulong, int> dictionary) => _factorCountDictionary = dictionary;

		[ThreadStatic]
		private static Dictionary<ulong, int>? _factorScratchDictionary;

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public static Dictionary<ulong, int> RentFactorScratchDictionary()
		{
			if (_factorScratchDictionary is { } dictionary)
			{
				_factorScratchDictionary = null;
				return dictionary;
			}

			return new Dictionary<ulong, int>(8);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public static void ReturnFactorScratchDictionary(Dictionary<ulong, int> dictionary) => _factorScratchDictionary = dictionary;

		[ThreadStatic]
		private static ExponentRemainderStepperCpu _exponentRemainderStepperCpu;

		[ThreadStatic]
		private static bool _hasExponentRemainderStepperCpu;

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		internal static ExponentRemainderStepperCpu RentExponentStepperCpu(in MontgomeryDivisorData divisorData, ulong cycleLength = 0UL)
		{
			if (_hasExponentRemainderStepperCpu && _exponentRemainderStepperCpu.MatchesDivisor(divisorData, cycleLength))
			{
				ExponentRemainderStepperCpu stepper = _exponentRemainderStepperCpu;
				_hasExponentRemainderStepperCpu = false;
				stepper.Reset();
				return stepper;
			}

			return new ExponentRemainderStepperCpu(divisorData, cycleLength);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		internal static void ReturnExponentStepperCpu(ExponentRemainderStepperCpu stepper)
		{
			_exponentRemainderStepperCpu = stepper;
			_hasExponentRemainderStepperCpu = true;
		}
	}
}
