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
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return _eintegerPool ??= ArrayPool<EInteger>.Create();
            }
        }

        [ThreadStatic]
        private static Queue<List<KeyValuePair<UInt128, int>>>? _keyValuePairUInt128IntegerPool;

        public static Queue<List<KeyValuePair<UInt128, int>>> KeyValuePairUInt128IntegerPool
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return _keyValuePairUInt128IntegerPool ??= [];
            }
        }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static List<KeyValuePair<UInt128, int>> Rent(this Queue<List<KeyValuePair<UInt128, int>>> pool, int capacity)
		{
			if (pool.TryDequeue(out var pooled))
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

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static void Return(this Queue<List<KeyValuePair<UInt128, int>>> pool, List<KeyValuePair<UInt128, int>> list) => pool.Enqueue(list);

        [ThreadStatic]
        private static Queue<List<UInt128>>? _uint128ListPool;

        public static Queue<List<UInt128>> UInt128ListPool
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return _uint128ListPool ??= [];
            }
        }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static List<UInt128> Rent(this Queue<List<UInt128>> pool, int capacity)
		{
			if (pool.TryDequeue(out var pooled))
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

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static void Return(this Queue<List<UInt128>> pool, List<UInt128> list) => pool.Enqueue(list);

        [ThreadStatic]
        private static Queue<Dictionary<UInt128, int>>? _uint128IntDictionaryPool;

        public static Queue<Dictionary<UInt128, int>> UInt128IntDictionaryPool
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return _uint128IntDictionaryPool ??= [];
            }
        }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Dictionary<UInt128, int> Rent(this Queue<Dictionary<UInt128, int>> pool, int capacity)
		{
			if (pool.TryDequeue(out var pooled))
			{
				pooled.EnsureCapacity(capacity);
				pooled.Clear();
				return pooled;
			}

			return new(capacity);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static void Return(this Queue<Dictionary<UInt128, int>> pool, Dictionary<UInt128, int> dictionary) => pool.Enqueue(dictionary);

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
        private static ArrayPool<GpuUInt128>? _gpuUInt128Pool;

        public static ArrayPool<GpuUInt128> GpuUInt128Pool
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return _gpuUInt128Pool ??= ArrayPool<GpuUInt128>.Create();
            }
        }

        [ThreadStatic]
        private static MersenneCpuDivisorScanSession? _mersenneCpuDivisorSession;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static MersenneCpuDivisorScanSession? RentMersenneCpuDivisorSession()
        {
            MersenneCpuDivisorScanSession? session = _mersenneCpuDivisorSession;
            if (session is null)
            {
                return null;
            }

            _mersenneCpuDivisorSession = null;
            return session;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void ReturnMersenneCpuDivisorSession(MersenneCpuDivisorScanSession session)
        {
            _mersenneCpuDivisorSession = session;
        }

        [ThreadStatic]
        private static ArrayPool<UInt128>? _uint128Pool;

        public static ArrayPool<UInt128> UInt128Pool
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return _uint128Pool ??= ArrayPool<UInt128>.Create();
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

        [ThreadStatic]
        private static ArrayPool<bool>? _boolPool;

        public static ArrayPool<bool> BoolPool
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return _boolPool ??= ArrayPool<bool>.Create();
            }
        }

        [ThreadStatic]
        private static ArrayPool<byte>? _bytePool;

        public static ArrayPool<byte> BytePool
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return _bytePool ??= ArrayPool<byte>.Create();
            }
        }

        [ThreadStatic]
        private static ArrayPool<MontgomeryDivisorData>? _montgomeryDivisorDataPool;

        public static ArrayPool<MontgomeryDivisorData> MontgomeryDivisorDataPool
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return _montgomeryDivisorDataPool ??= ArrayPool<MontgomeryDivisorData>.Create();
            }
        }

        [ThreadStatic]
        private static ArrayPool<GpuDivisorPartialData>? _gpuDivisorPartialDataPool;

        internal static ArrayPool<GpuDivisorPartialData> GpuDivisorPartialDataPool
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return _gpuDivisorPartialDataPool ??= ArrayPool<GpuDivisorPartialData>.Create();
            }
        }

        [ThreadStatic]
        private static ArrayPool<int>? _intPool;

        public static ArrayPool<int> IntPool
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return _intPool ??= ArrayPool<int>.Create();
            }
        }

        [ThreadStatic]
        private static ArrayPool<char>? _charPool;

        public static ArrayPool<char> CharPool
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return _charPool ??= ArrayPool<char>.Create();
            }
        }

        [ThreadStatic]
        private static List<List<ulong>>? _ulongListPool;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static List<ulong> RentUlongList(int capacityHint)
        {
            List<List<ulong>>? pool = _ulongListPool;
            if (pool is not null)
            {
                int poolCount = pool.Count;
                if (poolCount > 0)
                {
                    int lastIndex = poolCount - 1;
                    List<ulong> list = pool[lastIndex];
                    pool.RemoveAt(lastIndex);
                    if (list.Capacity < capacityHint)
                    {
                        list.Capacity = capacityHint;
                    }

                    return list;
                }
            }

            return new List<ulong>(capacityHint);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ReturnUlongList(List<ulong> list)
        {
            List<List<ulong>>? pool = _ulongListPool;
            if (pool is null)
            {
                pool = new List<List<ulong>>(4);
                _ulongListPool = pool;
            }

            pool.Add(list);
        }

        [ThreadStatic]
        private static ArrayPool<(ulong, ulong)>? _ulongUlongTupleArrayPool;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static (ulong, ulong)[] RentUlongUlongArray(int capacityHint)
		{
			_ulongUlongTupleArrayPool ??= ArrayPool<(ulong, ulong)>.Create();
			ArrayPool<(ulong, ulong)>? pool = _ulongUlongTupleArrayPool;			
			return pool.Rent(capacityHint);
        }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static void Return((ulong, ulong)[] array) => _ulongUlongTupleArrayPool!.Return(array, clearArray: false);

		[ThreadStatic]
        private static List<List<PrimeOrderCalculator.PendingEntry>>? _primeOrderPendingEntryListPool;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static List<PrimeOrderCalculator.PendingEntry> RentPrimeOrderPendingEntryList(int capacityHint)
        {
            List<List<PrimeOrderCalculator.PendingEntry>>? pool = _primeOrderPendingEntryListPool;
            if (pool is not null)
            {
                int poolCount = pool.Count;
                if (poolCount > 0)
                {
                    int lastIndex = poolCount - 1;
                    List<PrimeOrderCalculator.PendingEntry> list = pool[lastIndex];
                    pool.RemoveAt(lastIndex);
                    if (list.Capacity < capacityHint)
                    {
                        list.Capacity = capacityHint;
                    }

                    return list;
                }
            }

            return new List<PrimeOrderCalculator.PendingEntry>(capacityHint);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void ReturnPrimeOrderPendingEntryList(List<PrimeOrderCalculator.PendingEntry> list)
        {
            List<List<PrimeOrderCalculator.PendingEntry>>? pool = _primeOrderPendingEntryListPool;
            if (pool is null)
            {
                pool = new List<List<PrimeOrderCalculator.PendingEntry>>(4);
                _primeOrderPendingEntryListPool = pool;
            }

            pool.Add(list);
        }

        [ThreadStatic]
        private static List<Stack<ulong>>? _ulongStackPool;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Stack<ulong> RentUlongStack(int capacityHint)
        {
            List<Stack<ulong>>? pool = _ulongStackPool;
            if (pool is not null)
            {
                int poolCount = pool.Count;
                if (poolCount > 0)
                {
                    int lastIndex = poolCount - 1;
                    Stack<ulong> stack = pool[lastIndex];
                    pool.RemoveAt(lastIndex);

                    if (stack.Count < capacityHint)
                    {
                        stack.EnsureCapacity(capacityHint);
                    }

                    return stack;
                }
            }

            return new Stack<ulong>(capacityHint);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ReturnUlongStack(Stack<ulong> stack)
        {
            List<Stack<ulong>>? pool = _ulongStackPool;
            if (pool is null)
            {
                pool = new List<Stack<ulong>>(4);
                _ulongStackPool = pool;
            }

            pool.Add(stack);
        }

        [ThreadStatic]
        private static Dictionary<ulong, int>? _factorCountDictionary;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ReturnFactorCountDictionary(Dictionary<ulong, int> dictionary)
        {
            _factorCountDictionary = dictionary;
        }

        [ThreadStatic]
        private static Dictionary<ulong, int>? _factorScratchDictionary;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Dictionary<ulong, int> RentFactorScratchDictionary()
        {
            Dictionary<ulong, int>? dictionary = _factorScratchDictionary;
            if (dictionary is null)
            {
                return new Dictionary<ulong, int>(8);
            }

            _factorScratchDictionary = null;
            return dictionary;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ReturnFactorScratchDictionary(Dictionary<ulong, int> dictionary)
        {
            _factorScratchDictionary = dictionary;
        }

        [ThreadStatic]
        private static ExponentRemainderStepperCpu _exponentRemainderStepperCpu;

        [ThreadStatic]
        private static bool _hasExponentRemainderStepperCpu;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void ReturnExponentStepperCpu(ExponentRemainderStepperCpu stepper)
        {
            stepper.Reset();
            _exponentRemainderStepperCpu = stepper;
            _hasExponentRemainderStepperCpu = true;
        }
    }
}
