using System.Buffers;
using System.Runtime.CompilerServices;
using PerfectNumbers.Core.Cpu;

namespace PerfectNumbers.Core
{
    public readonly struct ThreadStaticPools
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
        private static ArrayPool<KeyValuePair<ulong, int>>? _factorKeyValuePairPool;

        public static ArrayPool<KeyValuePair<ulong, int>> FactorKeyValuePairPool
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return _factorKeyValuePairPool ??= ArrayPool<KeyValuePair<ulong, int>>.Create();
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
            session.Reset();
            return session;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void ReturnMersenneCpuDivisorSession(MersenneCpuDivisorScanSession session)
        {
            _mersenneCpuDivisorSession = session;
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
        private static List<List<ulong>>? _ulongListPool;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static List<ulong> RentUlongList(int capacityHint)
        {
            List<List<ulong>>? pool = _ulongListPool;
            if (pool is not null && pool.Count > 0)
            {
                int lastIndex = pool.Count - 1;
                List<ulong> list = pool[lastIndex];
                pool.RemoveAt(lastIndex);
                if (list.Count > 0)
                {
                    list.Clear();
                }

                if (list.Capacity < capacityHint)
                {
                    list.Capacity = capacityHint;
                }

                return list;
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
        private static List<Stack<ulong>>? _ulongStackPool;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Stack<ulong> RentUlongStack(int capacityHint)
        {
            List<Stack<ulong>>? pool = _ulongStackPool;
            if (pool is not null && pool.Count > 0)
            {
                int lastIndex = pool.Count - 1;
                Stack<ulong> stack = pool[lastIndex];
                pool.RemoveAt(lastIndex);
                if (stack.Count > 0)
                {
                    stack.Clear();
                }

                if (stack.Count < capacityHint)
                {
                    stack.EnsureCapacity(capacityHint);
                }

                return stack;
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
        private static List<Dictionary<ulong, int>>? _ulongIntDictionaryPool;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Dictionary<ulong, int> RentUlongIntDictionary(int capacityHint)
        {
            List<Dictionary<ulong, int>>? pool = _ulongIntDictionaryPool;
            if (pool is not null && pool.Count > 0)
            {
                int lastIndex = pool.Count - 1;
                Dictionary<ulong, int> dictionary = pool[lastIndex];
                pool.RemoveAt(lastIndex);
                if (dictionary.Count > 0)
                {
                    dictionary.Clear();
                }

                if (dictionary.Count < capacityHint)
                {
                    dictionary.EnsureCapacity(capacityHint);
                }

                return dictionary;
            }

            return new Dictionary<ulong, int>(capacityHint);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ReturnUlongIntDictionary(Dictionary<ulong, int> dictionary)
        {
            List<Dictionary<ulong, int>>? pool = _ulongIntDictionaryPool;
            if (pool is null)
            {
                pool = new List<Dictionary<ulong, int>>(4);
                _ulongIntDictionaryPool = pool;
            }

            pool.Add(dictionary);
        }

        [ThreadStatic]
        private static Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>? _mersenneFactorCacheDictionary;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry> RentMersenneFactorCacheDictionary()
        {
            Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>? cache = _mersenneFactorCacheDictionary;
            if (cache is null)
            {
                return new Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>(8);
            }

            _mersenneFactorCacheDictionary = null;
            cache.Clear();
            return cache;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ReturnMersenneFactorCacheDictionary(Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry> cache)
        {
            for (Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>.Enumerator enumerator = cache.GetEnumerator(); enumerator.MoveNext();)
            {
                enumerator.Current.Value.ReturnToPool();
            }

            cache.Clear();
            _mersenneFactorCacheDictionary = cache;
        }

        [ThreadStatic]
        private static MersenneDivisorCycles.FactorCacheEntry? _factorCacheEntryPoolHead;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static MersenneDivisorCycles.FactorCacheEntry RentFactorCacheEntry()
        {
            MersenneDivisorCycles.FactorCacheEntry? entry = _factorCacheEntryPoolHead;
            if (entry is null)
            {
                entry = new MersenneDivisorCycles.FactorCacheEntry();
            }
            else
            {
                _factorCacheEntryPoolHead = entry.Next;
            }

            entry.Reset();
            return entry;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ReturnFactorCacheEntry(MersenneDivisorCycles.FactorCacheEntry entry)
        {
            entry.Next = _factorCacheEntryPoolHead;
            _factorCacheEntryPoolHead = entry;
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
            dictionary.Clear();
            return dictionary;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ReturnFactorCountDictionary(Dictionary<ulong, int> dictionary)
        {
            dictionary.Clear();
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
            dictionary.Clear();
            return dictionary;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ReturnFactorScratchDictionary(Dictionary<ulong, int> dictionary)
        {
            dictionary.Clear();
            _factorScratchDictionary = dictionary;
        }
    }
}
