using System;
using System.Buffers;
using System.Collections.Generic;
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

        [ThreadStatic]
        private static Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>? _mersenneFactorCacheDictionary;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry> TakeMersenneFactorCacheDictionary()
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
            cache.Clear();
            _mersenneFactorCacheDictionary = cache;
        }

        [ThreadStatic]
        private static Dictionary<ulong, int>? _factorCountDictionary;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Dictionary<ulong, int> TakeFactorCountDictionary()
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
        public static Dictionary<ulong, int> TakeFactorScratchDictionary()
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
