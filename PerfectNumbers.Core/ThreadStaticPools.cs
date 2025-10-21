using System.Buffers;
using System.Collections.Generic;
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
                    int stackCount = stack.Count;
                    if (stackCount > 0)
                    {
                        stack.Clear();
                        stackCount = 0;
                    }

                    if (stackCount < capacityHint)
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
        private static List<Dictionary<ulong, int>>? _ulongIntDictionaryPool;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Dictionary<ulong, int> RentUlongIntDictionary(int capacityHint)
        {
            List<Dictionary<ulong, int>>? pool = _ulongIntDictionaryPool;
            if (pool is not null)
            {
                int poolCount = pool.Count;
                if (poolCount > 0)
                {
                    int lastIndex = poolCount - 1;
                    Dictionary<ulong, int> dictionary = pool[lastIndex];
                    pool.RemoveAt(lastIndex);
                    if (dictionary.Count > 0)
                    {
                        dictionary.Clear();
                    }

                    dictionary.EnsureCapacity(capacityHint);

                    return dictionary;
                }
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

        [ThreadStatic]
        private static ExponentRemainderStepper _exponentRemainderStepper;

        [ThreadStatic]
        private static bool _hasExponentRemainderStepper;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static ExponentRemainderStepper RentExponentStepper(in MontgomeryDivisorData divisorData)
        {
            if (_hasExponentRemainderStepper && _exponentRemainderStepper.MatchesDivisor(divisorData))
            {
                ExponentRemainderStepper stepper = _exponentRemainderStepper;
                _hasExponentRemainderStepper = false;
                stepper.Reset();
                return stepper;
            }

            return new ExponentRemainderStepper(divisorData);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void ReturnExponentStepper(ExponentRemainderStepper stepper)
        {
            stepper.Reset();
            _exponentRemainderStepper = stepper;
            _hasExponentRemainderStepper = true;
        }
    }
}
