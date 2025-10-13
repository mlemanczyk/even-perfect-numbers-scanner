using System;
using System.Buffers;

namespace PerfectNumbers.Core
{
    internal sealed class PrimeFactorPartialResult
    {
        [ThreadStatic]
        private static PrimeFactorPartialResult? s_poolHead;

        private PrimeFactorPartialResult? _next;
        private ulong[]? _primes;
        private byte[]? _exponents;
        private bool _arraysFromPool;
        private int _capacity;

        private PrimeFactorPartialResult()
        {
        }

        public static PrimeFactorPartialResult Rent(int capacity)
        {
            PrimeFactorPartialResult? instance = s_poolHead;
            if (instance is null)
            {
                instance = new PrimeFactorPartialResult();
            }
            else
            {
                s_poolHead = instance._next;
            }

            instance._next = null;
            instance.EnsureCapacity(capacity);
            instance.Count = 0;
            return instance;
        }

        public int Count { get; private set; }

        internal ReadOnlySpan<ulong> Primes => _primes is null ? ReadOnlySpan<ulong>.Empty : _primes.AsSpan(0, Count);

        internal ReadOnlySpan<byte> Exponents => _exponents is null ? ReadOnlySpan<byte>.Empty : _exponents.AsSpan(0, Count);

        internal Span<ulong> PrimeWriteSpan => _primes is null ? Span<ulong>.Empty : _primes.AsSpan(0, _capacity);

        internal Span<byte> ExponentWriteSpan => _exponents is null ? Span<byte>.Empty : _exponents.AsSpan(0, _capacity);

        internal void CommitCount(int count)
        {
            Count = count;
        }

        internal void Sort()
        {
            if (Count == 0 || _primes is null || _exponents is null)
            {
                return;
            }

            Array.Sort(_primes, _exponents, 0, Count);
        }

        internal MersenneDivisorCycles.FactorCacheEntry ToCacheEntry()
        {
            if (Count == 0 || _primes is null || _exponents is null)
            {
                return default;
            }

            ulong[] primes = new ulong[Count];
            byte[] exponents = new byte[Count];
            Array.Copy(_primes, 0, primes, 0, Count);
            Array.Copy(_exponents, 0, exponents, 0, Count);
            return new MersenneDivisorCycles.FactorCacheEntry(primes, exponents, Count);
        }

        public void Dispose()
        {
            if (_arraysFromPool)
            {
                if (_primes is not null)
                {
                    ThreadLocalArrayPool<ulong>.Shared.Return(_primes, clearArray: false);
                }

                if (_exponents is not null)
                {
                    ThreadLocalArrayPool<byte>.Shared.Return(_exponents, clearArray: false);
                }

                _arraysFromPool = false;
                _primes = null;
                _exponents = null;
            }

            _capacity = 0;
            Count = 0;

            _next = s_poolHead;
            s_poolHead = this;
        }

        private void EnsureCapacity(int capacity)
        {
            if (capacity <= 0)
            {
                _capacity = 0;
                return;
            }

            if (_primes is not null && _primes.Length >= capacity)
            {
                _capacity = capacity;
                return;
            }

            if (_arraysFromPool)
            {
                if (_primes is not null)
                {
                    ThreadLocalArrayPool<ulong>.Shared.Return(_primes, clearArray: false);
                }

                if (_exponents is not null)
                {
                    ThreadLocalArrayPool<byte>.Shared.Return(_exponents, clearArray: false);
                }

                _arraysFromPool = false;
                _primes = null;
                _exponents = null;
            }

            if (capacity >= PerfectNumberConstants.PooledArrayThreshold)
            {
                _primes = ThreadLocalArrayPool<ulong>.Shared.Rent(capacity);
                _exponents = ThreadLocalArrayPool<byte>.Shared.Rent(capacity);
                _arraysFromPool = true;
            }
            else
            {
                _primes = new ulong[capacity];
                _exponents = new byte[capacity];
            }

            _capacity = capacity;
        }
    }
}
