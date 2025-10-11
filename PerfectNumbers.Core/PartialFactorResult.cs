using System;

namespace PerfectNumbers.Core
{
    public sealed class PartialFactorResult
    {
        [ThreadStatic]
        private static PartialFactorResult? s_poolHead;

        public static readonly PartialFactorResult Empty = new()
        {
            Cofactor = 1UL,
            FullyFactored = true,
        };

        public FactorEntry[]? Factors;
        public ulong Cofactor;
        public bool FullyFactored;
        public int Count;

        private PartialFactorResult? _next;
        private bool _factorsFromPool;

        private PartialFactorResult()
        {
        }

        public static PartialFactorResult Rent(in FactorEntry[]? factors, ulong cofactor, bool fullyFactored, int count)
        {
            PartialFactorResult? instance = s_poolHead;
            if (instance is null)
            {
                instance = new PartialFactorResult();
            }
            else
            {
                s_poolHead = instance._next;
            }

            FactorEntry[]? factorArray = factors;
            instance._next = null;
            instance.Factors = factorArray;
            instance.Cofactor = cofactor;
            instance.FullyFactored = fullyFactored;
            instance.Count = count;
            instance._factorsFromPool = factorArray is not null && factorArray.Length >= PerfectNumberConstants.PooledArrayThreshold;
            return instance;
        }

        public PartialFactorResult WithAdditionalPrime(ulong prime)
        {
            FactorEntry[]? source = Factors;
            if (source is null || Count == 0)
            {
                FactorEntry[] single = new FactorEntry[1];
                single[0] = new FactorEntry(prime);
                return Rent(single, 1UL, true, 1);
            }

            int newCount = Count + 1;
            FactorEntry[] extended = newCount >= PerfectNumberConstants.PooledArrayThreshold
                ? ThreadLocalArrayPool<FactorEntry>.Shared.Rent(newCount)
                : new FactorEntry[newCount];

            Array.Copy(source, 0, extended, 0, Count);
            extended[Count] = new FactorEntry(prime);
            Span<FactorEntry> span = extended.AsSpan(0, newCount);
            span.Sort(static (a, b) => a.Value.CompareTo(b.Value));
            return Rent(extended, 1UL, true, newCount);
        }

        public void Dispose()
        {
            if (this == Empty)
            {
                return;
            }

            FactorEntry[]? factors = Factors;
            if (factors is not null)
            {
                Factors = null;
                if (_factorsFromPool)
                {
                    ThreadLocalArrayPool<FactorEntry>.Shared.Return(factors, clearArray: false);
                }
            }

            _factorsFromPool = false;
            _next = s_poolHead;
            s_poolHead = this;
        }
    }
}
