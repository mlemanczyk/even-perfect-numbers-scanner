using System;
using System.Buffers;

namespace PerfectNumbers.Core
{
    public sealed class PartialFactorResult
    {
		[ThreadStatic]
		private static PartialFactorResult? s_poolHead;

        public FactorEntry[]? Factors;
		private PartialFactorResult? _next;
        public ulong Cofactor;
        public bool FullyFactored;
        public int Count;

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

            instance._next = null;
            instance.Factors = factors;
            instance.Cofactor = cofactor;
            instance.FullyFactored = fullyFactored;
            instance.Count = count;
            return instance;
        }

        public static readonly PartialFactorResult Empty = new()
		{
            Cofactor = 1UL,
            FullyFactored = true,
        };

        public PartialFactorResult WithAdditionalPrime(ulong prime)
        {
            FactorEntry[]? source = Factors;
            if (source is null || Count == 0)
            {
                FactorEntry[] local = ArrayPool<FactorEntry>.Shared.Rent(1);
                local[0] = new FactorEntry(prime);
                return Rent(local, 1UL, true, 1);
            }

            FactorEntry[] extended = ArrayPool<FactorEntry>.Shared.Rent(Count + 1);
            Array.Copy(source, 0, extended, 0, Count);
            extended[Count] = new FactorEntry(prime);
            Span<FactorEntry> span = extended.AsSpan(0, Count + 1);
            span.Sort(static (a, b) => a.Value.CompareTo(b.Value));
            return Rent(extended, 1UL, true, Count + 1);
        }

        public void Dispose()
        {
            if (this != Empty)
            {
                FactorEntry[]? factors = Factors;
                if (factors is not null)
                {
                    Factors = null;
                    ArrayPool<FactorEntry>.Shared.Return(factors, clearArray: false);
                }

                _next = s_poolHead;
                s_poolHead = this;
            }
        }
    }
}
