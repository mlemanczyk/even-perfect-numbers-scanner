namespace PerfectNumbers.Core
{
    public sealed class PartialFactorResult
    {
        [ThreadStatic]
        private static PartialFactorResult? s_poolHead;

		public ulong[]? Factors;
		public int[]? Exponents;
        private PartialFactorResult? _next;
        public ulong Cofactor;
        public bool FullyFactored;
		public int Count;
		public bool CofactorIsPrime;

        private PartialFactorResult()
        {
        }

        public static PartialFactorResult Rent(in ulong[]? factors, in int[]? exponents, ulong cofactor, bool fullyFactored, int count, bool cofactorIsPrime)
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
			instance.Exponents = exponents;
			instance.Cofactor = cofactor;
            instance.FullyFactored = fullyFactored;
			instance.Count = count;
			instance.CofactorIsPrime = cofactorIsPrime;
            return instance;
        }

        public static PartialFactorResult Rent(ulong cofactor, bool fullyFactored, int count, bool cofactorIsPrime)
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
			instance.Factors = null;
			instance.Exponents = null;
			instance.Cofactor = cofactor;
            instance.FullyFactored = fullyFactored;
			instance.Count = count;
			instance.CofactorIsPrime = cofactorIsPrime;
            return instance;
        }

        public static readonly PartialFactorResult Empty = new()
		{
			Cofactor = 1UL,
			FullyFactored = true
		};

        public PartialFactorResult WithAdditionalPrime(ulong prime)
		{			
            ulong[]? source = Factors;
			int sourceCount = Count;
			int[] exponents;
			if (source is null || sourceCount == 0)
			{
				source = ThreadStaticPools.UlongPool.Rent(1);
				exponents = ThreadStaticPools.IntPool.Rent(1);
				// source = new ulong[1];
				// exponents = new int[1];
				source[0] = prime;
				exponents[0] = 1;

				return Rent(source, exponents, 1UL, true, 1, false);
            }

            ulong[] extended = ThreadStaticPools.UlongPool.Rent(sourceCount + 1);
            Array.Copy(source, 0, extended, 0, sourceCount);
			extended[sourceCount] = prime;

			exponents = ThreadStaticPools.IntPool.Rent(sourceCount + 1);
            Array.Copy(Exponents!, 0, exponents, 0, sourceCount);
			exponents[sourceCount] = 1;
			Array.Sort(extended, exponents, 0, sourceCount + 1);

			return Rent(extended, exponents, 1UL, true, sourceCount + 1, false);
        }

        public void Dispose()
        {
            if (this != Empty)
            {
                ulong[]? factors = Factors;
                if (factors is not null)
                {
					int[] exponents = Exponents!;
					Factors = null;
					Exponents = null;

					// if (factors.Length > 1)
					// {
					ThreadStaticPools.UlongPool.Return(factors, clearArray: false);
					ThreadStaticPools.IntPool.Return(exponents, clearArray: false);						
					// }
                }

                _next = s_poolHead;
                s_poolHead = this;
            }
        }
    }
}
