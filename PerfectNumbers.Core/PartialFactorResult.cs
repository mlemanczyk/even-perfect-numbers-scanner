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

		public void WithAdditionalPrime(ulong prime)
		{
			ulong[]? sourceFactors = Factors;
			int sourceCount = Count;
			if (sourceFactors is null || sourceCount == 0)
			{
				InitializeWithPrime(prime);
				return;
			}

			int newSize = sourceCount + 1;
			ulong[] extendedFactors = sourceFactors.Length >= newSize
				? sourceFactors
				: ResizeFactors(sourceFactors, sourceCount, newSize);

			int[] sourceExponents = Exponents!;
			int[] extendedExponents = sourceExponents.Length >= newSize
				? sourceExponents
				: ResizeExponents(sourceCount, newSize, sourceExponents);

			extendedFactors[sourceCount] = prime;
			extendedExponents[sourceCount] = 1;
			Array.Sort(extendedFactors, extendedExponents, 0, newSize);

			Factors = extendedFactors;
			Exponents = extendedExponents;
			Cofactor = 1UL;
			FullyFactored = true;
			Count = newSize;
			CofactorIsPrime = false;
		}

		private static int[] ResizeExponents(int sourceCount, int newSize, in int[] sourceExponents)
		{
			int[] extendedExponents = Pools.ExclusiveIntArray.Rent(newSize);
			Array.Copy(sourceExponents, 0, extendedExponents, 0, sourceCount);
			Pools.ExclusiveIntArray.Return(sourceExponents);
			return extendedExponents;
		}

		private static ulong[] ResizeFactors(in ulong[] sourceFactors, int sourceCount, int newSize)
		{
			ulong[] extendedFactors = Pools.ExclusiveUlongArray.Rent(newSize);
			Array.Copy(sourceFactors, 0, extendedFactors, 0, sourceCount);
			Pools.ExclusiveUlongArray.Return(sourceFactors);
			return extendedFactors;
		}

		private void InitializeWithPrime(ulong prime)
		{
			ulong[] factors = Pools.ExclusiveUlongArray.Rent(1);
			int[] exponents = Pools.ExclusiveIntArray.Rent(1);
			factors[0] = prime;
			exponents[0] = 1;
			Factors = factors;
			Exponents = exponents;
			Cofactor = 1UL;
			FullyFactored = true;
			Count = 1;
			CofactorIsPrime = false;
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

					Pools.ExclusiveUlongArray.Return(factors);
					Pools.ExclusiveIntArray.Return(exponents);
				}

				_next = s_poolHead;
				s_poolHead = this;
			}
		}
	}
}
