using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core
{
	public sealed class PartialFactorResult
	{
		[ThreadStatic]
		private static PartialFactorResult? s_poolHead;

		public readonly ulong[] Factors = new ulong[PrimeOrderConstants.GpuSmallPrimeFactorSlots];
		public readonly int[] Exponents = new int[PrimeOrderConstants.GpuSmallPrimeFactorSlots];
		public readonly FixedCapacityStack<ulong> CompositeStack = new (PrimeOrderConstants.GpuSmallPrimeFactorSlots);
		public readonly List<PartialFactorPendingEntry> PendingFactors = new (PrimeOrderConstants.GpuSmallPrimeFactorSlots);
		public readonly ulong[] TempFactorsArray = new ulong[PrimeOrderConstants.GpuSmallPrimeFactorSlots << 3];
		public readonly int[] TempExponentsArray = new int[PrimeOrderConstants.GpuSmallPrimeFactorSlots << 3];
		public readonly ulong[] StackCandidates = new ulong[PrimeOrderConstants.GpuSmallPrimeFactorSlots << 3];
		public readonly bool[] StackEvaluations = new bool[PrimeOrderConstants.GpuSmallPrimeFactorSlots << 3];
		private PartialFactorResult? _next;
		public ulong Cofactor;
		public bool FullyFactored;
		public int Count;
		public bool CofactorIsPrime;

		private PartialFactorResult()
		{
		}

		public static PartialFactorResult Rent()
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
			instance.Cofactor = 0UL;
			instance.FullyFactored = false;
			instance.Count = 0;
			instance.CofactorIsPrime = false;
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
			if (sourceCount == 0)
			{
				InitializeWithPrime(prime);
				return;
			}

			int[] sourceExponents = Exponents!;

			int newSize = sourceCount + 1;
			sourceFactors[sourceCount] = prime;
			sourceExponents[sourceCount] = 1;
			Array.Sort(sourceFactors, sourceExponents, 0, newSize);

			Cofactor = 1UL;
			FullyFactored = true;
			Count = newSize;
			CofactorIsPrime = false;
		}

		private void InitializeWithPrime(ulong prime)
		{
			Factors[0] = prime;
			Exponents[0] = 1;
			Cofactor = 1UL;
			FullyFactored = true;
			Count = 1;
			CofactorIsPrime = false;
		}

		public void Dispose()
		{
			if (this != Empty)
			{
				_next = s_poolHead;
				s_poolHead = this;
			}
		}
	}
}
