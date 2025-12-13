using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core
{
	public sealed class PartialFactorResult
	{
		private const int ExponentHardLimit = 256;

		[ThreadStatic]
		private static PartialFactorResult? s_poolHead;

		public static readonly PartialFactorResult Empty = new(MontgomeryDivisorData.Empty)
		{
			Cofactor = 1UL,
			FullyFactored = true
		};

		internal readonly List<PartialFactorPendingEntry> Pending = new(PrimeOrderConstants.GpuSmallPrimeFactorSlots);
		internal readonly FixedCapacityStack<ulong> CompositeStack = new(PrimeOrderConstants.GpuSmallPrimeFactorSlots);
		internal readonly ulong[] SpecialMaxBuffer = new ulong[PrimeOrderConstants.GpuSmallPrimeFactorSlots];

		public readonly ulong[] Factors = new ulong[ExponentHardLimit];
		public readonly int[] Exponents = new int[ExponentHardLimit];
		public readonly ulong[] FactorCandidates = new ulong[ExponentHardLimit];
		public readonly List<ulong> FactorCandidatesList = new(ExponentHardLimit);
		public readonly ulong[] ValidateOrderFactorCandidates = new ulong[ExponentHardLimit];
		public readonly bool[] FactorCandidateEvaluations = new bool[ExponentHardLimit];
		internal readonly Dictionary<ulong, int> FactorCounts = new(capacity: 8);
		public ExponentRemainderStepperCpu ExponentRemainderStepper;

		private PartialFactorResult? _next;
		public ulong Cofactor;
		public bool CofactorIsPrime;
		public int Count;
		public bool FullyFactored;
		public bool HasFactors;

		private PartialFactorResult(in MontgomeryDivisorData divisorData)
		{
			ExponentRemainderStepper = new(divisorData);
		}

		public static PartialFactorResult Rent(in MontgomeryDivisorData divisorData)
		{
			if (s_poolHead is { } instance)
			{
				s_poolHead = instance._next;
				instance._next = null;
				instance.Cofactor = 0UL;
				instance.FullyFactored = false;
				instance.Count = 0;
				instance.CofactorIsPrime = false;
				instance.HasFactors = false;
				if (!instance.ExponentRemainderStepper.MatchesDivisor(divisorData, 0UL))
				{
					instance.ExponentRemainderStepper = new ExponentRemainderStepperCpu(divisorData, 0UL);					
				}

				return instance;
			}

			return new PartialFactorResult(divisorData);
		}

		public void WithAdditionalPrime(ulong prime)
		{
			ulong[] sourceFactors = Factors;
			int sourceCount = Count;
			if (sourceCount == 0)
			{
				InitializeWithPrime(prime);
				return;
			}

			int[] sourceExponents = Exponents;

			int newSize = sourceCount + 1;
			sourceFactors[sourceCount] = prime;
			sourceExponents[sourceCount] = 1;
			Array.Sort(sourceFactors, sourceExponents, 0, newSize);

			Cofactor = 1UL;
			FullyFactored = true;
			Count = newSize;
			CofactorIsPrime = false;
			HasFactors = true;
		}

		private void InitializeWithPrime(ulong prime)
		{
			Factors[0] = prime;
			Exponents[0] = 1;
			Cofactor = 1UL;
			FullyFactored = true;
			Count = 1;
			CofactorIsPrime = false;
			HasFactors = true;
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
