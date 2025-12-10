using System.Numerics;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu.Kernels;

namespace PerfectNumbers.Core.Gpu;

internal static partial class PrimeOrderKernels
{
	internal static void CheckFactorsKernel(
		int count,
		ulong phi,
		ArrayView1D<KeyValuePair<ulong, int>, Stride1D.Dense> toTest,
		ulong divisorModulus,
		ulong divisorNPrime,
		ulong divisorMontgomeryOne,
		ulong divisorMontgomeryTwo,
		ulong divisorMontgomeryTwoSquared,
		ArrayView1D<ulong, Stride1D.Dense> result
	)
	{
		ulong order = phi;
		int entryCount = count;
		MontgomeryDivisorDataGpu divisorData = new(divisorModulus, divisorNPrime, divisorMontgomeryOne, divisorMontgomeryTwo, divisorMontgomeryTwoSquared);

		for (int i = 0; i < entryCount; i++)
		{
			(ulong primeFactor, int exponent) = toTest[i];

			for (int iteration = 0; iteration < exponent; iteration++)
			{
				ulong candidate = order / primeFactor;
				ulong remainder = order - (candidate * primeFactor);
				// ulong remainder = order % primeFactor;
				if (remainder != 0UL)
				{
					break;
				}

				// ulong candidate = order / primeFactor;
				if (divisorData.Pow2MontgomeryModWindowedGpuConvertToStandard(candidate) == 1UL)
				{
					order = candidate;
					continue;
				}

				break;
			}
		}

		result[0] = order;
	}

	/// This kernel always sets the result and operates within the allowed / required bounds, never accessing elements outside of them.
	/// It sets the value only when it needs to. The callers don't need to clear the output / input buffers.
	internal static void PartialFactorKernel(
		Index1D index,
		ArrayView1D<uint, Stride1D.Dense> primes,
		ArrayView1D<ulong, Stride1D.Dense> squares,
		int primeCount,
		int slotCount,
		ulong value,
		uint limit,
		ArrayView1D<ulong, Stride1D.Dense> factorsOut,
		ArrayView1D<int, Stride1D.Dense> exponentsOut,
		ArrayView1D<int, Stride1D.Dense> countOut,
		ArrayView1D<ulong, Stride1D.Dense> remainingOut,
		ArrayView1D<byte, Stride1D.Dense> fullyFactoredOut)
	{
		if (index != 0)
		{
			return;
		}

		uint effectiveLimit = limit == 0 ? uint.MaxValue : limit;
		ulong remainingLocal = value;
		int count = 0;

		for (int i = 0; i < primeCount && count < slotCount; i++)
		{
			uint primeCandidate = primes[i];
			if (primeCandidate > effectiveLimit)
			{
				break;
			}

			ulong primeSquare = squares[i];
			if (primeSquare != 0UL && primeSquare > remainingLocal)
			{
				break;
			}

			ulong primeValue = primeCandidate;
			if (primeValue == 0UL || (remainingLocal % primeValue) != 0UL)
			{
				continue;
			}

			int exponent = 0;
			do
			{
				remainingLocal /= primeValue;
				exponent++;
			}
			while ((remainingLocal % primeValue) == 0UL);

			factorsOut[count] = primeValue;
			exponentsOut[count] = exponent;
			count++;
		}

		countOut[0] = count;
		remainingOut[0] = remainingLocal;
		fullyFactoredOut[0] = remainingLocal == 1UL ? (byte)1 : (byte)0;
	}

	private readonly struct CandidateKey
	{
		public CandidateKey(int primary, long secondary, long tertiary)
		{
			Primary = primary;
			Secondary = secondary;
			Tertiary = tertiary;
		}

		public int Primary { get; }

		public long Secondary { get; }

		public long Tertiary { get; }

		public int CompareTo(CandidateKey other)
		{
			if (Primary != other.Primary)
			{
				return Primary.CompareTo(other.Primary);
			}

			if (Secondary != other.Secondary)
			{
				return Secondary.CompareTo(other.Secondary);
			}

			return Tertiary.CompareTo(other.Tertiary);
		}
	}

	/// This kernel doesn't always set the result of the first element. It sets the value only when it needs to. The callers
	/// must clean the output buffer before the call to get a deterministic result.
	internal static void CalculateOrderKernel(
		Index1D index,
		ulong prime,
		CalculateOrderKernelConfig config,
		ulong divisorModulus,
		ulong divisorNPrime,
		ulong divisorMontgomeryOne,
		ulong divisorMontgomeryTwo,
		ulong divisorMontgomeryTwoSquared,
		ArrayView1D<uint, Stride1D.Dense> primes,
		ArrayView1D<ulong, Stride1D.Dense> squares,
		int primeCount,
		OrderKernelBuffers buffers)
	{
		if (index != 0)
		{
			return;
		}

		// TODO: Modify this kernel to always set the output result so that callers never need to clear any buffers and remove
		// any existing cleaning of the buffers after that.
		ArrayView1D<ulong, Stride1D.Dense> phiFactors = buffers.PhiFactors;
		ArrayView1D<int, Stride1D.Dense> phiExponents = buffers.PhiExponents;
		ArrayView1D<ulong, Stride1D.Dense> workFactors = buffers.WorkFactors;
		ArrayView1D<int, Stride1D.Dense> workExponents = buffers.WorkExponents;
		ArrayView1D<ulong, Stride1D.Dense> candidates = buffers.Candidates;
		ArrayView1D<int, Stride1D.Dense> stackIndex = buffers.StackIndex;
		ArrayView1D<int, Stride1D.Dense> stackExponent = buffers.StackExponent;
		ArrayView1D<ulong, Stride1D.Dense> stackProduct = buffers.StackProduct;
		ArrayView1D<ulong, Stride1D.Dense> resultOut = buffers.Result;
		ArrayView1D<byte, Stride1D.Dense> statusOut = buffers.Status;

		uint limit = config.SmallFactorLimit;
		ulong previousOrder = config.PreviousOrder;
		byte hasPreviousOrder = config.HasPreviousOrder;
		int maxPowChecks = config.MaxPowChecks;
		int mode = config.Mode;

		statusOut[0] = (byte)PrimeOrderKernelStatus.Fallback;

		// TODO: Is this condition ever satisfied on EventPerfectBitScanner's execution paths? Comment it out with explanatory
		// comment before it, if not. Otherwise add a comment explaining how it's used. If tests or benchmarks require this,
		// modify such tests and benchmark to avoid that.
		if (prime <= 3UL)
		{
			ulong orderValue = prime == 3UL ? 2UL : 1UL;
			resultOut[0] = orderValue;
			statusOut[0] = (byte)PrimeOrderKernelStatus.Found;
			return;
		}

		ulong phi = prime - 1UL;

		int phiFactorCount = FactorWithSmallPrimes(phi, limit, primes, squares, primeCount, phiFactors, phiExponents, out ulong phiRemaining);
		if (phiRemaining != 1UL)
		{
			// Reuse stackProduct as the Pollard Rho stack so factoring stays within this kernel.
			if (!TryFactorWithPollardKernel(
					phiRemaining,
					limit,
					primes,
					squares,
					primeCount,
					phiFactors,
					phiExponents,
					ref phiFactorCount,
					stackProduct,
					statusOut))
			{
				if (statusOut[0] != (byte)PrimeOrderKernelStatus.PollardOverflow)
				{
					resultOut[0] = CalculateByDoublingKernel(prime);
				}

				return;
			}
		}

		SortFactors(phiFactors, phiExponents, phiFactorCount);

		MontgomeryDivisorDataGpu divisor = new(divisorModulus, divisorNPrime, divisorMontgomeryOne, divisorMontgomeryTwo, divisorMontgomeryTwoSquared);

		if (TrySpecialMaxKernel(phi, prime, phiFactors, phiFactorCount, divisor))
		{
			resultOut[0] = phi;
			statusOut[0] = (byte)PrimeOrderKernelStatus.Found;
			return;
		}

		ulong candidateOrder = InitializeStartingOrderKernel(prime, phi, divisor);
		candidateOrder = ExponentLoweringKernel(candidateOrder, prime, phiFactors, phiExponents, phiFactorCount, divisor);

		if (TryConfirmOrderKernel(
				prime,
				candidateOrder,
				divisor,
				limit,
				primes,
				squares,
				primeCount,
				workFactors,
				workExponents,
				stackProduct,
				statusOut))
		{
			resultOut[0] = candidateOrder;
			statusOut[0] = (byte)PrimeOrderKernelStatus.Found;
			return;
		}

		bool isStrict = mode == 1;
		if (isStrict)
		{
			ulong strictOrder = CalculateByDoublingKernel(prime);
			resultOut[0] = strictOrder;
			statusOut[0] = (byte)PrimeOrderKernelStatus.Found;
			return;
		}

		if (TryHeuristicFinishKernel(
				prime,
				candidateOrder,
				previousOrder,
				hasPreviousOrder,
				divisor,
				limit,
				maxPowChecks,
				primes,
				squares,
				primeCount,
				workFactors,
				workExponents,
				candidates,
				stackIndex,
				stackExponent,
				stackProduct,
				statusOut,
				out ulong confirmedOrder))
		{
			resultOut[0] = confirmedOrder;
			statusOut[0] = (byte)PrimeOrderKernelStatus.Found;
			return;
		}

		ulong fallbackOrder = CalculateByDoublingKernel(prime);
		resultOut[0] = fallbackOrder;
		statusOut[0] = (byte)PrimeOrderKernelStatus.HeuristicUnresolved;
	}

	private static int FactorWithSmallPrimes(
		ulong value,
		uint limit,
		ArrayView1D<uint, Stride1D.Dense> primes,
		ArrayView1D<ulong, Stride1D.Dense> squares,
		int primeCount,
		ArrayView1D<ulong, Stride1D.Dense> factors,
		ArrayView1D<int, Stride1D.Dense> exponents,
		out ulong remaining)
	{
		remaining = value;
		int factorCount = 0;
		long factorLength = factors.Length;
		long exponentLength = exponents.Length;
		int capacity = factorLength < exponentLength ? (int)factorLength : (int)exponentLength;

		for (int i = 0; i < primeCount && remaining > 1UL && factorCount < capacity; i++)
		{
			uint primeCandidate = primes[i];
			if (primeCandidate == 0U || primeCandidate > limit)
			{
				break;
			}

			ulong square = squares[i];
			if (square != 0UL && square > remaining)
			{
				break;
			}

			ulong primeValue = primeCandidate;
			if ((remaining % primeValue) != 0UL)
			{
				continue;
			}

			int exponent = 0;
			do
			{
				remaining /= primeValue;
				exponent++;
			}
			while ((remaining % primeValue) == 0UL);

			factors[factorCount] = primeValue;
			exponents[factorCount] = exponent;
			factorCount++;
		}

		for (int i = factorCount; i < factors.Length; i++)
		{
			factors[i] = 0UL;
		}

		for (int i = factorCount; i < exponents.Length; i++)
		{
			exponents[i] = 0;
		}

		return factorCount;
	}

	private static void SortFactors(ArrayView1D<ulong, Stride1D.Dense> factors, ArrayView1D<int, Stride1D.Dense> exponents, int count)
	{
		for (int i = 1; i < count; i++)
		{
			ulong factor = factors[i];
			int exponent = exponents[i];
			int j = i - 1;

			while (j >= 0 && factors[j] > factor)
			{
				factors[j + 1] = factors[j];
				exponents[j + 1] = exponents[j];
				j--;
			}

			factors[j + 1] = factor;
			exponents[j + 1] = exponent;
		}
	}

	private static bool TrySpecialMaxKernel(
		ulong phi,
		ulong prime,
		ArrayView1D<ulong, Stride1D.Dense> factors,
		int factorCount,
		in MontgomeryDivisorDataGpu divisor)
	{
		for (int i = 0; i < factorCount; i++)
		{
			ulong factor = factors[i];
			if (factor <= 1UL)
			{
				continue;
			}

			ulong reduced = phi / factor;
			if (divisor.Pow2MontgomeryModWindowedGpuConvertToStandard(reduced) == 1UL)
			{
				return false;
			}
		}

		return true;
	}

	private static ulong InitializeStartingOrderKernel(ulong prime, ulong phi, in MontgomeryDivisorDataGpu divisor)
	{
		ulong order = phi;
		ulong residue = prime & 7UL;
		if (residue == 1UL || residue == 7UL)
		{
			ulong half = phi >> 1;
			if (Pow2EqualsOneKernel(half, divisor))
			{
				order = half;
			}
		}

		return order;
	}

	private static ulong ExponentLoweringKernel(
		ulong order,
		ulong prime,
		ArrayView1D<ulong, Stride1D.Dense> factors,
		ArrayView1D<int, Stride1D.Dense> exponents,
		int factorCount,
		in MontgomeryDivisorDataGpu divisor)
	{
		for (int i = 0; i < factorCount; i++)
		{
			ulong primeFactor = factors[i];
			int exponent = exponents[i];
			if (primeFactor <= 1UL)
			{
				continue;
			}

			for (int iteration = 0; iteration < exponent; iteration++)
			{
				if ((order % primeFactor) != 0UL)
				{
					break;
				}

				ulong reduced = order / primeFactor;
				if (Pow2EqualsOneKernel(reduced, divisor))
				{
					order = reduced;
					continue;
				}

				break;
			}
		}

		return order;
	}

	private static bool TryConfirmOrderKernel(
		ulong prime,
		ulong order,
		in MontgomeryDivisorDataGpu divisor,
		uint limit,
		ArrayView1D<uint, Stride1D.Dense> primes,
		ArrayView1D<ulong, Stride1D.Dense> squares,
		int primeCount,
		ArrayView1D<ulong, Stride1D.Dense> factors,
		ArrayView1D<int, Stride1D.Dense> exponents,
		ArrayView1D<ulong, Stride1D.Dense> compositeStack,
		ArrayView1D<byte, Stride1D.Dense> statusOut)
	{
		if (order == 0UL)
		{
			return false;
		}

		if (!Pow2EqualsOneKernel(order, divisor))
		{
			return false;
		}

		int factorCount = FactorWithSmallPrimes(order, limit, primes, squares, primeCount, factors, exponents, out ulong remaining);
		if (remaining != 1UL)
		{
			if (!TryFactorWithPollardKernel(
					remaining,
					limit,
					primes,
					squares,
					primeCount,
					factors,
					exponents,
					ref factorCount,
					compositeStack,
					statusOut))
			{
				return false;
			}
		}

		SortFactors(factors, exponents, factorCount);

		for (int i = 0; i < factorCount; i++)
		{
			ulong primeFactor = factors[i];
			int exponent = exponents[i];
			if (primeFactor <= 1UL)
			{
				continue;
			}

			ulong reduced = order;
			for (int iteration = 0; iteration < exponent; iteration++)
			{
				if ((reduced % primeFactor) != 0UL)
				{
					break;
				}

				reduced /= primeFactor;
				if (Pow2EqualsOneKernel(reduced, divisor))
				{
					return false;
				}
			}
		}

		return true;
	}

	private static bool TryHeuristicFinishKernel(
		ulong prime,
		ulong order,
		ulong previousOrder,
		byte hasPreviousOrder,
		in MontgomeryDivisorDataGpu divisor,
		uint limit,
		int maxPowChecks,
		ArrayView1D<uint, Stride1D.Dense> primes,
		ArrayView1D<ulong, Stride1D.Dense> squares,
		int primeCount,
		ArrayView1D<ulong, Stride1D.Dense> workFactors,
		ArrayView1D<int, Stride1D.Dense> workExponents,
		ArrayView1D<ulong, Stride1D.Dense> candidates,
		ArrayView1D<int, Stride1D.Dense> stackIndex,
		ArrayView1D<int, Stride1D.Dense> stackExponent,
		ArrayView1D<ulong, Stride1D.Dense> stackProduct,
		ArrayView1D<byte, Stride1D.Dense> statusOut,
		out ulong confirmedOrder)
	{
		confirmedOrder = 0UL;

		if (order <= 1UL)
		{
			return false;
		}

		int factorCount = FactorWithSmallPrimes(order, limit, primes, squares, primeCount, workFactors, workExponents, out ulong remaining);
		if (remaining != 1UL)
		{
			// Reuse stackProduct as the Pollard Rho stack while factoring the order candidates.
			if (!TryFactorWithPollardKernel(
					remaining,
					limit,
					primes,
					squares,
					primeCount,
					workFactors,
					workExponents,
					ref factorCount,
					stackProduct,
					statusOut))
			{
				return false;
			}
		}

		SortFactors(workFactors, workExponents, factorCount);

		long candidateCapacity = candidates.Length;
		int candidateLimit = candidateCapacity < PrimeOrderConstants.HeuristicCandidateLimit ? (int)candidateCapacity : PrimeOrderConstants.HeuristicCandidateLimit;
		int candidateCount = BuildCandidatesKernel(order, workFactors, workExponents, factorCount, candidates, stackIndex, stackExponent, stackProduct, candidateLimit);
		if (candidateCount == 0)
		{
			return false;
		}

		SortCandidatesKernel(prime, previousOrder, hasPreviousOrder != 0, candidates, candidateCount);

		int powBudget = maxPowChecks <= 0 ? candidateCount : maxPowChecks;
		if (powBudget <= 0)
		{
			powBudget = candidateCount;
		}

		int powUsed = 0;

		for (int i = 0; i < candidateCount && powUsed < powBudget; i++)
		{
			ulong candidate = candidates[i];
			if (candidate <= 1UL)
			{
				continue;
			}

			if (powUsed >= powBudget)
			{
				break;
			}

			powUsed++;
			if (!Pow2EqualsOneKernel(candidate, divisor))
			{
				continue;
			}

			if (!TryConfirmCandidateKernel(
					prime,
					candidate,
					divisor,
					limit,
					primes,
					squares,
					primeCount,
					workFactors,
					workExponents,
					stackProduct,
					statusOut,
					ref powUsed,
					powBudget))
			{
				continue;
			}

			confirmedOrder = candidate;
			return true;
		}

		return false;
	}

	private static int BuildCandidatesKernel(
		ulong order,
		ArrayView1D<ulong, Stride1D.Dense> factors,
		ArrayView1D<int, Stride1D.Dense> exponents,
		int factorCount,
		ArrayView1D<ulong, Stride1D.Dense> candidates,
		ArrayView1D<int, Stride1D.Dense> stackIndex,
		ArrayView1D<int, Stride1D.Dense> stackExponent,
		ArrayView1D<ulong, Stride1D.Dense> stackProduct,
		int limit)
	{
		long stackIndexLength = stackIndex.Length;
		long stackExponentLength = stackExponent.Length;
		long stackProductLength = stackProduct.Length;
		if (factorCount == 0 || limit <= 0 || stackIndexLength == 0L || stackExponentLength == 0L || stackProductLength == 0L)
		{
			return 0;
		}

		int stackCapacity = stackIndexLength < stackExponentLength ? (int)stackIndexLength : (int)stackExponentLength;
		if (stackProductLength < stackCapacity)
		{
			stackCapacity = (int)stackProductLength;
		}

		int candidateCount = 0;
		int stackTop = 0;

		stackIndex[0] = 0;
		stackExponent[0] = 0;
		stackProduct[0] = 1UL;
		stackTop = 1;

		while (stackTop > 0)
		{
			stackTop--;
			int index = stackIndex[stackTop];
			int exponent = stackExponent[stackTop];
			ulong product = stackProduct[stackTop];

			if (index >= factorCount)
			{
				if (product != 1UL && product != order && candidateCount < limit)
				{
					ulong candidate = order / product;
					if (candidate > 1UL && candidate < order)
					{
						candidates[candidateCount] = candidate;
						candidateCount++;
					}
				}

				continue;
			}

			int maxExponent = exponents[index];
			if (exponent > maxExponent)
			{
				continue;
			}

			if (stackTop >= stackCapacity)
			{
				return candidateCount;
			}

			stackIndex[stackTop] = index + 1;
			stackExponent[stackTop] = 0;
			stackProduct[stackTop] = product;
			stackTop++;

			if (exponent == maxExponent)
			{
				continue;
			}

			ulong primeFactor = factors[index];
			if (primeFactor == 0UL || product > order / primeFactor)
			{
				continue;
			}

			if (stackTop >= stackCapacity)
			{
				return candidateCount;
			}

			stackIndex[stackTop] = index;
			stackExponent[stackTop] = exponent + 1;
			stackProduct[stackTop] = product * primeFactor;
			stackTop++;
		}

		return candidateCount;
	}

	private static void SortCandidatesKernel(
		ulong prime,
		ulong previousOrder,
		bool hasPrevious,
		ArrayView1D<ulong, Stride1D.Dense> candidates,
		int count)
	{
		for (int i = 1; i < count; i++)
		{
			ulong value = candidates[i];
			CandidateKey key = BuildCandidateKey(value, prime, previousOrder, hasPrevious);
			int j = i - 1;

			while (j >= 0)
			{
				CandidateKey other = BuildCandidateKey(candidates[j], prime, previousOrder, hasPrevious);
				if (other.CompareTo(key) <= 0)
				{
					break;
				}

				candidates[j + 1] = candidates[j];
				j--;
			}

			candidates[j + 1] = value;
		}
	}

	private static CandidateKey BuildCandidateKey(ulong value, ulong prime, ulong previousOrder, bool hasPrevious)
	{
		int group = GetGroup(value, prime);
		if (group == 0)
		{
			return new CandidateKey(int.MaxValue, long.MaxValue, long.MaxValue);
		}

		ulong reference = hasPrevious ? previousOrder : 0UL;
		bool isGe = !hasPrevious || value >= reference;
		int previousGroup = hasPrevious ? GetGroup(reference, prime) : 1;
		int primary = ComputePrimary(group, isGe, previousGroup);
		long secondary;
		long tertiary;

		if (group == 3)
		{
			secondary = -(long)value;
			tertiary = -(long)value;
		}
		else
		{
			ulong distance = hasPrevious ? (value > reference ? value - reference : reference - value) : value;
			secondary = (long)distance;
			tertiary = (long)value;
		}

		return new CandidateKey(primary, secondary, tertiary);
	}

	private static int GetGroup(ulong value, ulong prime)
	{
		ulong threshold1 = prime >> 3;
		if (value <= threshold1)
		{
			return 1;
		}

		ulong threshold2 = prime >> 2;
		if (value <= threshold2)
		{
			return 2;
		}

		ulong threshold3 = (prime * 3UL) >> 3;
		if (value <= threshold3)
		{
			return 3;
		}

		return 0;
	}

	private static int ComputePrimary(int group, bool isGe, int previousGroup)
	{
		int groupOffset;
		switch (group)
		{
			case 1:
				groupOffset = 0;
				break;
			case 2:
				groupOffset = 2;
				break;
			case 3:
				groupOffset = 4;
				break;
			default:
				groupOffset = 6;
				break;
		}

		if (group == previousGroup)
		{
			if (group == 3)
			{
				return groupOffset + (isGe ? 0 : 3);
			}

			return groupOffset + (isGe ? 0 : 1);
		}

		return groupOffset + (isGe ? 0 : 1);
	}

	private static bool TryConfirmCandidateKernel(
		ulong prime,
		ulong candidate,
		in MontgomeryDivisorDataGpu divisor,
		uint limit,
		ArrayView1D<uint, Stride1D.Dense> primes,
		ArrayView1D<ulong, Stride1D.Dense> squares,
		int primeCount,
		ArrayView1D<ulong, Stride1D.Dense> factors,
		ArrayView1D<int, Stride1D.Dense> exponents,
		ArrayView1D<ulong, Stride1D.Dense> compositeStack,
		ArrayView1D<byte, Stride1D.Dense> statusOut,
		ref int powUsed,
		int powBudget)
	{
		int factorCount = FactorWithSmallPrimes(candidate, limit, primes, squares, primeCount, factors, exponents, out ulong remaining);
		if (remaining != 1UL)
		{
			if (!TryFactorWithPollardKernel(
					remaining,
					limit,
					primes,
					squares,
					primeCount,
					factors,
					exponents,
					ref factorCount,
					compositeStack,
					statusOut))
			{
				return false;
			}
		}

		SortFactors(factors, exponents, factorCount);

		for (int i = 0; i < factorCount; i++)
		{
			ulong primeFactor = factors[i];
			int exponent = exponents[i];
			if (primeFactor <= 1UL)
			{
				continue;
			}

			ulong reduced = candidate;
			for (int iteration = 0; iteration < exponent; iteration++)
			{
				if ((reduced % primeFactor) != 0UL)
				{
					break;
				}

				reduced /= primeFactor;
				if (powBudget > 0 && powUsed >= powBudget)
				{
					return false;
				}

				powUsed++;
				if (Pow2EqualsOneKernel(reduced, divisor))
				{
					return false;
				}
			}
		}

		return true;
	}

	private static bool TryFactorWithPollardKernel(
		ulong initial,
		uint limit,
		ArrayView1D<uint, Stride1D.Dense> primes,
		ArrayView1D<ulong, Stride1D.Dense> squares,
		int primeCount,
		ArrayView1D<ulong, Stride1D.Dense> factors,
		ArrayView1D<int, Stride1D.Dense> exponents,
		ref int factorCount,
		ArrayView1D<ulong, Stride1D.Dense> compositeStack,
		ArrayView1D<byte, Stride1D.Dense> statusOut)
	{
		if (initial <= 1UL)
		{
			return true;
		}

		int stackCapacity = (int)compositeStack.Length;
		if (stackCapacity <= 0)
		{
			statusOut[0] = (byte)PrimeOrderKernelStatus.PollardOverflow;
			return false;
		}

		int stackTop = 0;
		compositeStack[stackTop] = initial;
		stackTop++;

		while (stackTop > 0)
		{
			stackTop--;
			ulong composite = compositeStack[stackTop];
			if (composite <= 1UL)
			{
				continue;
			}

			if (!PeelSmallPrimesKernel(
					composite,
					limit,
					primes,
					squares,
					primeCount,
					factors,
					exponents,
					ref factorCount,
					out ulong remaining))
			{
				statusOut[0] = (byte)PrimeOrderKernelStatus.PollardOverflow;
				return false;
			}

			if (remaining == 1UL)
			{
				continue;
			}

			ulong factor = PollardRhoKernel(remaining);
			if (factor <= 1UL || factor == remaining)
			{
				if (!TryAppendFactorKernel(factors, exponents, ref factorCount, remaining, 1))
				{
					statusOut[0] = (byte)PrimeOrderKernelStatus.PollardOverflow;
					return false;
				}

				continue;
			}

			ulong quotient = remaining / factor;
			if (stackTop + 2 > stackCapacity)
			{
				statusOut[0] = (byte)PrimeOrderKernelStatus.PollardOverflow;
				return false;
			}

			compositeStack[stackTop] = factor;
			compositeStack[stackTop + 1] = quotient;
			stackTop += 2;
		}

		return true;
	}

	private static bool PeelSmallPrimesKernel(
		ulong value,
		uint limit,
		ArrayView1D<uint, Stride1D.Dense> primes,
		ArrayView1D<ulong, Stride1D.Dense> squares,
		int primeCount,
		ArrayView1D<ulong, Stride1D.Dense> factors,
		ArrayView1D<int, Stride1D.Dense> exponents,
		ref int factorCount,
		out ulong remaining)
	{
		ulong remainingLocal = value;

		for (int i = 0; i < primeCount && remainingLocal > 1UL; i++)
		{
			uint primeCandidate = primes[i];
			if (primeCandidate == 0U || primeCandidate > limit)
			{
				break;
			}

			ulong square = squares[i];
			if (square != 0UL && square > remainingLocal)
			{
				break;
			}

			ulong primeValue = primeCandidate;
			if ((remainingLocal % primeValue) != 0UL)
			{
				continue;
			}

			int exponent = 0;
			do
			{
				remainingLocal /= primeValue;
				exponent++;
			}
			while ((remainingLocal % primeValue) == 0UL);

			if (!TryAppendFactorKernel(factors, exponents, ref factorCount, primeValue, exponent))
			{
				remaining = value;
				return false;
			}
		}

		remaining = remainingLocal;
		return true;
	}

	private static bool TryAppendFactorKernel(
		ArrayView1D<ulong, Stride1D.Dense> factors,
		ArrayView1D<int, Stride1D.Dense> exponents,
		ref int count,
		ulong prime,
		int exponent)
	{
		if (prime <= 1UL || exponent <= 0)
		{
			return true;
		}

		for (int i = 0; i < count; i++)
		{
			if (factors[i] == prime)
			{
				exponents[i] += exponent;
				return true;
			}
		}

		int capacity = (int)factors.Length;
		if (count >= capacity)
		{
			return false;
		}

		factors[count] = prime;
		exponents[count] = exponent;
		count++;
		return true;
	}

	private static ulong MulModKernel(ulong left, ulong right, ulong modulus)
	{
		GpuUInt128 product = new GpuUInt128(left);
		return product.MulMod(right, modulus);
	}

	private static ulong PollardRhoKernel(ulong value)
	{
		if ((value & 1UL) == 0UL)
		{
			return 2UL;
		}

		ulong c = 1UL;
		while (true)
		{
			ulong x = 2UL;
			ulong y = 2UL;
			ulong d = 1UL;

			while (d == 1UL)
			{
				x = AdvancePolynomialKernel(x, c, value);
				y = AdvancePolynomialKernel(y, c, value);
				y = AdvancePolynomialKernel(y, c, value);

				ulong diff = x > y ? x - y : y - x;
				d = BinaryGcdKernel(diff, value);
			}

			if (d == value)
			{
				c++;
				if (c == 0UL)
				{
					c = 1UL;
				}

				continue;
			}

			return d;
		}
	}

	private static ulong AdvancePolynomialKernel(ulong x, ulong c, ulong modulus)
	{
		ulong squared = MulModKernel(x, x, modulus);
		GpuUInt128 accumulator = new GpuUInt128(squared);
		accumulator.AddMod(c, modulus);
		return accumulator.Low;
	}

	private static ulong BinaryGcdKernel(ulong a, ulong b)
	{
		if (a == 0UL)
		{
			return b;
		}

		if (b == 0UL)
		{
			return a;
		}

		int shift = BitOperations.TrailingZeroCount(a | b);
		ulong aLocal = a >> BitOperations.TrailingZeroCount(a);
		ulong bLocal = b;

		while (true)
		{
			bLocal >>= BitOperations.TrailingZeroCount(bLocal);
			if (aLocal > bLocal)
			{
				ulong temp = aLocal;
				aLocal = bLocal;
				bLocal = temp;
			}

			bLocal -= aLocal;
			if (bLocal == 0UL)
			{
				return aLocal << shift;
			}
		}
	}

	private static bool Pow2EqualsOneKernel(ulong exponent, in MontgomeryDivisorDataGpu divisor)
	{
		return divisor.Pow2MontgomeryModWindowedGpuConvertToStandard(exponent) == 1UL;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static ulong CalculateByDoublingKernel(ulong prime)
	{
		ulong order = 1UL;
		ulong value = 2UL % prime;

		while (value != 1UL)
		{
			value <<= 1;
			if (value >= prime)
			{
				value -= prime;
			}

			order++;
		}

		return order;
	}

	/// This kernel always sets the result of the corresponding element. Callers don't need to clear the output buffers.

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	internal static void Pow2ModKernel(Index1D index, ArrayView1D<ulong, Stride1D.Dense> exponents, ulong divisorModulus, ulong divisorNPrime, ulong divisorMontgomeryOne, ulong divisorMontgomeryTwo, ulong divisorMontgomeryTwoSquared, ArrayView1D<ulong, Stride1D.Dense> remainders)
	{
		ulong exponent = exponents[index];
		MontgomeryDivisorDataGpu divisor = new(divisorModulus, divisorNPrime, divisorMontgomeryOne, divisorMontgomeryTwo, divisorMontgomeryTwoSquared);
		remainders[index] = divisor.Pow2MontgomeryModWindowedGpuConvertToStandard(exponent);
	}

	/// This kernel always sets the result of the corresponding element. Callers don't need to clear the output buffers.
	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	internal static void Pow2ModKernelWide(Index1D index, ArrayView1D<GpuUInt128, Stride1D.Dense> exponents, GpuUInt128 modulus, ArrayView1D<GpuUInt128, Stride1D.Dense> remainders)
	{
		GpuUInt128 exponent = exponents[index];
		remainders[index] = Pow2ModKernelCore(exponent, modulus);
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static GpuUInt128[] InitializeOddPowersTable(GpuUInt128 baseValue, GpuUInt128 modulus, int oddPowerCount)
	{
		GpuUInt128[] result = new GpuUInt128[PerfectNumberConstants.MaxOddPowersCount];
		result[0] = baseValue;
		if (oddPowerCount == 1)
		{
			return result;
		}

		// Reusing baseValue to hold base^2 for the shared odd-power ladder that follows.
		baseValue.MulMod(baseValue, modulus);

		// TODO: We can calculate baseValue % modulus before loop and use it to increase ladderEntry calculation speed - we'll reuse the base for incremental calculations.
		GpuUInt128 current = baseValue;

		// We're manually assigning each field to prevent the compiler to initialize each field twice due to auto-initialization. We're using the action to lower the code base size.
		for (int i = 1; i < oddPowerCount; i++)
		{
			current.MulMod(baseValue, modulus);
			result[i] = current;
		}

		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	internal static GpuUInt128 Pow2ModKernelCore(GpuUInt128 exponent, GpuUInt128 modulus)
	{
		// This should never happen in production code.
		// if (modulus == GpuUInt128.One)
		// {
		//     return GpuUInt128.Zero;
		// }

		// This should never happen in production code.
		// if (exponent.IsZero)
		// {
		//     return GpuUInt128.One;
		// }

		GpuUInt128 baseValue = UInt128Numbers.TwoGpu;

		// This should never happen in production code - 2 should never be greater or equal modulus.
		// if (baseValue.CompareTo(modulus) >= 0)
		// {
		//     baseValue.Sub(modulus);
		// }

		if (ShouldUseSingleBit(exponent))
		{
			return Pow2MontgomeryModSingleBit(exponent, modulus, baseValue);
		}

		int index = exponent.GetBitLength();
		int windowSize = GetWindowSize(index);
		index--;

		int oddPowerCount = 1 << (windowSize - 1);
		GpuUInt128[] oddPowers = InitializeOddPowersTable(baseValue, modulus, oddPowerCount);
		GpuUInt128 result = GpuUInt128.One;

		int windowStart;
		ulong windowValue;
		while (index >= 0)
		{
			if (!IsBitSet(exponent, index))
			{
				result.MulMod(result, modulus);
				index--;
				continue;
			}

			windowStart = index - windowSize + 1;
			if (windowStart < 0)
			{
				windowStart = 0;
			}

			while (!IsBitSet(exponent, windowStart))
			{
				windowStart++;
			}

			// We're reusing oddPowerCount as windowBitCount here to lower registry pressure & avoid additional allocation
			oddPowerCount = index - windowStart + 1;
			index = windowStart - 1;
			windowValue = ExtractWindowValue(exponent, windowStart, oddPowerCount);

			// We're reusing windowStart as square here to lower registry pressure & avoid additional allocation
			for (windowStart = 0; windowStart < oddPowerCount; windowStart++)
			{
				result.MulMod(result, modulus);
			}

			// We're reusing windowStart as tableIndex here to lower registry pressure & avoid additional allocation
			windowStart = (int)((windowValue - 1UL) >> 1);
			result.MulMod(oddPowers[windowStart], modulus);
		}

		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool ShouldUseSingleBit(GpuUInt128 exponent) => exponent.High == 0UL && exponent.Low <= PrimeOrderConstants.Pow2WindowFallbackThreshold;

	private static int GetWindowSize(int bitLength)
	{
		if (bitLength <= PrimeOrderConstants.Pow2WindowSizeBits)
		{
			return Math.Max(bitLength, 1);
		}

		if (bitLength <= 23)
		{
			return 4;
		}

		if (bitLength <= 79)
		{
			return 5;
		}

		if (bitLength <= 239)
		{
			return 6;
		}

		if (bitLength <= 671)
		{
			return 7;
		}

		return PrimeOrderConstants.Pow2WindowSizeBits;
	}

	private static GpuUInt128 Pow2MontgomeryModSingleBit(GpuUInt128 exponent, GpuUInt128 modulus, GpuUInt128 baseValue)
	{
		GpuUInt128 result = GpuUInt128.One;

		while (!exponent.IsZero)
		{
			if ((exponent.Low & 1UL) != 0UL)
			{
				result.MulMod(baseValue, modulus);
			}

			exponent.ShiftRight(1);
			if (exponent.IsZero)
			{
				break;
			}

			// Reusing baseValue to store the squared base for the next iteration.
			baseValue.MulMod(baseValue, modulus);
		}

		return result;
	}

	private static bool IsBitSet(GpuUInt128 value, int bitIndex)
	{
		if (bitIndex >= 64)
		{
			return ((value.High >> (bitIndex - 64)) & 1UL) != 0UL;
		}

		return ((value.Low >> bitIndex) & 1UL) != 0UL;
	}

	private static ulong ExtractWindowValue(GpuUInt128 exponent, int windowStart, int windowBitCount)
	{
		if (windowStart != 0)
		{
			exponent.ShiftRight(windowStart);
			ulong mask = (1UL << windowBitCount) - 1UL;
			return exponent.Low & mask;
		}

		ulong directMask = (1UL << windowBitCount) - 1UL;
		return exponent.Low & directMask;
	}
}
