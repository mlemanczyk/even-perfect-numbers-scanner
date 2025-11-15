using System.Buffers;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using PerfectNumbers.Core.Cpu;
using PerfectNumbers.Core.Gpu;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core;

internal static partial class PrimeOrderCalculator
{

	internal readonly struct PendingEntry
	{
		public readonly ulong Value;
		public readonly bool KnownComposite;
		public readonly bool HasKnownPrimality;
		public readonly bool IsPrime;

		public PendingEntry(ulong value, bool knownComposite)
		{
			Value = value;
			KnownComposite = knownComposite;
			HasKnownPrimality = knownComposite;
			IsPrime = false;
		}

		private PendingEntry(ulong value, bool knownComposite, bool hasKnownPrimality, bool isPrime)
		{
			Value = value;
			KnownComposite = knownComposite;
			HasKnownPrimality = hasKnownPrimality;
			IsPrime = isPrime;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public PendingEntry WithPrimality(bool isPrime)
		{
			bool knownComposite = KnownComposite || !isPrime;
			return new PendingEntry(Value, knownComposite, true, isPrime);
		}
	}

	private static ulong CalculateInternal(ulong prime, ulong? previousOrder, in MontgomeryDivisorData divisorData, in PrimeOrderSearchConfig config)
	{
		// TODO: Is this condition ever met on EvenPerfectBitScanner's execution path? If not, we can add a clarification comment and comment out the entire block. We want to support p candidates at least greater or equal to 31.
		if (prime <= 3UL)
		{
			return prime == 3UL ? 2UL : 1UL;
		}

		ulong phi = prime - 1UL;

		PrimeOrderCalculatorAccelerator gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		if (IsGpuHeuristicDevice && PrimeOrderGpuHeuristics.TryCalculateOrder(prime, previousOrder, config, divisorData, out ulong gpuOrder))
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
			return gpuOrder;
		}

		PartialFactorResult phiFactors = PartialFactor(gpu, phi, config);

		ulong result;
		if (phiFactors.Factors is null)
		{
			result = prime <= PerfectNumberConstants.MaxQForDivisorCycles
				? CalculateByFactorizationGpu(gpu, prime, divisorData)
				: CalculateByFactorizationCpu(gpu, prime, divisorData);
				
			phiFactors.Dispose();
			PrimeOrderCalculatorAccelerator.Return(gpu);
			return result;
		}

		result = RunHeuristicPipelineCpu(gpu, prime, previousOrder, config, divisorData, phi, phiFactors);
		phiFactors.Dispose();
		PrimeOrderCalculatorAccelerator.Return(gpu);
		return result;
	}

	private static ulong RunHeuristicPipelineCpu(
		PrimeOrderCalculatorAccelerator gpu,
		ulong prime,
		ulong? previousOrder,
		in PrimeOrderSearchConfig config,
		in MontgomeryDivisorData divisorData,
		ulong phi,
		PartialFactorResult phiFactors)
	{
		if (phiFactors.FullyFactored && TrySpecialMaxCpu(gpu, phi, prime, phiFactors, divisorData))
		{
			return phi;
		}

		// GpuPrimeWorkLimiter.Acquire();
		ulong candidateOrder = InitializeStartingOrderCpu(gpu, prime, phi, divisorData);
		candidateOrder = ExponentLoweringCpu(candidateOrder, prime, phiFactors, divisorData);

		if (TryConfirmOrderCpu(gpu, prime, candidateOrder, divisorData, config, out PartialFactorResult? orderFactors))
		{
			// GpuPrimeWorkLimiter.Release();
			return candidateOrder;
		}

		if (config.Mode == PrimeOrderMode.Strict)
		{
			orderFactors?.Dispose();
			// GpuPrimeWorkLimiter.Release();
			return CalculateByFactorizationGpu(gpu, prime, divisorData);
		}

		if (TryHeuristicFinishCpu(gpu, prime, candidateOrder, previousOrder, divisorData, config, phiFactors, orderFactors, out ulong order))
		{
			// GpuPrimeWorkLimiter.Release();
			return order;
		}

		// GpuPrimeWorkLimiter.Release();
		return candidateOrder;
	}

	// private static ulong specialMaxHits;
	private static bool TrySpecialMaxCpu(PrimeOrderCalculatorAccelerator gpu, ulong phi, ulong prime, PartialFactorResult factors, in MontgomeryDivisorData divisorData)
	{
		int length = factors.Count;
		// The phi factorization on the scanner path always yields at least one entry for phi >= 2,
		// preventing the zero-length case from occurring outside synthetic tests.
		// if (length == 0)
		// {
		//     return true;
		// }

		ReadOnlySpan<ulong> factorSpan = new(factors.Factors, 0, length);

		Span<ulong> stackBuffer = stackalloc ulong[length];

		if (length <= 7)
		{
			return EvaluateSpecialMaxCandidates(stackBuffer, factorSpan, phi, prime, divisorData);
		}

		return EvaluateSpecialMaxCandidatesGpu(gpu, factorSpan, phi, prime, divisorData);
	}

	private static bool EvaluateSpecialMaxCandidates(Span<ulong> buffer, ReadOnlySpan<ulong> factors, ulong phi, ulong prime, in MontgomeryDivisorData divisorData)
	{
		int actual = 0;
		int factorCount = factors.Length;
		for (int i = 0; i < factorCount; i++)
		{
			ulong factor = factors[i];

			// The partial factor pipeline never feeds zero or oversized factors while scanning candidate orders.
			// if (factor == 0UL || factor > phi)
			// {
			// 	continue;
			// }

			ulong reduced = phi / factor;
			if (reduced == 0UL)
			{
				continue;
			}

			buffer[actual] = reduced;
			actual++;
		}

		if (actual == 0)
		{
			return true;
		}

		Span<ulong> candidates = buffer[..actual];
		candidates.Sort();

		ExponentRemainderStepperCpu stepper = ThreadStaticPools.RentExponentStepperCpu(divisorData);
		int candidateCount = candidates.Length;

		if (stepper.InitializeCpuIsUnity(candidates[0]))
		{
			ThreadStaticPools.ReturnExponentStepperCpu(stepper);
			return false;
		}

		for (int i = 1; i < candidateCount; i++)
		{
			if (stepper.ComputeNextIsUnity(candidates[i]))
			{
				ThreadStaticPools.ReturnExponentStepperCpu(stepper);
				return false;
			}
		}

		ThreadStaticPools.ReturnExponentStepperCpu(stepper);
		return true;
	}

	private static ulong InitializeStartingOrderCpu(PrimeOrderCalculatorAccelerator gpu, ulong prime, ulong phi, in MontgomeryDivisorData divisorData)
	{
		ulong order = phi;
		if ((prime & 7UL) == 1UL || (prime & 7UL) == 7UL)
		{
			ulong half = phi >> 1;
			if (Pow2EqualsOneCpu(gpu, half, prime, divisorData))
			{
				order = half;
			}
		}

		return order;
	}

	private static ulong ExponentLoweringCpu(ulong order, ulong prime, PartialFactorResult factors, in MontgomeryDivisorData divisorData)
	{
		ArrayPool<ulong> ulongPool = ThreadStaticPools.UlongPool;
		ArrayPool<int> intPool = ThreadStaticPools.IntPool;

		ReadOnlySpan<ulong> factorSpan = factors.Factors;
		ReadOnlySpan<int> exponentsSpan = factors.Exponents;

		int length = factors.Count;
		ulong[] tempFactorsArray = ulongPool.Rent(length + 1);
		int[] tempExponentsArray = intPool.Rent(length + 1);

		Span<ulong> ulongBuffer = tempFactorsArray;
		factorSpan.CopyTo(ulongBuffer);
		Span<int> intBuffer = tempExponentsArray;
		exponentsSpan.CopyTo(intBuffer);

		if (!factors.FullyFactored && factors.Cofactor > 1UL && factors.CofactorIsPrime)
		{
			ulongBuffer[length] = factors.Cofactor;
			intBuffer[length] = 1;
			length++;
		}

		Array.Sort(tempFactorsArray, tempExponentsArray, 0, length);

		ExponentRemainderStepperCpu stepper = ThreadStaticPools.RentExponentStepperCpu(divisorData);

		const int StackExponentCapacity = 16;
		const int ExponentHardLimit = 256;

		Span<ulong> stackCandidates = stackalloc ulong[StackExponentCapacity];
		Span<bool> stackEvaluations = stackalloc bool[StackExponentCapacity];

		ulong[]? heapCandidateArray = null;
		bool[]? heapEvaluationArray = null;

		for (int i = 0; i < length; i++)
		{
			ulong primeFactor = ulongBuffer[i];
			int exponent = intBuffer[i];
			// Factor exponents produced by Pollard-Rho and trial division are always positive in the scanner flow.
			// if (exponent <= 0)
			// {
			// 	continue;
			// }

			if (exponent > ExponentHardLimit)
			{
				throw new InvalidOperationException($"Prime factor exponent {exponent} exceeds the supported limit of {ExponentHardLimit}.");
			}

			if (exponent <= StackExponentCapacity)
			{
				ProcessExponentLoweringPrime(stackCandidates[..exponent], stackEvaluations[..exponent], ref order, primeFactor, exponent, ref stepper);
				continue;
			}

			heapCandidateArray ??= ThreadStaticPools.UlongPool.Rent(ExponentHardLimit);
			heapEvaluationArray ??= ThreadStaticPools.BoolPool.Rent(ExponentHardLimit);
			ProcessExponentLoweringPrime(heapCandidateArray.AsSpan(0, exponent), heapEvaluationArray.AsSpan(0, exponent), ref order, primeFactor, exponent, ref stepper);
		}

		ThreadStaticPools.ReturnExponentStepperCpu(stepper);

		if (heapCandidateArray is not null)
		{
			ThreadStaticPools.UlongPool.Return(heapCandidateArray);
			ThreadStaticPools.BoolPool.Return(heapEvaluationArray!);
		}

		ulongPool.Return(tempFactorsArray, clearArray: false);
		return order;
	}

	private static void ProcessExponentLoweringPrime(Span<ulong> candidateBuffer, Span<bool> evaluationBuffer, ref ulong order, ulong primeFactor, int exponent, ref ExponentRemainderStepperCpu stepper)
	{
		ulong working = order;
		int actual = 0;
		while (actual < exponent)
		{
			if (working < primeFactor)
			{
				break;
			}

			ulong reduced = working / primeFactor;
			if (reduced == 0UL)
			{
				break;
			}

			candidateBuffer[actual] = reduced;
			working = reduced;
			actual++;
		}

		if (actual == 0)
		{
			return;
		}

		Span<ulong> candidates = candidateBuffer[..actual];
		Span<bool> evaluations = evaluationBuffer[..actual];

		stepper.Reset();

		int last = actual - 1;
		evaluations[last] = stepper.InitializeCpuIsUnity(candidates[last]);
		for (int j = last - 1; j >= 0; j--)
		{
			evaluations[j] = stepper.ComputeNextIsUnity(candidates[j]);
		}

		for (int j = 0; j < actual; j++)
		{
			if (!evaluations[j])
			{
				break;
			}

			order = candidates[j];
		}
	}

	private static bool TryConfirmOrderCpu(
		PrimeOrderCalculatorAccelerator gpu,
		ulong prime,
		ulong order,
		in MontgomeryDivisorData divisorData,
		in PrimeOrderSearchConfig config,
		out PartialFactorResult? reusableFactorization)
	{
		reusableFactorization = null;

		if (order == 0UL)
		{
			return false;
		}

		if (!Pow2EqualsOneCpu(gpu, order, prime, divisorData))
		{
			return false;
		}

		PartialFactorResult factorization = PartialFactor(gpu, order, config);
		if (factorization.Factors is null)
		{
			factorization.Dispose();
			return false;
		}

		if (!factorization.FullyFactored)
		{
			if (factorization.Cofactor <= 1UL || !factorization.CofactorIsPrime)
			{
				factorization.Dispose();
				return false;
			}

			PartialFactorResult extended = factorization.WithAdditionalPrime(factorization.Cofactor);
			factorization.Dispose();
			factorization = extended;
		}

		if (!ValidateOrderAgainstFactors(prime, order, divisorData, factorization))
		{
			reusableFactorization = factorization;
			return false;
		}

		factorization.Dispose();
		return true;
	}

	private static bool ValidateOrderAgainstFactors(
		ulong prime,
		ulong order,
		in MontgomeryDivisorData divisorData,
		PartialFactorResult factorization)
	{
		ReadOnlySpan<ulong> factorsSpan = factorization.Factors!;
		ReadOnlySpan<int> exponentsSpan = factorization.Exponents!;

		int length = factorization.Count;

		const int StackExponentCapacity = 32;
		const int ExponentHardLimit = 256;

		Span<ulong> stackBuffer = stackalloc ulong[StackExponentCapacity];

		ulong[]? heapCandidateArray = null;
		ExponentRemainderStepperCpu stepper = ThreadStaticPools.RentExponentStepperCpu(divisorData);
		bool violates = false;

		for (int i = 0; i < length; i++)
		{
			ulong primeFactor = factorsSpan[i];
			int exponent = exponentsSpan[i];

			// Factor exponents emitted by TryConfirmCandidateCpu stay positive for production workloads.
			// if (exponent <= 0)
			// {
			// 	continue;
			// }

			if (exponent > ExponentHardLimit)
			{
				throw new InvalidOperationException($"Factor exponent {exponent} exceeds the supported limit of {ExponentHardLimit}.");
			}

			if (exponent <= StackExponentCapacity)
			{
				if (ValidateOrderForFactor(stackBuffer[..exponent], primeFactor, exponent, order, ref stepper))
				{
					violates = true;
					break;
				}

				continue;
			}

			heapCandidateArray ??= ThreadStaticPools.UlongPool.Rent(ExponentHardLimit);
			if (ValidateOrderForFactor(heapCandidateArray.AsSpan(0, exponent), primeFactor, exponent, order, ref stepper))
			{
				violates = true;
				break;
			}
		}

		ThreadStaticPools.ReturnExponentStepperCpu(stepper);

		if (heapCandidateArray is not null)
		{
			ThreadStaticPools.UlongPool.Return(heapCandidateArray);
		}

		return !violates;
	}

	private static bool ValidateOrderForFactor(Span<ulong> buffer, ulong primeFactor, int exponent, ulong order, ref ExponentRemainderStepperCpu stepper)
	{
		ulong working = order;
		int actual = 0;
		while (actual < exponent)
		{
			if (working < primeFactor)
			{
				break;
			}

			ulong reduced = working / primeFactor;
			if (reduced == 0UL)
			{
				break;
			}

			buffer[actual] = reduced;
			working = reduced;
			actual++;
		}

		if (actual == 0)
		{
			return false;
		}

		Span<ulong> candidates = buffer[..actual];

		stepper.Reset();

		int last = actual - 1;
		if (stepper.InitializeCpuIsUnity(candidates[last]))
		{
			return true;
		}

		for (int j = last - 1; j >= 0; j--)
		{
			if (stepper.ComputeNextIsUnity(candidates[j]))
			{
				return true;
			}
		}

		return false;
	}

	private static bool TryHeuristicFinishCpu(
		PrimeOrderCalculatorAccelerator gpu,
		ulong prime,
		ulong order,
		ulong? previousOrder,
		in MontgomeryDivisorData divisorData,
		in PrimeOrderSearchConfig config,
		PartialFactorResult phiFactors,
		PartialFactorResult? cachedOrderFactors,
		out ulong result)
	{
		result = 0UL;
		if (order <= 1UL)
		{
			cachedOrderFactors?.Dispose();
			return false;
		}

		// Reuse the partial factorization from TryConfirmOrderCpu when available.
		PartialFactorResult orderFactors = cachedOrderFactors ?? PartialFactor(gpu, order, config);
		try
		{
			if (orderFactors.Factors is null)
			{
				return false;
			}

			if (!orderFactors.FullyFactored)
			{
				if (orderFactors.Cofactor <= 1UL)
				{
					return false;
				}

				if (!orderFactors.CofactorIsPrime)
				{
					return false;
				}

				PartialFactorResult extended = orderFactors.WithAdditionalPrime(orderFactors.Cofactor);
				orderFactors.Dispose();
				orderFactors = extended;
			}

			int capacity = config.MaxPowChecks <= 0 ? 64 : config.MaxPowChecks << 2;
			List<ulong> candidates = ThreadStaticPools.RentUlongList(capacity);
			candidates.Clear();
			ulong[] factorArray = orderFactors.Factors!;
			int[] exponentsArray = orderFactors.Exponents!;
			// DebugLog("Building candidates list");
			BuildCandidates(order, factorArray, exponentsArray, orderFactors.Count, candidates, capacity);
			if (candidates.Count == 0)
			{
				ThreadStaticPools.ReturnUlongList(candidates);
				return false;
			}

			// DebugLog("Sorting candidates");
			SortCandidates(prime, previousOrder, candidates);

			int powBudget = config.MaxPowChecks <= 0 ? candidates.Count : config.MaxPowChecks;
			int powUsed = 0;
			int candidateCount = candidates.Count;
			bool allowGpuBatch = true;
			Span<ulong> candidateSpan = CollectionsMarshal.AsSpan(candidates);
			ExponentRemainderStepperCpu powStepper = ThreadStaticPools.RentExponentStepperCpu(divisorData);
			bool powStepperInitialized = false;

			// DebugLog(() => $"Checking candidates ({candidateCount} candidates, {powBudget} pow budget)");
			int index = 0;
			const int MaxGpuBatchSize = 256;
			const int StackGpuBatchSize = 64;
			Span<ulong> stackGpuRemainders = stackalloc ulong[StackGpuBatchSize];
			ArrayPool<ulong> pool = ThreadStaticPools.UlongPool;
			while (index < candidateCount && powUsed < powBudget)
			{
				int remaining = candidateCount - index;
				int budgetRemaining = powBudget - powUsed;
				int batchSize = Math.Min(remaining, Math.Min(budgetRemaining, MaxGpuBatchSize));
				if (batchSize <= 0)
				{
					break;
				}

				ReadOnlySpan<ulong> batch = candidateSpan.Slice(index, batchSize);
				ulong[]? gpuPool = null;
				Span<ulong> pooledGpuRemainders = default;
				bool gpuSuccess = false;
				bool gpuStackRemainders = false;
				GpuPow2ModStatus status = GpuPow2ModStatus.Unavailable;

				if (allowGpuBatch && IsGpuPow2Allowed)
				{
					if (batchSize <= StackGpuBatchSize)
					{
						Span<ulong> localRemainders = stackGpuRemainders[..batchSize];
						status = PrimeOrderGpuHeuristics.TryPow2ModBatch(batch, prime, localRemainders, divisorData);
						if (status == GpuPow2ModStatus.Success)
						{
							gpuSuccess = true;
							gpuStackRemainders = true;
						}
					}
					else
					{
						gpuPool = pool.Rent(batchSize);
						Span<ulong> pooledRemainders = gpuPool.AsSpan(0, batchSize);
						status = PrimeOrderGpuHeuristics.TryPow2ModBatch(batch, prime, pooledRemainders, divisorData);
						if (status == GpuPow2ModStatus.Success)
						{
							pooledGpuRemainders = pooledRemainders;
							gpuSuccess = true;
						}
						else
						{
							pool.Return(gpuPool, clearArray: false);
							gpuPool = null;
						}
					}

					if (!gpuSuccess && (status == GpuPow2ModStatus.Overflow || status == GpuPow2ModStatus.Unavailable))
					{
						allowGpuBatch = false;
					}
				}

				int startIndex = 0;

				if (!gpuSuccess && !powStepperInitialized)
				{
					ulong candidate = batch[0];
					powUsed++;

					bool equalsOne = powStepper.InitializeCpuIsUnity(candidate);
					powStepperInitialized = true;
					if (equalsOne && TryConfirmCandidateCpu(gpu, prime, candidate, divisorData, config, ref powUsed, powBudget))
					{
						if (gpuPool is not null)
						{
							pool.Return(gpuPool, clearArray: false);
						}

						candidates.Clear();
						ThreadStaticPools.ReturnUlongList(candidates);
						ThreadStaticPools.ReturnExponentStepperCpu(powStepper);
						result = candidate;
						return true;
					}

					startIndex = 1;
				}

				for (int i = startIndex; i < batchSize && powUsed < powBudget; i++)
				{
					ulong candidate = batch[i];
					powUsed++;

					bool equalsOne;
					if (gpuSuccess)
					{
						ulong remainderValue = gpuStackRemainders ? stackGpuRemainders[i] : pooledGpuRemainders[i];
						equalsOne = remainderValue == 1UL;
					}
					else
					{
						equalsOne = powStepper.ComputeNextIsUnity(candidate);
					}
					if (!equalsOne)
					{
						continue;
					}

					if (!TryConfirmCandidateCpu(gpu, prime, candidate, divisorData, config, ref powUsed, powBudget))
					{
						continue;
					}

					if (gpuPool is not null)
					{
						pool.Return(gpuPool, clearArray: false);
					}

					candidates.Clear();
					ThreadStaticPools.ReturnUlongList(candidates);
					ThreadStaticPools.ReturnExponentStepperCpu(powStepper);
					result = candidate;
					return true;
				}

				if (gpuPool is not null)
				{
					pool.Return(gpuPool, clearArray: false);
				}

				index += batchSize;
			}

			candidates.Clear();
			ThreadStaticPools.ReturnUlongList(candidates);
			ThreadStaticPools.ReturnExponentStepperCpu(powStepper);
			// DebugLog("No candidate confirmed");
			return false;
		}
		finally
		{
			orderFactors.Dispose();
		}
	}
	private static void SortCandidates(ulong prime, ulong? previousOrder, List<ulong> candidates)
	{
		ulong previous = previousOrder ?? 0UL;
		bool hasPrevious = previousOrder.HasValue;
		ulong threshold1 = prime >> 3;
		ulong threshold2 = prime >> 2;
		ulong threshold3 = (prime * 3UL) >> 3;
		int previousGroup = hasPrevious ? GetGroup(previous, threshold1, threshold2, threshold3) : 1;

		candidates.Sort((x, y) =>
		{
			CandidateKey keyX = BuildKey(x, previous, previousGroup, hasPrevious, threshold1, threshold2, threshold3);
			CandidateKey keyY = BuildKey(y, previous, previousGroup, hasPrevious, threshold1, threshold2, threshold3);
			int compare = keyX.Primary.CompareTo(keyY.Primary);
			if (compare != 0)
			{
				return compare;
			}

			compare = keyX.Secondary.CompareTo(keyY.Secondary);
			if (compare != 0)
			{
				return compare;
			}

			return keyX.Tertiary.CompareTo(keyY.Tertiary);
		});
	}

	private static CandidateKey BuildKey(ulong value, ulong previous, int previousGroup, bool hasPrevious, ulong threshold1, ulong threshold2, ulong threshold3)
	{
		int group = GetGroup(value, threshold1, threshold2, threshold3);
		if (group == 0)
		{
			return new CandidateKey(int.MaxValue, long.MaxValue, long.MaxValue);
		}

		bool isGe = !hasPrevious || value >= previous;
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
			ulong reference = hasPrevious ? previous : 0UL;
			ulong distance = hasPrevious ? (value > reference ? value - reference : reference - value) : value;
			secondary = (long)distance;
			tertiary = (long)value;
		}

		return new CandidateKey(primary, secondary, tertiary);
	}

	private static int ComputePrimary(int group, bool isGe, int previousGroup)
	{
		int groupOffset = group switch
		{
			1 => 0,
			2 => 2,
			3 => 4,
			_ => 6,
		};

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

	private static int GetGroup(ulong value, ulong threshold1, ulong threshold2, ulong threshold3)
	{
		if (value <= threshold1)
		{
			return 1;
		}

		if (value <= threshold2)
		{
			return 2;
		}

		if (value <= threshold3)
		{
			return 3;
		}

		return 0;
	}

	private static readonly Comparer<ulong> _comparer = Comparer<ulong>.Default;

	private static void BuildCandidates(ulong order, ulong[] factors, int[] exponents, int count, List<ulong> candidates, int limit)
	{
		if (count == 0)
		{
			return;
		}

		Array.Sort(factors, exponents, 0, count);
		BuildCandidatesRecursive(order, factors, exponents, 0, 1UL, candidates, limit);
	}

	// private static void BuildCandidatesRecursive(ulong order, in ReadOnlySpan<FactorEntry> factors, int index, ulong divisorProduct, List<ulong> candidates, int limit)
	private static void BuildCandidatesRecursive(ulong order, in ulong[] factors, in int[] exponents, int index, ulong divisorProduct, List<ulong> candidates, int limit)
	{
		if (candidates.Count >= limit)
		{
			return;
		}

		if (index >= factors.Length)
		{
			if (divisorProduct == 1UL || divisorProduct == order)
			{
				return;
			}

			ulong candidate = order / divisorProduct;
			if (candidate > 1UL && candidate < order)
			{
				candidates.Add(candidate);
			}

			return;
		}

		// FactorEntry factor = factors[index];
		// int factorExponent = factor.Exponent;
		// ulong primeFactor = factor.Value;

		ulong factor = factors[index];
		int factorExponent = exponents[index];

		ulong contribution = 1UL;
		ulong contributionLimit = factorExponent == 0 ? 0UL : order / factor;
		for (int exponent = 0; exponent <= factorExponent; exponent++)
		{
			ulong nextDivisor = divisorProduct * contribution;
			if (nextDivisor > order)
			{
				break;
			}

			BuildCandidatesRecursive(order, factors, exponents, index + 1, nextDivisor, candidates, limit);
			if (candidates.Count >= limit)
			{
				return;
			}

			if (exponent == factorExponent)
			{
				break;
			}

			if (contribution > contributionLimit)
			{
				break;
			}

			contribution *= factor;
		}
	}

	private static bool TryConfirmCandidateCpu(PrimeOrderCalculatorAccelerator gpu, ulong prime, ulong candidate, in MontgomeryDivisorData divisorData, in PrimeOrderSearchConfig config, ref int powUsed, int powBudget)
	{
		PartialFactorResult factorization = PartialFactor(gpu, candidate, config);
		try
		{
			if (factorization.Factors is null)
			{
				return false;
			}

			if (!factorization.FullyFactored)
			{
				if (factorization.Cofactor <= 1UL)
				{
					return false;
				}

				if (!factorization.CofactorIsPrime)
				{
					return false;
				}

				PartialFactorResult extended = factorization.WithAdditionalPrime(factorization.Cofactor);
				factorization.Dispose();
				factorization = extended;
			}

			// ReadOnlySpan<FactorEntry> span = factorization.Factors;
			ReadOnlySpan<ulong> factorsSpan = factorization.Factors;
			ReadOnlySpan<int> exponentsSpan = factorization.Exponents;
			int length = factorization.Count;

			const int StackExponentCapacity = 32;
			const int ExponentHardLimit = 256;

			Span<ulong> stackBuffer = stackalloc ulong[StackExponentCapacity];
			ulong[]? heapCandidateArray = null;
			ExponentRemainderStepperCpu stepper = ThreadStaticPools.RentExponentStepperCpu(divisorData);
			bool violates = false;

			for (int i = 0; i < length; i++)
			{
				// ulong primeFactor = factorsSpan[i].Value;
				// int exponent = factorsSpan[i].Exponent;
				ulong primeFactor = factorsSpan[i];
				int exponent = exponentsSpan[i];

				// Factorization entries for candidate confirmation are always positive here.
				// if (exponent <= 0)
				// {
				// 	continue;
				// }

				if (exponent > ExponentHardLimit)
				{
					throw new InvalidOperationException($"Candidate factor exponent {exponent} exceeds the supported limit of {ExponentHardLimit}.");
				}

				if (exponent <= StackExponentCapacity)
				{
					if (CheckCandidateViolation(stackBuffer[..exponent], primeFactor, exponent, candidate, prime, ref powUsed, powBudget, ref stepper))
					{
						violates = true;
						break;
					}

					continue;
				}

				heapCandidateArray ??= ThreadStaticPools.UlongPool.Rent(ExponentHardLimit);
				if (CheckCandidateViolation(heapCandidateArray.AsSpan(0, exponent), primeFactor, exponent, candidate, prime, ref powUsed, powBudget, ref stepper))
				{
					violates = true;
					break;
				}
			}

			ThreadStaticPools.ReturnExponentStepperCpu(stepper);

			if (heapCandidateArray is not null)
			{
				ThreadStaticPools.UlongPool.Return(heapCandidateArray);
			}

			return !violates;
		}
		finally
		{
			factorization.Dispose();
		}
	}

	private static bool CheckCandidateViolation(Span<ulong> buffer, ulong primeFactor, int exponent, ulong candidate, ulong prime, ref int powUsed, int powBudget, ref ExponentRemainderStepperCpu stepper)
	{
		ulong working = candidate;
		int actual = 0;
		while (actual < exponent)
		{
			if (working < primeFactor)
			{
				break;
			}

			ulong reduced = working / primeFactor;
			if (reduced == 0UL)
			{
				break;
			}

			buffer[actual] = reduced;
			working = reduced;
			actual++;
		}

		if (actual == 0)
		{
			return false;
		}

		Span<ulong> candidates = buffer[..actual];

		stepper.Reset();
		int last = actual - 1;

		bool enforceBudget = powBudget > 0;
		if (!enforceBudget)
		{
			powUsed++;
			if (stepper.InitializeCpuIsUnity(candidates[last]))
			{
				return true;
			}

			for (int j = last - 1; j >= 0; j--)
			{
				powUsed++;
				if (stepper.ComputeNextIsUnity(candidates[j]))
				{
					return true;
				}
			}

			return false;
		}

		if (powUsed >= powBudget)
		{
			return true;
		}

		powUsed++;
		if (stepper.InitializeCpuIsUnity(candidates[last]))
		{
			return true;
		}

		for (int j = last - 1; j >= 0; j--)
		{
			if (powUsed >= powBudget)
			{
				return true;
			}

			powUsed++;
			if (stepper.ComputeNextIsUnity(candidates[j]))
			{
				return true;
			}
		}

		return false;
	}

	private static ulong _pow2GpuHits;
	private static ulong _pow2CpuHits;
	private static bool Pow2EqualsOneCpu(PrimeOrderCalculatorAccelerator gpu, ulong exponent, ulong prime, in MontgomeryDivisorData divisorData)
	{
		if (exponent <= (PerfectNumberConstants.MaxQForDivisorCycles) || prime <= (PerfectNumberConstants.MaxQForDivisorCycles))
		{
			// Atomic.Add(ref _pow2GpuHits, 1);
			// Console.WriteLine($"pow2 GPU Hits {_pow2GpuHits}");
			ulong remainder = exponent.Pow2MontgomeryModWindowedConvertGpu(gpu, divisorData);
			// GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2Mod(exponent, prime, out ulong remainder, divisorData);
			// if (status == GpuPow2ModStatus.Success)
			// {
			return remainder == 1UL;
			// }
		}

		// Atomic.Add(ref _pow2CpuHits, 1);
		// Console.WriteLine($"pow2 CPU Hits {_pow2CpuHits}");
		return exponent.Pow2MontgomeryModWindowedCpu(divisorData, keepMontgomery: false) == 1UL;
	}

	// private static ulong _partialFactorHits;
	// private static ulong _partialFactorPendingHits;
	// private static ulong _partialFactorCofactorHits;

	private static PartialFactorResult PartialFactor(PrimeOrderCalculatorAccelerator gpu, ulong value, in PrimeOrderSearchConfig config)
	{
		if (value <= 1UL)
		{
			return PartialFactorResult.Empty;
		}

		const int FactorSlotCount = GpuSmallPrimeFactorSlots;

		// stackalloc is faster than pooling
		// ulong[] primeSlotsArray = ThreadStaticPools.UlongPool.Rent(FactorSlotCount);
		// int[] exponentSlotsArray = ThreadStaticPools.IntPool.Rent(FactorSlotCount);
		// Span<ulong> primeSlots = primeSlotsArray.AsSpan(0, FactorSlotCount);
		// Span<int> exponentSlots = exponentSlotsArray.AsSpan(0, FactorSlotCount);
		Span<ulong> primeSlots = stackalloc ulong[FactorSlotCount];
		Span<int> exponentSlots = stackalloc int[FactorSlotCount];
		ulong[]? primeSlotArray = null;
		int[]? exponentSlotArray = null;
		// We don't need to worry about leftovers, because we always use indexes within the calculated counts
		// primeSlots.Clear();
		// exponentSlots.Clear();

		int factorCount = 0;

		uint limit = config.SmallFactorLimit == 0 ? uint.MaxValue : config.SmallFactorLimit;
		ulong remaining = value;

		bool gpuFactored = false;
		if (IsGpuHeuristicDevice)
		{
			gpuFactored = PrimeOrderGpuHeuristics.TryPartialFactor(gpu, value, limit, primeSlots, exponentSlots, out factorCount, out remaining);
		}

		List<PendingEntry> pending = ThreadStaticPools.RentPrimeOrderPendingEntryList(2);
		Stack<ulong>? compositeStack = null;
		PartialFactorResult result;
		PrimeTester primeTester = PrimeTester.Exclusive;
		CancellationToken ct = CancellationToken.None;
		// HeuristicPrimeTester? tester = _tester ??= new();

		if (!gpuFactored)
		{
			PopulateSmallPrimeFactorsCpu(
				value,
				limit,
				ref primeSlots,
				ref exponentSlots,
				ref primeSlotArray,
				ref exponentSlotArray,
				out factorCount,
				out remaining);
		}


		if (remaining > 1UL)
		{
			pending.Add(new PendingEntry(remaining, knownComposite: false));
		}

		if (config.PollardRhoMilliseconds > 0 && pending.Count > 0)
		{
			long deadlineTimestamp = CreateDeadlineTimestamp(config.PollardRhoMilliseconds);
			compositeStack = ThreadStaticPools.RentUlongStack(Math.Max(pending.Count << 1, 4));
			compositeStack.Push(remaining);
			pending.Clear();

			bool pollardRhoDeadlineReached = Stopwatch.GetTimestamp() > deadlineTimestamp;
			if (!pollardRhoDeadlineReached)
			{
				// GpuPrimeWorkLimiter.Acquire();

				while (compositeStack.Count > 0)
				{
					ulong composite = compositeStack.Pop();
					// composite will never be smaller than 1 on the execution path
					// if (composite <= 1UL)
					// {
					// 	continue;
					// }

					// bool isPrime = primeTester.HeuristicIsPrimeGpu(composite);
					// bool isPrime = PrimeTester.IsPrimeInternal(composite, CancellationToken.None);

					// Atomic.Add(ref _partialFactorHits, 1UL);
					// Console.WriteLine($"Partial factor hits {Volatile.Read(ref _partialFactorHits)}");

					// bool isPrime = PrimeTester.IsPrimeCpu(composite, CancellationToken.None);
					bool isPrime = PrimeTester.IsPrimeGpu(gpu, composite);
					// bool isPrime = Open.Numeric.Primes.Prime.Numbers.IsPrime(composite);

					if (isPrime)
					{
						AddFactorToCollector(ref primeSlots, ref exponentSlots, ref primeSlotArray, ref exponentSlotArray, ref factorCount, composite, 1);
						continue;
					}

					long currentTimestamp = Stopwatch.GetTimestamp();
					if (currentTimestamp > deadlineTimestamp)
					{
						pollardRhoDeadlineReached = true;
						pending.Add(new PendingEntry(composite, knownComposite: true));
						continue;
					}

					if (!TryPollardRhoGpu(gpu, composite, out ulong factor))
					// if (!TryPollardRhoCpu(composite, deadlineTimestamp, out ulong factor))
					{
						pending.Add(new PendingEntry(composite, knownComposite: true));
						continue;
					}

					ulong quotient = composite / factor;
					compositeStack.Push(factor);
					compositeStack.Push(quotient);
				}

				if (!pollardRhoDeadlineReached)
				{
					pollardRhoDeadlineReached = Stopwatch.GetTimestamp() > deadlineTimestamp;
				}

				// GpuPrimeWorkLimiter.Release();
			}

			if (pollardRhoDeadlineReached)
			{
				while (compositeStack.Count > 0)
				{
					pending.Add(new PendingEntry(compositeStack.Pop(), knownComposite: false));
				}
			}
		}

		ulong cofactor = 1UL;
		bool cofactorContainsComposite = false;
		int pendingCount = pending.Count;
		int index = 0;
		// GpuPrimeWorkLimiter.Acquire();
		for (; index < pendingCount; index++)
		{
			PendingEntry entry = pending[index];
			ulong composite = entry.Value;

			if (entry.KnownComposite)
			{
				cofactor = checked(cofactor * composite);
				cofactorContainsComposite = true;
				continue;
			}

			if (!entry.HasKnownPrimality)
			{
				// bool isPrime = primeTester.HeuristicIsPrimeGpu(composite);
				// bool isPrime = PrimeTester.IsPrimeInternal(composite, CancellationToken.None);

				// Atomic.Add(ref _partialFactorPendingHits, 1UL);
				// Console.WriteLine($"Partial factor pending hits {Volatile.Read(ref _partialFactorPendingHits)}");

				// bool isPrime = PrimeTester.IsPrimeCpu(composite, CancellationToken.None);
				bool isPrime = PrimeTester.IsPrimeGpu(gpu, composite);
				// bool isPrime = Open.Numeric.Primes.Prime.Numbers.IsPrime(composite);

				entry = entry.WithPrimality(isPrime);
				pending[index] = entry;
			}

			if (entry.IsPrime)
			{
				AddFactorToCollector(ref primeSlots, ref exponentSlots, ref primeSlotArray, ref exponentSlotArray, ref factorCount, composite, 1);
			}
			// composite is never smaller on the execution path
			// else if (composite > 1UL)
			else
			{
				cofactor = checked(cofactor * composite);
				cofactorContainsComposite = true;
			}
		}
		
		bool cofactorIsPrime;
		if (cofactor <= 1UL)
		{
			cofactorIsPrime = false;
		}
		else if (cofactorContainsComposite)
		{
			cofactorIsPrime = false;
		}
		else
		{
			// cofactorIsPrime = primeTester.HeuristicIsPrimeGpu(cofactor);
			// cofactorIsPrime = PrimeTester.IsPrimeInternal(cofactor, CancellationToken.None);

			// Atomic.Add(ref _partialFactorCofactorHits, 1UL);
			// Console.WriteLine($"Partial factor cofactor hits {Volatile.Read(ref _partialFactorCofactorHits)}");

			cofactorIsPrime = PrimeTester.IsPrimeGpu(gpu, cofactor);
			// cofactorIsPrime = PrimeTester.IsPrimeGpu(cofactor);
			// cofactorIsPrime = Open.Numeric.Primes.Prime.Numbers.IsPrime(cofactor);
		}

		// GpuPrimeWorkLimiter.Release();

		// ArrayPool<FactorEntry> pool = ThreadStaticPools.FactorEntryPool;
		bool temp;
		if (factorCount == 0)
		{
			if (cofactor == value)
			{
				result = PartialFactorResult.Rent(cofactor, false, 0, cofactorIsPrime);
				goto ReturnResult;
			}

			result = PartialFactorResult.Rent(cofactor, cofactor == 1UL, 0, cofactorIsPrime);
			goto ReturnResult;
		}

		// This will never happen in production code. We'll always get at least 1 factor
		// if (factorCount == 0)
		// {
		// 	if (cofactor == value)
		// 	{
		// 		result = PartialFactorResult.Rent(null, value, false, 0);
		// 		goto ReturnResult;
		// 	}

		// 	result = PartialFactorResult.Rent(null, cofactor, cofactor == 1UL, 0);
		// 	goto ReturnResult;
		// }

		ArrayPool<ulong> ulongPool = ThreadStaticPools.UlongPool;
		ArrayPool<int> intPool = ThreadStaticPools.IntPool;
		// ArrayPool<bool> boolPool = ThreadStaticPools.BoolPool;

		// FactorEntry[] array = pool.Rent(factorCount);
		ulong[] factorsArray = ulongPool.Rent(factorCount);
		int[] exponentsArray = intPool.Rent(factorCount);
		// bool[] cofactorIsPrimesArray = boolPool.Rent(factorCount);

		int arrayIndex = 0;
		for (int i = 0; i < factorCount; i++)
		{
			ulong primeValue = primeSlots[i];
			int exponentValue = exponentSlots[i];

			// This will never happen on the execution path from production code
			// if (primeValue == 0UL || exponentValue == 0)
			// {
			// 	throw new Exception("Prime value or exponent equals zero");
			// 	continue;
			// }

			factorsArray[arrayIndex] = primeValue;
			exponentsArray[arrayIndex] = exponentValue;
			arrayIndex++;
		}

		Array.Sort(factorsArray, exponentsArray, 0, arrayIndex);
		temp = cofactor == 1UL;
		
		result = PartialFactorResult.Rent(factorsArray, exponentsArray, cofactor, temp, arrayIndex, cofactorIsPrime);

	ReturnResult:
		if (primeSlotArray is not null)
		{
			ThreadStaticPools.UlongPool.Return(primeSlotArray, clearArray: false);
			ThreadStaticPools.IntPool.Return(exponentSlotArray!, clearArray: false);
		}

		if (compositeStack is not null)
		{
			compositeStack.Clear();
			ThreadStaticPools.ReturnUlongStack(compositeStack);
		}

		pending.Clear();
		ThreadStaticPools.ReturnPrimeOrderPendingEntryList(pending);
		return result;
	}

	private static void PopulateSmallPrimeFactorsCpu(
				ulong value,
				uint limit,
				ref Span<ulong> primeTargets,
				ref Span<int> exponentTargets,
				ref ulong[]? primeArray,
				ref int[]? exponentArray,
				out int factorCount,
				out ulong remaining)
	{
		factorCount = 0;
		ulong remainingLocal = value;

		if (value <= 1UL)
		{
			remaining = remainingLocal;
			return;
		}

		uint[] primes = PrimesGenerator.SmallPrimes;
		ulong[] squares = PrimesGenerator.SmallPrimesPow2;
		int primeCount = primes.Length;
		uint effectiveLimit = limit == 0 ? uint.MaxValue : limit;

		for (int i = 0; i < primeCount && remainingLocal > 1UL; i++)
		{
			uint primeCandidate = primes[i];
			if (primeCandidate > effectiveLimit)
			{
				break;
			}

			ulong primeSquare = squares[i];
			// primeSquare will never == 0 in production code
			// if (primeSquare != 0 && primeSquare > remainingLocal)
			if (primeSquare > remainingLocal)
			{
				break;
			}

			ulong primeValue = primeCandidate;
			// primeCandidate will never equal 0 in production code
			// if (primeCandidate == 0UL)
			// {
			// 	continue;
			// }

			if ((remainingLocal % primeValue) != 0UL)
			{
				continue;
			}

			int exponent = ExtractSmallPrimeExponent(ref remainingLocal, primeValue);
			if (exponent == 0)
			{
				continue;
			}

			EnsureCollectorCapacity(
				ref primeTargets,
				ref exponentTargets,
				ref primeArray,
				ref exponentArray,
				factorCount,
				factorCount + 1);

			primeTargets[factorCount] = primeValue;
			exponentTargets[factorCount] = exponent;
			factorCount++;
		}

		remaining = remainingLocal;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	// Unroll the initial divisions because small exponents dominate in practice.
	private static int ExtractSmallPrimeExponent(ref ulong value, ulong primeValue)
	{
		ulong dividend = value;
		ulong quotient = dividend / primeValue;
		ulong remainder = dividend - (quotient * primeValue);
		if (remainder != 0UL)
		{
			return 0;
		}

		dividend = quotient;
		int exponent = 1;

		quotient = dividend / primeValue;
		remainder = dividend - (quotient * primeValue);
		if (remainder != 0UL)
		{
			value = dividend;
			return exponent;
		}

		dividend = quotient;
		exponent++;

		quotient = dividend / primeValue;
		remainder = dividend - (quotient * primeValue);
		if (remainder != 0UL)
		{
			value = dividend;
			return exponent;
		}

		dividend = quotient;
		exponent++;

		quotient = dividend / primeValue;
		remainder = dividend - (quotient * primeValue);
		if (remainder != 0UL)
		{
			value = dividend;
			return exponent;
		}

		dividend = quotient;
		exponent++;

		while (true)
		{
			quotient = dividend / primeValue;
			remainder = dividend - (quotient * primeValue);
			if (remainder != 0UL)
			{
				break;
			}

			dividend = quotient;
			exponent++;
		}

		value = dividend;
		return exponent;
	}

	private static long CreateDeadlineTimestamp(int milliseconds)
	{
		if (milliseconds <= 0)
		{
			return long.MaxValue;
		}

		long stopwatchTicks = ConvertMillisecondsToStopwatchTicks(milliseconds);
		long startTimestamp = Stopwatch.GetTimestamp();
		long deadline = startTimestamp + stopwatchTicks;
		if (deadline < startTimestamp)
		{
			return long.MaxValue;
		}

		return deadline;
	}

	private static long ConvertMillisecondsToStopwatchTicks(int milliseconds)
	{
		long frequency = Stopwatch.Frequency;
		long baseTicks = (frequency / 1000L) * milliseconds;
		long remainder = (frequency % 1000L) * milliseconds;
		return baseTicks + (remainder / 1000L);
	}

	private static bool TryPollardRhoCpu(ulong n, long deadlineTimestamp, out ulong factor)
	{
		factor = 0UL;
		if ((n & 1UL) == 0UL)
		{
			factor = 2UL;
			return true;
		}

		long timestamp = 0L; // reused for deadline checks.
		while (true)
		{
			timestamp = Stopwatch.GetTimestamp();
			if (timestamp > deadlineTimestamp)
			{
				return false;
			}

			ulong c = (DeterministicRandomCpu.NextUInt64() % (n - 1UL)) + 1UL;
			ulong x = (DeterministicRandomCpu.NextUInt64() % (n - 2UL)) + 2UL;
			ulong y = x;
			ulong d = 1UL;

			while (d == 1UL)
			{
				timestamp = Stopwatch.GetTimestamp();
				if (timestamp > deadlineTimestamp)
				{
					return false;
				}

				x = AdvancePolynomialCpu(x, c, n);
				y = AdvancePolynomialCpu(y, c, n);
				y = AdvancePolynomialCpu(y, c, n);
				ulong diff = x > y ? x - y : y - x;
				d = BinaryGcd(diff, n);
			}

			if (d != n)
			{
				factor = d;
				return true;
			}
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ulong AdvancePolynomialCpu(ulong x, ulong c, ulong modulus)
	{
		UInt128 value = (UInt128)x * x + c;
		return (ulong)(value % modulus);
	}

	private static ulong BinaryGcd(ulong a, ulong b)
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
		a >>= BitOperations.TrailingZeroCount(a);

		while (true)
		{
			b >>= BitOperations.TrailingZeroCount(b);
			if (a > b)
			{
				(a, b) = (b, a);
			}

			b -= a;
			if (b == 0UL)
			{
				return a << shift;
			}
		}
	}

	private static void AddFactor(Dictionary<ulong, int> counts, ulong prime, int exponent)
	{
		if (counts.TryGetValue(prime, out int existing))
		{
			counts[prime] = existing + exponent;
		}
		else
		{
			counts[prime] = exponent;
		}
	}

	private static void AddFactorToCollector(
				ref Span<ulong> primes,
				ref Span<int> exponents,
				ref ulong[]? primeArray,
				ref int[]? exponentArray,
				ref int count,
				ulong prime,
				int exponent)
	{
		if (exponent <= 0)
		{
			return;
		}

		for (int i = 0; i < count; i++)
		{
			if (primes[i] == prime)
			{
				exponents[i] += exponent;
				return;
			}
		}

		EnsureCollectorCapacity(
			ref primes,
			ref exponents,
			ref primeArray,
			ref exponentArray,
			count,
			count + 1);

		primes[count] = prime;
		exponents[count] = exponent;
		count++;
	}

	private static void EnsureCollectorCapacity(
				ref Span<ulong> primes,
				ref Span<int> exponents,
				ref ulong[]? primeArray,
				ref int[]? exponentArray,
				int count,
				int requiredCapacity)
	{
		if (requiredCapacity <= primes.Length)
		{
			return;
		}

		int newCapacity = primes.Length << 1;
		if (newCapacity < requiredCapacity)
		{
			newCapacity = requiredCapacity;
		}

		ulong[] newPrimeArray = ThreadStaticPools.UlongPool.Rent(newCapacity);
		int[] newExponentArray = ThreadStaticPools.IntPool.Rent(newCapacity);
		Span<ulong> newPrimeSpan = newPrimeArray.AsSpan();
		Span<int> newExponentSpan = newExponentArray.AsSpan();

		primes.Slice(0, count).CopyTo(newPrimeSpan);
		exponents.Slice(0, count).CopyTo(newExponentSpan);

		if (primeArray is not null)
		{
			ThreadStaticPools.UlongPool.Return(primeArray, clearArray: false);
			ThreadStaticPools.IntPool.Return(exponentArray!, clearArray: false);
		}

		primeArray = newPrimeArray;
		exponentArray = newExponentArray;
		primes = newPrimeSpan;
		exponents = newExponentSpan;
	}

	private static ulong CalculateByFactorizationCpu(PrimeOrderCalculatorAccelerator gpu, ulong prime, in MontgomeryDivisorData divisorData)
	{
		ulong phi = prime - 1UL;
		Dictionary<ulong, int> counts = new(capacity: 8);
		FactorCompletelyCpu(phi, counts);
		if (counts.Count == 0)
		{
			return phi;
		}

		KeyValuePair<ulong, int>[] entries = [.. counts];
		Array.Sort(entries, static (a, b) => a.Key.CompareTo(b.Key));

		ulong order = phi;
		int entryCount = entries.Length;

		// GpuPrimeWorkLimiter.Acquire();
		for (int i = 0; i < entryCount; i++)
		{
			ulong primeFactor = entries[i].Key;
			int exponent = entries[i].Value;
			for (int iteration = 0; iteration < exponent; iteration++)
			{
				ulong remainder = order % primeFactor;
				if (remainder != 0UL)
				{
					break;
				}

				ulong candidate = order - primeFactor * remainder;
				// ulong candidate = order / primeFactor;

				// if (candidate.Pow2MontgomeryModWindowedCpu(divisorData, keepMontgomery: false) == 1UL)
				if (candidate.Pow2MontgomeryModWindowedConvertGpu(gpu, divisorData) == 1UL)
				// if (Pow2EqualsOneCpu(candidate, prime, divisorData))
				{
					order = candidate;
					continue;
				}

				break;
			}
		}

		// GpuPrimeWorkLimiter.Release();
		return order;
	}

	private static void FactorCompletelyCpu(ulong value, Dictionary<ulong, int> counts)
	{
		FactorCompletely(value, counts, knownComposite: false);
	}

	// private static ulong _factorCompletelyHits;
	// private static ulong _factorCompletelyPollardRhoHits;
	// [ThreadStatic]
	// private static HeuristicPrimeTester? _tester;

	private static void FactorCompletely(ulong value, Dictionary<ulong, int> counts, bool knownComposite)
	{
		if (value <= 1UL)
		{
			return;
		}

		// Atomic.Add(ref _factorCompletelyHits, 1UL);
		// Console.WriteLine($"Factor completely hits {Volatile.Read(ref _factorCompletelyHits)}");
		// HeuristicPrimeTester tester = _tester ??= new();

		// bool isPrime = primeTester.HeuristicIsPrimeGpu(value);
		bool isPrime = PrimeTester.IsPrimeGpu(value);
		// bool isPrime = PrimeTester.IsPrimeGpu(value);
		// bool isPrime = Open.Numeric.Primes.Prime.Numbers.IsPrime(value);
		if (!knownComposite && isPrime)
		{
			AddFactor(counts, value, 1);
			return;
		}

		ulong factor = PollardRhoStrict(value);
		ulong quotient = value / factor;

		// Atomic.Add(ref _factorCompletelyPollardRhoHits, 1UL);
		// Console.WriteLine($"Factor completely after PollardRho hits {Volatile.Read(ref _factorCompletelyPollardRhoHits)}");

		// isPrime = primeTester.HeuristicIsPrimeGpu(factor);
		isPrime = PrimeTester.IsPrimeGpu(factor);
		// isPrime = PrimeTester.IsPrimeGpu(factor);
		// isPrime = Open.Numeric.Primes.Prime.Numbers.IsPrime(factor);
		if (isPrime)
		{
			int exponent = 1;
			ulong remaining = quotient;
			while ((remaining % factor) == 0UL)
			{
				remaining /= factor;
				exponent++;
			}

			AddFactor(counts, factor, exponent);

			if (remaining > 1UL)
			{
				FactorCompletely(remaining, counts, knownComposite: false);
			}

			return;
		}

		FactorCompletely(factor, counts, knownComposite: true);

		if (quotient == factor)
		{
			isPrime = false;
		}
		else
		{
			// isPrime = primeTester.HeuristicIsPrimeGpu(quotient);
			// isPrime = PrimeTester.IsPrimeInternal(quotient, CancellationToken.None);

			// Atomic.Add(ref _partialFactorCofactorHits, 1UL);
			// Console.WriteLine($"Partial factor cofactor hits {Volatile.Read(ref _partialFactorCofactorHits)}");

			isPrime = PrimeTester.IsPrimeGpu(quotient);
			// isPrime = PrimeTester.IsPrimeGpu(quotient);
			// isPrime = Open.Numeric.Primes.Prime.Numbers.IsPrime(quotient);
		}

		if (isPrime)
		{
			AddFactor(counts, quotient, 1);
		}
		else
		{
			FactorCompletely(quotient, counts, knownComposite: true);
		}
	}

	private static ulong PollardRhoStrict(ulong n)
	{
		if ((n & 1UL) == 0UL)
		{
			return 2UL;
		}

		while (true)
		{
			ulong c = (DeterministicRandomCpu.NextUInt64() % (n - 1UL)) + 1UL;
			ulong x = (DeterministicRandomCpu.NextUInt64() % (n - 2UL)) + 2UL;
			ulong y = x;
			ulong d = 1UL;

			while (d == 1UL)
			{
				x = AdvancePolynomialCpu(x, c, n);
				y = AdvancePolynomialCpu(y, c, n);
				y = AdvancePolynomialCpu(y, c, n);
				ulong diff = x > y ? x - y : y - x;
				d = BinaryGcd(diff, n);
			}

			if (d != n)
			{
				return d;
			}
		}
	}

	private readonly struct CandidateKey(int primary, long secondary, long tertiary)
	{
		public readonly int Primary = primary;
		public readonly long Secondary = secondary;
		public readonly long Tertiary = tertiary;
	}
}

