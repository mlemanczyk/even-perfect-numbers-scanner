using System;
using System.Buffers;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ILGPU;
using PerfectNumbers.Core.Gpu;

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

		if (IsGpuHeuristicDevice && PrimeOrderGpuHeuristics.TryCalculateOrder(prime, previousOrder, config, divisorData, out ulong gpuOrder))
		{
			return gpuOrder;
		}

		PartialFactorResult phiFactors = PartialFactor(phi, config);

		ulong result;
		if (phiFactors.Factors is null)
		{
			result = CalculateByFactorizationCpu(prime, divisorData);
			phiFactors.Dispose();
			return result;
		}

		result = RunHeuristicPipelineCpu(prime, previousOrder, config, divisorData, phi, phiFactors);
		phiFactors.Dispose();
		return result;
	}

	private static ulong RunHeuristicPipelineCpu(
		ulong prime,
		ulong? previousOrder,
		in PrimeOrderSearchConfig config,
		in MontgomeryDivisorData divisorData,
		ulong phi,
		PartialFactorResult phiFactors)
	{
		if (phiFactors.FullyFactored && TrySpecialMaxCpu(phi, prime, phiFactors, divisorData))
		{
			return phi;
		}

		ulong candidateOrder = InitializeStartingOrderCpu(prime, phi, divisorData);
		candidateOrder = ExponentLoweringCpu(candidateOrder, prime, phiFactors, divisorData);

		if (TryConfirmOrderCpu(prime, candidateOrder, divisorData, config, out PartialFactorResult? orderFactors))
		{
			return candidateOrder;
		}

		if (config.Mode == PrimeOrderMode.Strict)
		{
			orderFactors?.Dispose();
			return CalculateByFactorizationCpu(prime, divisorData);
		}

		if (TryHeuristicFinishCpu(prime, candidateOrder, previousOrder, divisorData, config, phiFactors, orderFactors, out ulong order))
		{
			return order;
		}

		return candidateOrder;
	}


	private static bool TrySpecialMaxCpu(ulong phi, ulong prime, PartialFactorResult factors, in MontgomeryDivisorData divisorData)
	{
		int length = factors.Count;
		// The phi factorization on the scanner path always yields at least one entry for phi >= 2,
		// preventing the zero-length case from occurring outside synthetic tests.
		// if (length == 0)
		// {
		//     return true;
		// }

		ReadOnlySpan<FactorEntry> factorSpan = new(factors.Factors, 0, length);

		if (length <= 32)
		{
			Span<ulong> stackBuffer = stackalloc ulong[length];
			return EvaluateSpecialMaxCandidates(stackBuffer, factorSpan, phi, prime, divisorData);
		}

		ulong[] rented = ThreadStaticPools.UlongPool.Rent(length);
		bool result = EvaluateSpecialMaxCandidates(rented.AsSpan(0, length), factorSpan, phi, prime, divisorData);
		ThreadStaticPools.UlongPool.Return(rented, clearArray: false);
		return result;
	}

	private static bool EvaluateSpecialMaxCandidates(Span<ulong> buffer, ReadOnlySpan<FactorEntry> factors, ulong phi, ulong prime, in MontgomeryDivisorData divisorData)
	{
		int actual = 0;
		int factorCount = factors.Length;
		for (int i = 0; i < factorCount; i++)
		{
			ulong factor = factors[i].Value;
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

	private static ulong InitializeStartingOrderCpu(ulong prime, ulong phi, in MontgomeryDivisorData divisorData)
	{
		ulong order = phi;
		if ((prime & 7UL) == 1UL || (prime & 7UL) == 7UL)
		{
			ulong half = phi >> 1;
			if (Pow2EqualsOneCpu(half, prime, divisorData))
			{
				order = half;
			}
		}

		return order;
	}

	private static ulong ExponentLoweringCpu(ulong order, ulong prime, PartialFactorResult factors, in MontgomeryDivisorData divisorData)
	{
		ArrayPool<FactorEntry> pool = ThreadStaticPools.FactorEntryPool;

		ReadOnlySpan<FactorEntry> factorSpan = factors.Factors;
		int length = factors.Count;
		FactorEntry[] tempArray = pool.Rent(length + 1);

		Span<FactorEntry> buffer = tempArray;
		factorSpan.CopyTo(buffer);

		if (!factors.FullyFactored && factors.Cofactor > 1UL && factors.CofactorIsPrime)
		{
			buffer[length] = new FactorEntry(factors.Cofactor, 1, true);
			length++;
		}

		buffer[..length].Sort(static (a, b) => a.Value.CompareTo(b.Value));

		ExponentRemainderStepperCpu stepper = ThreadStaticPools.RentExponentStepperCpu(divisorData);

		const int StackExponentCapacity = 16;
		const int ExponentHardLimit = 256;

		Span<ulong> stackCandidates = stackalloc ulong[StackExponentCapacity];
		Span<bool> stackEvaluations = stackalloc bool[StackExponentCapacity];

		ulong[]? heapCandidateArray = null;
		bool[]? heapEvaluationArray = null;

		for (int i = 0; i < length; i++)
		{
			ulong primeFactor = buffer[i].Value;
			int exponent = buffer[i].Exponent;
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

		pool.Return(tempArray, clearArray: false);
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

		if (!Pow2EqualsOneCpu(order, prime, divisorData))
		{
			return false;
		}

		PartialFactorResult factorization = PartialFactor(order, config);
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
		ReadOnlySpan<FactorEntry> span = factorization.Factors!;
		int length = factorization.Count;

		const int StackExponentCapacity = 32;
		const int ExponentHardLimit = 256;

		Span<ulong> stackBuffer = stackalloc ulong[StackExponentCapacity];

		ulong[]? heapCandidateArray = null;
		ExponentRemainderStepperCpu stepper = ThreadStaticPools.RentExponentStepperCpu(divisorData);
		bool violates = false;

		for (int i = 0; i < length; i++)
		{
			ulong primeFactor = span[i].Value;
			int exponent = span[i].Exponent;
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
		PartialFactorResult orderFactors = cachedOrderFactors ?? PartialFactor(order, config);
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

				// if (!PrimeTester.IsPrimeCpu(orderFactors.Cofactor, CancellationToken.None))
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
			FactorEntry[] factorArray = orderFactors.Factors!;
			// DebugLog("Building candidates list");
			BuildCandidates(order, factorArray, orderFactors.Count, candidates, capacity);
			if (candidates.Count == 0)
			{
				candidates.Clear();
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
					if (equalsOne && TryConfirmCandidateCpu(prime, candidate, divisorData, config, ref powUsed, powBudget))
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

					if (!TryConfirmCandidateCpu(prime, candidate, divisorData, config, ref powUsed, powBudget))
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

	private static void BuildCandidates(ulong order, FactorEntry[] factors, int count, List<ulong> candidates, int limit)
	{
		if (count == 0)
		{
			return;
		}

		Span<FactorEntry> buffer = factors.AsSpan(0, count);
		buffer.Sort(static (a, b) => a.Value.CompareTo(b.Value));
		BuildCandidatesRecursive(order, buffer, 0, 1UL, candidates, limit);
	}

	private static void BuildCandidatesRecursive(ulong order, in ReadOnlySpan<FactorEntry> factors, int index, ulong divisorProduct, List<ulong> candidates, int limit)
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

		FactorEntry factor = factors[index];
		int factorExponent = factor.Exponent;
		ulong primeFactor = factor.Value;
		ulong contribution = 1UL;
		ulong contributionLimit = factorExponent == 0 ? 0UL : order / primeFactor;
		for (int exponent = 0; exponent <= factorExponent; exponent++)
		{
			ulong nextDivisor = divisorProduct * contribution;
			if (nextDivisor > order)
			{
				break;
			}

			BuildCandidatesRecursive(order, factors, index + 1, nextDivisor, candidates, limit);
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

			contribution *= primeFactor;
		}
	}

	private static bool TryConfirmCandidateCpu(ulong prime, ulong candidate, in MontgomeryDivisorData divisorData, in PrimeOrderSearchConfig config, ref int powUsed, int powBudget)
	{
		PartialFactorResult factorization = PartialFactor(candidate, config);
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

			ReadOnlySpan<FactorEntry> span = factorization.Factors;
			int length = factorization.Count;

			const int StackExponentCapacity = 32;
			const int ExponentHardLimit = 256;

			Span<ulong> stackBuffer = stackalloc ulong[StackExponentCapacity];
			ulong[]? heapCandidateArray = null;
			ExponentRemainderStepperCpu stepper = ThreadStaticPools.RentExponentStepperCpu(divisorData);
			bool violates = false;

			for (int i = 0; i < length; i++)
			{
				ulong primeFactor = span[i].Value;
				int exponent = span[i].Exponent;
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

	private static bool Pow2EqualsOneCpu(ulong exponent, ulong prime, in MontgomeryDivisorData divisorData)
	{
		if (IsGpuPow2Allowed)
		{
			ulong remainder = exponent.Pow2MontgomeryModWindowedGpu(divisorData, false);
			// GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2Mod(exponent, prime, out ulong remainder, divisorData);
			// if (status == GpuPow2ModStatus.Success)
			// {
			return remainder == 1UL;
			// }
		}

		return exponent.Pow2MontgomeryModWindowedCpu(divisorData, keepMontgomery: false) == 1UL;
	}

	// private static ulong _partialFactorHits;
	// private static ulong _partialFactorPendingHits;
	// private static ulong _partialFactorCofactorHits;

	private static PartialFactorResult PartialFactor(ulong value, in PrimeOrderSearchConfig config)
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
		// We don't need to worry about leftovers, because we always use indexes within the calculated counts
		// primeSlots.Clear();
		// exponentSlots.Clear();

		int factorCount = 0;
		Dictionary<ulong, int>? counts = null;
		bool useDictionary = false;

		uint limit = config.SmallFactorLimit == 0 ? uint.MaxValue : config.SmallFactorLimit;
		ulong remaining = value;

		bool gpuFactored = false;
		if (IsGpuHeuristicDevice)
		{
			gpuFactored = PrimeOrderGpuHeuristics.TryPartialFactor(value, limit, primeSlots, exponentSlots, out factorCount, out remaining, out _);
		}

		List<PendingEntry> pending = ThreadStaticPools.RentPrimeOrderPendingEntryList(2);
		Stack<ulong>? compositeStack = null;
		PartialFactorResult result;
		PrimeTester primeTester = PrimeTester.Exclusive;
		CancellationToken ct = CancellationToken.None;
		HeuristicPrimeTester? tester = _tester ??= new();

		if (!gpuFactored)
		{
			// We don't need to worry about leftovers, because we always use indexes within the calculated counts
			// primeSlots.Clear();
			// exponentSlots.Clear();

			counts = ThreadStaticPools.RentUlongIntDictionary(Math.Max(FactorSlotCount, 8));
			counts.Clear();

			// bool gpuPopulated = TryPopulateSmallPrimeFactorsGpu(value, limit, primeSlots, exponentSlots, out factorCount, out remaining);
			// if (!gpuPopulated)
			// {
			// 	throw new InvalidOperationException($"GPU didn't populate factors for {value}");
			// counts.Clear();
			remaining = PopulateSmallPrimeFactorsCpu(value, limit, counts);
			// }

			int dictionaryCount = counts.Count;
			if (dictionaryCount <= FactorSlotCount)
			{
				int copyIndex = 0;
				foreach (KeyValuePair<ulong, int> entry in counts)
				{
					primeSlots[copyIndex] = entry.Key;
					exponentSlots[copyIndex] = entry.Value;
					copyIndex++;
				}

				factorCount = copyIndex;
				ThreadStaticPools.ReturnUlongIntDictionary(counts);
				counts = null;
			}
			else
			{
				useDictionary = true;
				factorCount = dictionaryCount;
			}
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
					bool isPrime = tester.IsPrimeGpu(composite);
					// bool isPrime = Open.Numeric.Primes.Prime.Numbers.IsPrime(composite);

					if (isPrime)
					{
						AddFactorToCollector(ref useDictionary, ref counts, primeSlots, exponentSlots, ref factorCount, composite, 1);
						continue;
					}

					long currentTimestamp = Stopwatch.GetTimestamp();
					if (currentTimestamp > deadlineTimestamp)
					{
						pollardRhoDeadlineReached = true;
						pending.Add(new PendingEntry(composite, knownComposite: true));
						continue;
					}

					if (!TryPollardRho(composite, deadlineTimestamp, out ulong factor))
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
				bool isPrime = tester.IsPrimeGpu(composite);
				// bool isPrime = Open.Numeric.Primes.Prime.Numbers.IsPrime(composite);

				entry = entry.WithPrimality(isPrime);
				pending[index] = entry;
			}

			if (entry.IsPrime)
			{
				AddFactorToCollector(ref useDictionary, ref counts, primeSlots, exponentSlots, ref factorCount, composite, 1);
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

			cofactorIsPrime = tester.IsPrimeGpu(cofactor);
			// cofactorIsPrime = PrimeTester.IsPrimeGpu(cofactor);
			// cofactorIsPrime = Open.Numeric.Primes.Prime.Numbers.IsPrime(cofactor);
		}

		ArrayPool<FactorEntry> pool = ThreadStaticPools.FactorEntryPool;
		bool temp;
		if (useDictionary)
		{
			temp = counts is null || counts.Count == 0;
			if (temp && cofactor == value)
			{
				result = PartialFactorResult.Rent(null, cofactor, false, 0, cofactorIsPrime);
				goto ReturnResult;
			}

			if (temp)
			{
				result = PartialFactorResult.Rent(null, cofactor, cofactor == 1UL, 0, cofactorIsPrime);
				goto ReturnResult;
			}

			FactorEntry[] factors = pool.Rent(counts!.Count);
			index = 0;
			foreach (KeyValuePair<ulong, int> entry in counts)
			{
				factors[index] = new FactorEntry(entry.Key, entry.Value, true);
				index++;
			}

			Array.Sort(factors, static (a, b) => a.Value.CompareTo(b.Value));

			temp = cofactor == 1UL;
			result = PartialFactorResult.Rent(factors, cofactor, temp, index, cofactorIsPrime);
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

		FactorEntry[] array = pool.Rent(factorCount);
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

			array[arrayIndex] = new FactorEntry(primeValue, exponentValue, true);
			arrayIndex++;
		}

		Span<FactorEntry> arraySpan = array.AsSpan(0, arrayIndex);
		arraySpan.Sort(static (a, b) => a.Value.CompareTo(b.Value));
		temp = cofactor == 1UL;
		result = PartialFactorResult.Rent(array, cofactor, temp, arrayIndex, cofactorIsPrime);

	ReturnResult:
		// ThreadStaticPools.UlongPool.Return(primeSlotsArray);
		// ThreadStaticPools.IntPool.Return(exponentSlotsArray);


		if (counts is not null)
		{
			counts.Clear();
			ThreadStaticPools.ReturnUlongIntDictionary(counts);
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


	private static bool TryPopulateSmallPrimeFactorsCpu(
			ulong value,
			uint limit,
			Span<ulong> primeTargets,
			Span<int> exponentTargets,
			out int factorCount,
			out ulong remaining)
	{
		factorCount = 0;
		remaining = value;

		// value will never <= 1 in production code
		// if (value <= 1UL)
		// {
		//     return true;
		// }

		int capacity = Math.Min(primeTargets.Length, exponentTargets.Length);
		// capacity will never equal 0 in production code
		// if (capacity == 0)
		// {
		// 	return false;
		// }

		uint[] primes = PrimesGenerator.SmallPrimes;
		ulong[] squares = PrimesGenerator.SmallPrimesPow2;
		int primeCount = primes.Length;
		ulong remainingLocal = value;
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

			// primeCandidate will never equal 0 in production code
			// if (primeCandidate == 0UL)
			// {
			// 	continue;
			// }

			ulong primeValue = primeCandidate;
			if ((remainingLocal % primeValue) != 0UL)
			{
				continue;
			}

			int exponent = ExtractSmallPrimeExponent(ref remainingLocal, primeValue);
			if (exponent == 0)
			{
				continue;
			}

			if (factorCount >= capacity)
			{
				throw new InvalidOperationException($"Capacity is smaller than factor count");
			}

			primeTargets[factorCount] = primeValue;
			exponentTargets[factorCount] = exponent;
			factorCount++;
		}

		remaining = remainingLocal;
		return true;
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

	private static ulong PopulateSmallPrimeFactorsCpu(ulong value, uint limit, Dictionary<ulong, int> counts)
	{
		ulong remaining = value;
		uint[] primes = PrimesGenerator.SmallPrimes;
		ulong[] squares = PrimesGenerator.SmallPrimesPow2;
		int primeCount = primes.Length;

		for (int i = 0; i < primeCount; i++)
		{
			uint primeCandidate = primes[i];
			if (primeCandidate > limit)
			{
				break;
			}

			if (squares[i] > remaining)
			{
				break;
			}

			if ((remaining % primeCandidate) != 0UL)
			{
				continue;
			}

			int exponent = 0;
			do
			{
				// TODO: Implement DivRem like solution or residue stepper to re-use previous iteration results in new calculation. Add / use method, likely to ULongExtensions
				remaining /= primeCandidate;
				exponent++;
			}
			while ((remaining % primeCandidate) == 0UL);

			counts[primeCandidate] = exponent;
		}

		return remaining;
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

	private static bool TryPollardRho(ulong n, long deadlineTimestamp, out ulong factor)
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

			ulong c = (DeterministicRandom.NextUInt64() % (n - 1UL)) + 1UL;
			ulong x = (DeterministicRandom.NextUInt64() % (n - 2UL)) + 2UL;
			ulong y = x;
			ulong d = 1UL;

			while (d == 1UL)
			{
				timestamp = Stopwatch.GetTimestamp();
				if (timestamp > deadlineTimestamp)
				{
					return false;
				}

				x = AdvancePolynomial(x, c, n);
				y = AdvancePolynomial(y, c, n);
				y = AdvancePolynomial(y, c, n);
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
	private static ulong AdvancePolynomial(ulong x, ulong c, ulong modulus)
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

	private static bool TryAppendFactor(Span<ulong> primes, Span<int> exponents, ref int count, ulong prime, int exponent)
	{
		if (exponent <= 0)
		{
			return true;
		}

		for (int i = 0; i < count; i++)
		{
			if (primes[i] == prime)
			{
				exponents[i] += exponent;
				return true;
			}
		}

		if (count >= primes.Length)
		{
			return false;
		}

		primes[count] = prime;
		exponents[count] = exponent;
		count++;
		return true;
	}

	private static void AddFactorToCollector(
		ref bool useDictionary,
		ref Dictionary<ulong, int>? counts,
		Span<ulong> primes,
		Span<int> exponents,
		ref int count,
		ulong prime,
		int exponent)
	{
		if (exponent <= 0)
		{
			return;
		}

		if (useDictionary)
		{
			AddFactor(counts!, prime, exponent);
			return;
		}

		if (TryAppendFactor(primes, exponents, ref count, prime, exponent))
		{
			return;
		}

		counts ??= ThreadStaticPools.RentUlongIntDictionary(Math.Max(count, 8));
		CopyFactorsToDictionary(primes, exponents, count, counts);
		count = 0;
		useDictionary = true;
		AddFactor(counts, prime, exponent);
	}

	private static void CopyFactorsToDictionary(
		in ReadOnlySpan<ulong> primes,
		in ReadOnlySpan<int> exponents,
		int count,
		Dictionary<ulong, int> target)
	{
		for (int i = 0; i < count; i++)
		{
			ulong prime = primes[i];
			int exponent = exponents[i];
			if (prime == 0UL || exponent == 0)
			{
				continue;
			}

			target[prime] = exponent;
		}
	}

	private static ulong CalculateByFactorizationCpu(ulong prime, in MontgomeryDivisorData divisorData)
	{
		ulong phi = prime - 1UL;
		Dictionary<ulong, int> counts = new(capacity: 8);
		FactorCompletely(phi, counts);
		if (counts.Count == 0)
		{
			return phi;
		}

		List<KeyValuePair<ulong, int>> entries = [.. counts];
		entries.Sort(static (a, b) => a.Key.CompareTo(b.Key));

		ulong order = phi;
		int entryCount = entries.Count;
		for (int i = 0; i < entryCount; i++)
		{
			ulong primeFactor = entries[i].Key;
			int exponent = entries[i].Value;
			for (int iteration = 0; iteration < exponent; iteration++)
			{
				if ((order % primeFactor) != 0UL)
				{
					break;
				}

				ulong candidate = order / primeFactor;
				if (Pow2EqualsOneCpu(candidate, prime, divisorData))
				{
					order = candidate;
					continue;
				}

				break;
			}
		}

		return order;
	}

	private static void FactorCompletely(ulong value, Dictionary<ulong, int> counts)
	{
		FactorCompletely(value, counts, knownComposite: false);
	}

	// private static ulong _factorCompletelyHits;
	// private static ulong _factorCompletelyPollardRhoHits;
	[ThreadStatic]
	private static HeuristicPrimeTester? _tester;

	private static void FactorCompletely(ulong value, Dictionary<ulong, int> counts, bool knownComposite)
	{
		if (value <= 1UL)
		{
			return;
		}

		// Atomic.Add(ref _factorCompletelyHits, 1UL);
		// Console.WriteLine($"Factor completely hits {Volatile.Read(ref _factorCompletelyHits)}");
		HeuristicPrimeTester tester = _tester ??= new();

		// bool isPrime = primeTester.HeuristicIsPrimeGpu(value);
		bool isPrime = tester.IsPrimeGpu(value);
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
		isPrime = tester.IsPrimeGpu(factor);
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

			isPrime = tester.IsPrimeGpu(quotient);
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
			ulong c = (DeterministicRandom.NextUInt64() % (n - 1UL)) + 1UL;
			ulong x = (DeterministicRandom.NextUInt64() % (n - 2UL)) + 2UL;
			ulong y = x;
			ulong d = 1UL;

			while (d == 1UL)
			{
				x = AdvancePolynomial(x, c, n);
				y = AdvancePolynomial(y, c, n);
				y = AdvancePolynomial(y, c, n);
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

