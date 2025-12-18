using System.Buffers;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ILGPU;
using PerfectNumbers.Core.Gpu;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core;

internal static partial class PrimeOrderCalculator
{
	public static ulong CalculateHybrid(
		PrimeOrderCalculatorAccelerator gpu,
		ulong prime,
		ulong? previousOrder,
		in MontgomeryDivisorData divisorData,
		in PrimeOrderCalculatorConfig config)
	{
		// TODO: Is this condition ever met on EvenPerfectBitScanner's execution path? If not, we can add a clarification comment and comment out the entire block. We want to support p candidates at least greater or equal to 31.
		if (prime <= 3UL)
		{
			return prime == 3UL ? 2UL : 1UL;
		}

		ulong phi = prime - 1UL;

		PartialFactorResult phiFactors = PartialFactorHybrid(gpu, phi, divisorData, config);

		ulong result;
		if (phiFactors.Factors is null)
		{
			result = CalculateByFactorizationCpu(prime, divisorData);

			phiFactors.Dispose();
			return result;
		}

		result = RunHeuristicPipelineHybrid(gpu, prime, previousOrder, config, divisorData, phi, phiFactors);

		phiFactors.Dispose();
		return result;
	}

	public static UInt128 CalculateHybrid(
			PrimeOrderCalculatorAccelerator gpu,
			in UInt128 prime,
			in UInt128? previousOrder,
			in PrimeOrderCalculatorConfig config)
	{
		MontgomeryDivisorData divisorData;
		UInt128 result;
		if (prime <= ulong.MaxValue)
		{
			ulong? previous = null;
			if (previousOrder.HasValue)
			{
				UInt128 previousValue = previousOrder.Value;
				if (previousValue <= ulong.MaxValue)
				{
					previous = (ulong)previousValue;
				}
				else
				{
					previous = ulong.MaxValue;
				}
			}

			ulong prime64 = (ulong)prime;
			divisorData = MontgomeryDivisorData.FromModulus(prime64);
			ulong order64 = CalculateHybrid(gpu, prime64, previous, divisorData, config);
			result = order64 == 0UL ? UInt128.Zero : (UInt128)order64;
		}
		else
		{
			divisorData = MontgomeryDivisorData.Empty;
			result = CalculateWideInternalCpu(prime, previousOrder, divisorData, config);
		}

		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static void CollectFactorsHybrid(PrimeOrderCalculatorAccelerator gpu, Span<ulong> primeSlots, Span<int> exponentSlots, ref int factorCount, List<PartialFactorPendingEntry> pending, FixedCapacityStack<ulong> compositeStack, long deadlineTimestamp, out bool pollardRhoDeadlineReached)
	{
		while (compositeStack.Count > 0)
		{
			ulong composite = compositeStack.Pop();
			bool isPrime = RunOnCpu() ? PrimeTesterByLastDigit.IsPrimeCpu(composite) : HeuristicCombinedPrimeTester.IsPrimeGpu(gpu, composite);

			if (isPrime)
			{
				AddFactorToCollector(primeSlots, exponentSlots, ref factorCount, composite);
				continue;
			}

			pollardRhoDeadlineReached = Stopwatch.GetTimestamp() > deadlineTimestamp;
			if (pollardRhoDeadlineReached)
			{
				pending.Add(new PartialFactorPendingEntry(composite, knownComposite: true));
				continue;
			}

			if (!TryPollardRhoCpu(composite, deadlineTimestamp, out ulong factor))
			{
				pending.Add(new PartialFactorPendingEntry(composite, knownComposite: true));
				continue;
			}

			ulong quotient = composite / factor;
			compositeStack.Push(factor);
			compositeStack.Push(quotient);
		}

		pollardRhoDeadlineReached = false;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static ulong InitializeStartingOrderHybrid(PrimeOrderCalculatorAccelerator gpu, ulong prime, ulong phi, in MontgomeryDivisorData divisorData)
	{
		ulong order = phi;
		if ((prime & 7UL) == 1UL || (prime & 7UL) == 7UL)
		{
			ulong half = phi >> 1;
			if (Pow2EqualsOneHybrid(gpu, half, prime, divisorData))
			{
				order = half;
			}
		}

		return order;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static PartialFactorResult PartialFactorHybrid(PrimeOrderCalculatorAccelerator gpu, ulong value, in MontgomeryDivisorData divisorData, in PrimeOrderCalculatorConfig config)
	{
		if (value <= 1UL)
		{
			return PartialFactorResult.Empty;
		}

		// stackalloc is faster than pooling

		var result = PartialFactorResult.Rent(divisorData);
		ulong[] factors = result.Factors;
		Span<ulong> primeSlots = new(factors);
		int[] exponents = result.Exponents;
		Span<int> exponentSlots = new(exponents);
		// We don't need to worry about leftovers, because we always use indexes within the calculated counts
		// primeSlots.Clear();
		// exponentSlots.Clear();

		uint limit = config.SmallFactorLimit == 0 ? uint.MaxValue : config.SmallFactorLimit;

		List<PartialFactorPendingEntry> pending = ThreadStaticPools.RentPrimeOrderPendingEntryList(2);

		PopulateSmallPrimeFactorsCpu(
			value,
			limit,
			primeSlots,
			exponentSlots,
			out int factorCount,
			out ulong remaining);

		bool limitReached;
		if (remaining > 1UL)
		{
			if (config.PollardRhoMilliseconds > 0)
			{
				long deadlineTimestamp = CreateDeadlineTimestamp(config.PollardRhoMilliseconds);
				FixedCapacityStack<ulong> compositeStack = ThreadStaticPools.RentUlongStack(4);
				compositeStack.Push(remaining);

				CollectFactorsHybrid(gpu, primeSlots, exponentSlots, ref factorCount, pending, compositeStack, deadlineTimestamp, out limitReached);

				if (limitReached)
				{
					while (compositeStack.Count > 0)
					{
						pending.Add(new PartialFactorPendingEntry(compositeStack.Pop(), knownComposite: false));
					}
				}

				compositeStack.Clear();
				ThreadStaticPools.ReturnUlongStack(compositeStack);
			}
			else
			{
				pending.Add(new PartialFactorPendingEntry(remaining, knownComposite: false));
			}
		}

		ulong cofactor = 1UL;
		bool cofactorContainsComposite = false;
		int pendingCount = pending.Count;
		int index = 0;
		for (; index < pendingCount; index++)
		{
			PartialFactorPendingEntry entry = pending[index];
			ulong composite = entry.Value;

			if (entry.KnownComposite)
			{
				cofactor = checked(cofactor * composite);
				cofactorContainsComposite = true;
				continue;
			}

			if (!entry.HasKnownPrimality)
			{
				bool isPrime = RunOnCpu() ? PrimeTesterByLastDigit.IsPrimeCpu(composite) : HeuristicCombinedPrimeTester.IsPrimeGpu(gpu, composite);

				entry.WithPrimality(isPrime);
				pending[index] = entry;
			}

			if (entry.IsPrime)
			{
				AddFactorToCollector(primeSlots, exponentSlots, ref factorCount, composite);
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
			cofactorIsPrime = RunOnCpu() ? PrimeTesterByLastDigit.IsPrimeCpu(cofactor) : HeuristicCombinedPrimeTester.IsPrimeGpu(gpu, cofactor);
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

		Array.Sort(factors, exponents, 0, factorCount);

		result.Cofactor = cofactor;
		result.FullyFactored = cofactor == 1UL;
		result.Count = factorCount;
		result.CofactorIsPrime = cofactorIsPrime;

		pending.Clear();
		ThreadStaticPools.ReturnPrimeOrderPendingEntryList(pending);
		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static bool Pow2EqualsOneHybrid(PrimeOrderCalculatorAccelerator gpu, ulong exponent, ulong prime, in MontgomeryDivisorData divisorData)
	{
		// if (exponent >= PerfectNumberConstants.MaxQForDivisorCycles || prime >= PerfectNumberConstants.MaxQForDivisorCycles)
		// {
		// 	// Atomic.Add(ref _pow2GpuHits, 1);
		// 	// Console.WriteLine($"pow2 GPU Hits {_pow2GpuHits}");
		// 	ulong remainder = exponent.Pow2MontgomeryModWindowedConvertGpu(gpu, divisorData);
		// 	// GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2Mod(exponent, prime, out ulong remainder, divisorData);
		// 	// if (status == GpuPow2ModStatus.Success)
		// 	// {
		// 	return remainder == 1UL;
		// 	// }
		// }

		// Atomic.Add(ref _pow2CpuHits, 1);
		// Console.WriteLine($"pow2 CPU Hits {_pow2CpuHits}");
		return exponent.Pow2MontgomeryModWindowedConvertToStandardCpu(divisorData) == 1UL;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static ulong RunHeuristicPipelineHybrid(
		PrimeOrderCalculatorAccelerator gpu,
		ulong prime,
		ulong? previousOrder,
		in PrimeOrderCalculatorConfig config,
		in MontgomeryDivisorData divisorData,
		ulong phi,
		PartialFactorResult phiFactors)
	{
		if (phiFactors.FullyFactored && TrySpecialMaxHybrid(gpu, phi, prime, phiFactors, divisorData))
		{
			return phi;
		}

		ulong candidateOrder = InitializeStartingOrderHybrid(gpu, prime, phi, divisorData);
		candidateOrder = ExponentLoweringCpu(candidateOrder, phiFactors, divisorData);

		if (TryConfirmOrderHybrid(gpu, prime, candidateOrder, divisorData, config, out PartialFactorResult? orderFactors))
		{
			return candidateOrder;
		}

		if (config.StrictMode)
		{
			orderFactors?.Dispose();
			return CalculateByFactorizationCpu(prime, divisorData);
		}

		if (TryHeuristicFinishHybrid(gpu, prime, candidateOrder, previousOrder, divisorData, config, phiFactors, orderFactors, out ulong order))
		{
			return order;
		}

		return candidateOrder;
	}

	private static int _cpuCount;

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static bool RunOnCpu()
	{
		int gpuRatio = EnvironmentConfiguration.GpuRatio;
		if (gpuRatio == 0)
		{
			return false;
		}

		if (gpuRatio == 1)
		{
			while (true)
			{
				if (Interlocked.CompareExchange(ref _cpuCount, 0, 1) == 0)
				{
					return true;
				}

				if (Interlocked.CompareExchange(ref _cpuCount, 1, 0) == 1)
				{
					return false;
				}

				Thread.Yield();
			}
		}

		int cpuCount = Interlocked.Increment(ref _cpuCount);
		if (cpuCount == gpuRatio)
		{
			Interlocked.Add(ref _cpuCount, -gpuRatio);
		}
		else if (cpuCount > gpuRatio)
		{
			cpuCount -= gpuRatio;
		}

		return cpuCount != gpuRatio;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static bool TryConfirmCandidateHybrid(PrimeOrderCalculatorAccelerator gpu, ulong prime, ulong candidate, in MontgomeryDivisorData divisorData, in PrimeOrderCalculatorConfig config, ref int powUsed, int powBudget)
	{
		PartialFactorResult factorization = PartialFactorHybrid(gpu, candidate, divisorData, config);

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

			factorization.WithAdditionalPrime(factorization.Cofactor);
		}

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
		factorization.Dispose();

		return !violates;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static bool TryConfirmOrderHybrid(
		PrimeOrderCalculatorAccelerator gpu,
		ulong prime,
		ulong order,
		in MontgomeryDivisorData divisorData,
		in PrimeOrderCalculatorConfig config,
		out PartialFactorResult? reusableFactorization)
	{
		reusableFactorization = null;

		if (order == 0UL)
		{
			return false;
		}

		if (!Pow2EqualsOneHybrid(gpu, order, prime, divisorData))
		{
			return false;
		}

		PartialFactorResult factorization = PartialFactorHybrid(gpu, order, divisorData, config);
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

			factorization.WithAdditionalPrime(factorization.Cofactor);
		}

		if (!ValidateOrderAgainstFactors(prime, order, divisorData, factorization))
		{
			reusableFactorization = factorization;
			return false;
		}

		factorization.Dispose();
		return true;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static bool TryHeuristicFinishHybrid(
		PrimeOrderCalculatorAccelerator gpu,
		ulong prime,
		ulong order,
		ulong? previousOrder,
		in MontgomeryDivisorData divisorData,
		in PrimeOrderCalculatorConfig config,
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
		PartialFactorResult orderFactors = cachedOrderFactors ?? PartialFactorHybrid(gpu, order, divisorData, config);
		try
		{
			if (orderFactors.Factors is null)
			{
				return false;
			}

			if (!orderFactors.FullyFactored)
			{
				if (orderFactors.Cofactor <= 1UL || !orderFactors.CofactorIsPrime)
				{
					return false;
				}

				orderFactors.WithAdditionalPrime(orderFactors.Cofactor);
			}

			int capacity = config.MaxPowChecks <= 0 ? 64 : config.MaxPowChecksCapacity;
			List<ulong> candidates = ThreadStaticPools.RentUlongList(capacity);
			candidates.Clear();
			ulong[] factorArray = orderFactors.Factors!;
			int[] exponentsArray = orderFactors.Exponents!;

			BuildCandidates(order, factorArray, exponentsArray, orderFactors.Count, candidates, capacity);
			int candidateCount = candidates.Count;
			if (candidateCount == 0)
			{
				ThreadStaticPools.ReturnUlongList(candidates);
				return false;
			}

			SortCandidates(prime, previousOrder, candidates);

			int powBudget = config.MaxPowChecks <= 0 ? candidateCount : config.MaxPowChecks;
			int powUsed = 0;
			Span<ulong> candidateSpan = CollectionsMarshal.AsSpan(candidates);
			ExponentRemainderStepperCpu powStepper = ThreadStaticPools.RentExponentStepperCpu(divisorData);
			bool powStepperInitialized = false;

			int index = 0;
			Span<ulong> stackGpuRemainders = stackalloc ulong[PerfectNumberConstants.DefaultSmallPrimeFactorSlotCount];
			ArrayPool<ulong> pool = ThreadStaticPools.UlongPool;
			while (index < candidateCount && powUsed < powBudget)
			{
				int remaining = candidateCount - index;
				int budgetRemaining = powBudget - powUsed;
				int batchSize = Math.Min(remaining, Math.Min(budgetRemaining, PrimeOrderConstants.MaxGpuBatchSize));
				if (batchSize <= 0)
				{
					break;
				}

				ReadOnlySpan<ulong> batch = candidateSpan.Slice(index, batchSize);
				int startIndex = 0;

				if (!powStepperInitialized)
				{
					ulong candidate = batch[0];
					powUsed++;

					bool equalsOne = powStepper.InitializeCpuIsUnity(candidate);
					powStepperInitialized = true;
					if (equalsOne && TryConfirmCandidateHybrid(gpu, prime, candidate, divisorData, config, ref powUsed, powBudget))
					{
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

					bool equalsOne = powStepper.ComputeNextIsUnity(candidate);
					if (!equalsOne)
					{
						continue;
					}

					if (!TryConfirmCandidateHybrid(gpu, prime, candidate, divisorData, config, ref powUsed, powBudget))
					{
						continue;
					}

					candidates.Clear();
					ThreadStaticPools.ReturnUlongList(candidates);
					ThreadStaticPools.ReturnExponentStepperCpu(powStepper);
					result = candidate;
					return true;
				}

				index += batchSize;
			}

			candidates.Clear();
			ThreadStaticPools.ReturnUlongList(candidates);
			ThreadStaticPools.ReturnExponentStepperCpu(powStepper);

			return false;
		}
		finally
		{
			orderFactors.Dispose();
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static bool TrySpecialMaxHybrid(PrimeOrderCalculatorAccelerator gpu, ulong phi, ulong prime, PartialFactorResult factors, in MontgomeryDivisorData divisorData)
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

		if (length <= 8)
		{
			return EvaluateSpecialMaxCandidatesCpu(stackBuffer, factorSpan, phi, prime, divisorData);
		}

		return EvaluateSpecialMaxCandidatesGpu(gpu, factorSpan, phi, prime, divisorData);
	}

}
