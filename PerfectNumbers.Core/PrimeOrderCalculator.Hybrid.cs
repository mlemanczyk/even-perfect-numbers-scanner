using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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

		PartialFactorResult phiFactors = PartialFactorHybrid(gpu, divisorData, phi, config);

		ulong result;
		if (phiFactors.Count == 0)
		{
			result = CalculateByFactorizationCpu(prime, divisorData, phiFactors);

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
			FixedCapacityStack<MontgomeryDivisorData> divisorPool = MontgomeryDivisorDataPool.Shared;
			divisorData = divisorPool.FromModulus(prime64);
			ulong order64 = CalculateHybrid(gpu, prime64, previous, divisorData, config);
			divisorPool.Return(divisorData);
			result = order64 == 0UL ? UInt128.Zero : (UInt128)order64;
		}
		else
		{
			result = CalculateWideInternalCpu(prime, previousOrder, config);
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
	private static PartialFactorResult PartialFactorHybrid(PrimeOrderCalculatorAccelerator gpu, in MontgomeryDivisorData divisorData, ulong value, in PrimeOrderCalculatorConfig config)
	{
		if (value <= 1UL)
		{
			return PartialFactorResult.Empty;
		}

		// stackalloc is faster than pooling
		PartialFactorResult result = PartialFactorResult.Rent(divisorData);
		Span<ulong> primeSlots = new (result.Factors);
		Span<int> exponentSlots = new (result.Exponents);
		// We don't need to worry about leftovers, because we always use indexes within the calculated counts
		// primeSlots.Clear();
		// exponentSlots.Clear();

		uint limit = config.SmallFactorLimit;

		List<PartialFactorPendingEntry> pending = result.Pending;
		pending.Clear();

		PopulateSmallPrimeFactorsCpu(
			value,
			limit,
			primeSlots,
			exponentSlots,
			out int factorCount,
			out ulong remaining);

		if (remaining > 1UL)
		{
			if (config.PollardRhoMilliseconds > 0)
			{
				long deadlineTimestamp = CreateDeadlineTimestamp(config.PollardRhoMilliseconds);
				FixedCapacityStack<ulong> compositeStack = result.CompositeStack;
				compositeStack.Clear();
				compositeStack.Push(remaining);

				CollectFactorsHybrid(gpu, primeSlots, exponentSlots, ref factorCount, pending, compositeStack, deadlineTimestamp, out bool limitReached);

				if (limitReached)
				{
					while (compositeStack.Count > 0)
					{
						pending.Add(PartialFactorPendingEntry.Rent(compositeStack.Pop(), knownComposite: false));
					}
				}
			}
			else
			{
				pending.Add(PartialFactorPendingEntry.Rent(remaining, knownComposite: false));
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
				PartialFactorPendingEntry.Return(entry);
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

			PartialFactorPendingEntry.Return(entry);
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
		// 	result.Cofactor = cofactor;
		// 	result.FullyFactored = cofactor != value && cofactor == 1UL;
		// 	result.Count = 0;
		// 	result.CofactorIsPrime = cofactorIsPrime;
		// 	goto ReturnResult;
		// }

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

		Array.Sort(result.Factors, result.Exponents, 0, factorCount);

		// Check if it was factored completely
		result.Cofactor = cofactor;
		result.FullyFactored = cofactor == 1UL;
		result.Count = factorCount;
		result.CofactorIsPrime = cofactorIsPrime;
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
		if (phiFactors.FullyFactored && TrySpecialMaxHybrid(gpu, phi, phiFactors, divisorData))
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
			return CalculateByFactorizationCpu(prime, divisorData, phiFactors);
		}

		if (TryHeuristicFinishHybrid(gpu, prime, candidateOrder, previousOrder, divisorData, config, orderFactors, out ulong order))
		{
			return order;
		}

		return candidateOrder;
	}

	private static int _cpuCount;

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static bool RunOnCpu()
	{
		if (PerfectNumberConstants.GpuRatio == 0)
		{
			return false;
		}

		if (PerfectNumberConstants.GpuRatio == 1)
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
		if (cpuCount == PerfectNumberConstants.GpuRatio)
		{
			Interlocked.Add(ref _cpuCount, -PerfectNumberConstants.GpuRatio);
		}
		else if (cpuCount > PerfectNumberConstants.GpuRatio)
		{
			cpuCount -= PerfectNumberConstants.GpuRatio;
		}

		return cpuCount != PerfectNumberConstants.GpuRatio;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static bool TryConfirmCandidateHybrid(PrimeOrderCalculatorAccelerator gpu, ulong prime, ulong candidate, in MontgomeryDivisorData divisorData, in PrimeOrderCalculatorConfig config, ref int powUsed, int powBudget)
	{
		PartialFactorResult factorization = PartialFactorHybrid(gpu, divisorData, candidate, config);

		if (factorization.Count == 0)
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

		Span<ulong> stackBuffer = new(factorization.SpecialMaxBuffer);
		ExponentRemainderStepperCpu stepper = factorization.ExponentRemainderStepper;
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

			if (CheckCandidateViolation(stackBuffer[..exponent], primeFactor, exponent, candidate, prime, ref powUsed, powBudget, ref stepper))
			{
				violates = true;
				break;
			}
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

		PartialFactorResult factorization = PartialFactorHybrid(gpu, divisorData, order, config);
		if (factorization.Count == 0)
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

		if (!ValidateOrderAgainstFactors(order, factorization))
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
		PartialFactorResult orderFactors = cachedOrderFactors ?? PartialFactorHybrid(gpu, divisorData, order, config);
		try
		{
			if (orderFactors.Count == 0)
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
			List<ulong> candidates = orderFactors.FactorCandidatesList;
			candidates.Clear();

			BuildCandidates(order, orderFactors, orderFactors.Count, candidates, capacity);
			int candidateCount = candidates.Count;
			if (candidateCount == 0)
			{
				return false;
			}

			SortCandidates(prime, previousOrder, candidates);

			int powBudget = config.MaxPowChecks <= 0 ? candidateCount : config.MaxPowChecks;
			int powUsed = 0;
			Span<ulong> candidateSpan = CollectionsMarshal.AsSpan(candidates);
			ExponentRemainderStepperCpu powStepper = orderFactors.ExponentRemainderStepper;
			powStepper.Reset();

			ulong candidate = candidateSpan[0];
			powUsed++;

			bool equalsOne = powStepper.InitializeCpuIsUnity(candidate);
			if (equalsOne && TryConfirmCandidateHybrid(gpu, prime, candidate, divisorData, config, ref powUsed, powBudget))
			{
				result = candidate;
				return true;
			}

			int index = 1;
			while (index < candidateCount && powUsed < powBudget)
			{
				int budgetRemaining = powBudget - powUsed;
				if (budgetRemaining <= 0)
				{
					break;
				}

				candidate = candidateSpan[index++];
				powUsed++;

				equalsOne = powStepper.ComputeNextIsUnity(candidate);
				if (!equalsOne)
				{
					continue;
				}

				if (!TryConfirmCandidateHybrid(gpu, prime, candidate, divisorData, config, ref powUsed, powBudget))
				{
					continue;
				}

				result = candidate;
				return true;
			}

			return false;
		}
		finally
		{
			orderFactors.Dispose();
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static bool TrySpecialMaxHybrid(PrimeOrderCalculatorAccelerator gpu, ulong phi, PartialFactorResult factors, in MontgomeryDivisorData divisorData)
	{
		int length = factors.Count;
		// The phi factorization on the scanner path always yields at least one entry for phi >= 2,
		// preventing the zero-length case from occurring outside synthetic tests.
		// if (length == 0)
		// {
		//     return true;
		// }

		if (length <= 8)
		{
			return EvaluateSpecialMaxCandidatesCpu(factors, phi);
		}

		ReadOnlySpan<ulong> factorSpan = new(factors.Factors, 0, length);
		return EvaluateSpecialMaxCandidatesGpu(gpu, factorSpan, phi, divisorData);
	}

}
