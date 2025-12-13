using System.Buffers;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using PerfectNumbers.Core.Cpu;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

internal static partial class PrimeOrderCalculator
{
	public static ulong CalculateCpu(
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

		PartialFactorResult phiFactors = PartialFactorCpu(phi, config);

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

	public static UInt128 CalculateCpu(
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
			ulong order64 = CalculateCpu(prime64, previous, divisorData, config);
			divisorPool.Return(divisorData);
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
	private static UInt128 CalculateWideInternalCpu(in UInt128 prime, in UInt128? previousOrder, in MontgomeryDivisorData? divisorData, in PrimeOrderCalculatorConfig config)
	{
		if (prime <= UInt128.One)
		{
			return UInt128.One;
		}

		if (prime == (UInt128)3UL)
		{
			return (UInt128)2UL;
		}

		UInt128 phi = prime - UInt128.One;
		PartialFactorResult128 phiFactors = PartialFactorWide(phi, config);
		UInt128 result;
		if (phiFactors.Factors is null)
		{
			result = FinishStrictlyWideCpu(prime, divisorData);
		}
		else
		{
			result = RunHeuristicPipelineWideCpu(prime, previousOrder, divisorData, config, phi, phiFactors);
		}

		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static ulong RunHeuristicPipelineCpu(
		ulong prime,
		ulong? previousOrder,
		in PrimeOrderCalculatorConfig config,
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

		if (config.StrictMode)
		{
			orderFactors?.Dispose();
			return CalculateByFactorizationCpu(prime, divisorData);
		}

		if (TryHeuristicFinishCpu(prime, candidateOrder, previousOrder, divisorData, config, orderFactors, out ulong order))
		{
			return order;
		}

		return candidateOrder;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static UInt128 RunHeuristicPipelineWideCpu(
		in UInt128 prime,
		in UInt128? previousOrder,
		in MontgomeryDivisorData? divisorData,
		in PrimeOrderCalculatorConfig config,
		in UInt128 phi,
		in PartialFactorResult128 phiFactors)
	{
		if (phiFactors.FullyFactored && TrySpecialMaxWideCpu(phi, prime, divisorData, phiFactors))
		{
			return phi;
		}

		UInt128 candidateOrder = InitializeStartingOrderWideCpu(prime, phi, divisorData);
		candidateOrder = ExponentLoweringWideCpu(candidateOrder, prime, divisorData, phiFactors);

		if (TryConfirmOrderWideCpu(prime, candidateOrder, divisorData, config))
		{
			return candidateOrder;
		}

		if (config.StrictMode)
		{
			return FinishStrictlyWideCpu(prime, divisorData);
		}

		if (TryHeuristicFinishWideCpu(prime, candidateOrder, previousOrder, divisorData, config, out UInt128 order))
		{
			return order;
		}

		return candidateOrder;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static UInt128 FinishStrictlyWideCpu(in UInt128 prime, in MontgomeryDivisorData? divisorData)
	{
		UInt128 phi = prime - UInt128.One;
		FixedCapacityStack<Dictionary<UInt128, int>> uInt128IntDictionaryPool = ThreadStaticPools.UInt128IntDictionaryPool;
		Dictionary<UInt128, int> counts = uInt128IntDictionaryPool.Rent(capacity: 8);
		FactorCompletelyWide(phi, counts);
		int entryCount = counts.Count;
		if (entryCount == 0)
		{
			uInt128IntDictionaryPool.Return(counts);
			return phi;
		}

		FixedCapacityStack<List<KeyValuePair<UInt128, int>>> keyValuePairUInt128IntegerPool = ThreadStaticPools.KeyValuePairUInt128IntegerPool;
		var entries = keyValuePairUInt128IntegerPool.Rent(entryCount);
		entries.AddRange(counts);
		uInt128IntDictionaryPool.Return(counts);

		entries.Sort(static (a, b) => a.Key.CompareTo(b.Key));

		UInt128 order = phi;
		for (int i = 0; i < entryCount; i++)
		{
			UInt128 primeFactor = entries[i].Key;
			int exponent = entries[i].Value;
			for (int iteration = 0; iteration < exponent; iteration++)
			{
				if (order.ReduceCycleRemainder(primeFactor) != UInt128.Zero)
				{
					UInt128 candidate = order / primeFactor;
					if (Pow2ModWideCpu(candidate, prime, divisorData) == UInt128.One)
					{
						order = candidate;
					}
					else
					{
						break;
					}
				}
			}
		}

		keyValuePairUInt128IntegerPool.Return(entries);
		return order;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static bool TrySpecialMaxWideCpu(in UInt128 phi, in UInt128 prime, in MontgomeryDivisorData? divisorData, in PartialFactorResult128 factors)
	{
		ReadOnlySpan<FactorEntry128> factorSpan = factors.Factors;
		int length = factors.Count;
		for (int i = 0; i < length; i++)
		{
			UInt128 factor = factorSpan[i].Value;
			UInt128 reduced = phi / factor;
			if (Pow2ModWideCpu(reduced, prime, divisorData) == UInt128.One)
			{
				return false;
			}
		}

		return true;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static UInt128 InitializeStartingOrderWideCpu(in UInt128 prime, in UInt128 phi, in MontgomeryDivisorData? divisorData)
	{
		UInt128 order = phi;
		UInt128 mod8 = prime & (UInt128)7UL;
		if (mod8 == UInt128.One || mod8 == (UInt128)7UL)
		{
			UInt128 half = phi >> 1;
			if (Pow2ModWideCpu(half, prime, divisorData) == UInt128.One)
			{
				order = half;
			}
		}

		return order;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static UInt128 ExponentLoweringWideCpu(UInt128 order, in UInt128 prime, in MontgomeryDivisorData? divisorData, in PartialFactorResult128 factors)
	{
		ArrayPool<FactorEntry128> pool = ThreadStaticPools.FactorEntry128Pool;
		ReadOnlySpan<FactorEntry128> factorSpan = factors.Factors;
		int length = factors.Count;
		// TODO: This should never trigger from production code - check
		// if (length == 0)
		// {
		//     return order;
		// }
		FactorEntry128[] tempArray = pool.Rent(length + 1);
		Span<FactorEntry128> buffer = tempArray.AsSpan(0, length);

		factorSpan.CopyTo(buffer);
		if (!factors.FullyFactored && factors.Cofactor > UInt128.One && IsPrimeWide(factors.Cofactor))
		{
			buffer[length] = new FactorEntry128(factors.Cofactor, 1);
			length++;
		}

		buffer[..length].Sort(static (a, b) => a.Value.CompareTo(b.Value));

		for (int i = 0; i < length; i++)
		{
			UInt128 primeFactor = buffer[i].Value;
			int exponent = buffer[i].Exponent;
			for (int iteration = 0; iteration < exponent; iteration++)
			{
				if (order.ReduceCycleRemainder(primeFactor) == UInt128.Zero)
				{
					UInt128 reduced = order / primeFactor;
					if (Pow2ModWideCpu(reduced, prime, divisorData) == UInt128.One)
					{
						order = reduced;
						continue;
					}
				}

				break;
			}
		}

		pool.Return(tempArray, clearArray: false);
		return order;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static bool TryConfirmOrderWideCpu(in UInt128 prime, in UInt128 order, in MontgomeryDivisorData? divisorData, in PrimeOrderCalculatorConfig config)
	{
		if (order == UInt128.Zero)
		{
			return false;
		}

		if (Pow2ModWideCpu(order, prime, divisorData) != UInt128.One)
		{
			return false;
		}

		PartialFactorResult128 factorization = PartialFactorWide(order, config);
		if (factorization.Factors is null)
		{
			return false;
		}

		if (!factorization.FullyFactored)
		{
			if (factorization.Cofactor <= UInt128.One)
			{
				return false;
			}

			if (!IsPrimeWide(factorization.Cofactor))
			{
				return false;
			}

			factorization = factorization.WithAdditionalPrime(factorization.Cofactor);
		}

		ReadOnlySpan<FactorEntry128> span = factorization.Factors;
		int length = factorization.Count;
		for (int i = 0; i < length; i++)
		{
			UInt128 primeFactor = span[i].Value;
			UInt128 reduced = order;
			for (int iteration = 0; iteration < span[i].Exponent; iteration++)
			{
				if (reduced.ReduceCycleRemainder(primeFactor) != UInt128.Zero)
				{
					break;
				}

				reduced /= primeFactor;
				if (Pow2ModWideCpu(reduced, prime, divisorData) == UInt128.One)
				{
					return false;
				}
			}
		}

		return true;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static bool TryHeuristicFinishWideCpu(
		in UInt128 prime,
		in UInt128 order,
		in UInt128? previousOrder,
		in MontgomeryDivisorData? divisorData,
		in PrimeOrderCalculatorConfig config,
		out UInt128 result)
	{
		result = UInt128.Zero;
		if (order <= UInt128.One)
		{
			return false;
		}

		PartialFactorResult128 orderFactors = PartialFactorWide(order, config);
		if (orderFactors.Factors is null)
		{
			return false;
		}

		if (!orderFactors.FullyFactored)
		{
			if (orderFactors.Cofactor <= UInt128.One)
			{
				return false;
			}

			if (!IsPrimeWide(orderFactors.Cofactor))
			{
				return false;
			}

			orderFactors = orderFactors.WithAdditionalPrime(orderFactors.Cofactor);
		}

		int capacity = config.MaxPowChecks <= 0 ? 64 : config.MaxPowChecksCapacity;
		FixedCapacityStack<List<UInt128>> uInt128ListPool = ThreadStaticPools.UInt128ListPool;
		List<UInt128> candidates = uInt128ListPool.Rent(capacity);
		FactorEntry128[] factorArray = orderFactors.Factors!;

		BuildCandidatesWide(order, factorArray, orderFactors.Count, candidates, capacity);

		int candidateCount = candidates.Count;
		if (candidateCount == 0)
		{
			uInt128ListPool.Return(candidates);
			return false;
		}

		SortCandidatesWide(prime, previousOrder, candidates);

		int powBudget = config.MaxPowChecks <= 0 ? candidateCount : config.MaxPowChecks;
		int powUsed = 0;

		for (int i = 0; i < candidateCount; i++)
		{
			if (powUsed >= powBudget && powBudget > 0)
			{
				break;
			}

			UInt128 candidate = candidates[i];
			powUsed++;

			if (Pow2ModWideCpu(candidate, prime, divisorData) != UInt128.One)
			{
				continue;
			}

			if (!TryConfirmCandidateWideCpu(prime, candidate, config, ref powUsed, powBudget, divisorData))
			{
				continue;
			}

			result = candidate;
			uInt128ListPool.Return(candidates);
			return true;
		}

		uInt128ListPool.Return(candidates);
		return false;
	}

	// private static ulong specialMaxHits;
	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static bool TrySpecialMaxCpu(ulong phi, ulong prime, PartialFactorResult factors, in MontgomeryDivisorData divisorData)
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

		return EvaluateSpecialMaxCandidatesCpu(stackBuffer, factorSpan, phi, prime, divisorData);
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static bool EvaluateSpecialMaxCandidatesCpu(Span<ulong> buffer, ReadOnlySpan<ulong> factors, ulong phi, ulong prime, in MontgomeryDivisorData divisorData)
	{
		int factorCount = factors.Length;
		for (int i = 0; i < factorCount; i++)
		{
			ulong factor = factors[i];

			// The partial factor pipeline never feeds zero or oversized factors while scanning candidate orders.
			// if (factor == 0UL || factor > phi)
			// {
			// 	continue;
			// }

			// if (reduced == 0UL)
			// {
			// 	continue;
			// }

			buffer[i] = phi / factor;
		}

		buffer.Sort();

		ExponentRemainderStepperCpu stepper = ThreadStaticPools.RentExponentStepperCpu(divisorData);

		if (stepper.InitializeCpuIsUnity(buffer[0]))
		{
			ThreadStaticPools.ReturnExponentStepperCpu(stepper);
			return false;
		}

		for (int i = 1; i < factorCount; i++)
		{
			if (stepper.ComputeNextIsUnity(buffer[i]))
			{
				ThreadStaticPools.ReturnExponentStepperCpu(stepper);
				return false;
			}
		}

		ThreadStaticPools.ReturnExponentStepperCpu(stepper);
		return true;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static ulong InitializeStartingOrderCpu(ulong prime, ulong phi, in MontgomeryDivisorData divisorData)
	{
		ulong order = phi;
		if ((prime & 7UL) == 1UL || (prime & 7UL) == 7UL)
		{
			ulong half = phi >> 1;
			if (Pow2EqualsOneCpu(half, divisorData))
			{
				order = half;
			}
		}

		return order;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static ulong ExponentLoweringCpu(ulong order, ulong prime, PartialFactorResult factors, in MontgomeryDivisorData divisorData)
	{
		int length = factors.Count;
		ReadOnlySpan<ulong> factorSpan = factors.Factors.AsSpan(0, length);
		ReadOnlySpan<int> exponentsSpan = factors.Exponents.AsSpan(0, length);

		int newLength = length + 1;
		ulong[] tempFactorsArray = FixedCapacityPools.ExclusiveUlongArray.Rent(newLength);
		int[] tempExponentsArray = FixedCapacityPools.ExclusiveIntArray.Rent(newLength);

		Span<ulong> ulongBuffer = tempFactorsArray;
		factorSpan.CopyTo(ulongBuffer);
		Span<int> intBuffer = tempExponentsArray;
		exponentsSpan.CopyTo(intBuffer);

		if (!factors.FullyFactored && factors.Cofactor > 1UL && factors.CofactorIsPrime)
		{
			ulongBuffer[length] = factors.Cofactor;
			intBuffer[length] = 1;
			length = newLength;
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

			heapCandidateArray ??= FixedCapacityPools.ExclusiveUlongArray.Rent(ExponentHardLimit);
			heapEvaluationArray ??= ThreadStaticPools.BoolPool.Rent(ExponentHardLimit);
			ProcessExponentLoweringPrime(heapCandidateArray.AsSpan(0, exponent), heapEvaluationArray.AsSpan(0, exponent), ref order, primeFactor, exponent, ref stepper);
		}

		ThreadStaticPools.ReturnExponentStepperCpu(stepper);

		if (heapCandidateArray is not null)
		{
			FixedCapacityPools.ExclusiveUlongArray.Return(heapCandidateArray);
			ThreadStaticPools.BoolPool.Return(heapEvaluationArray!);
		}

		FixedCapacityPools.ExclusiveUlongArray.Return(tempFactorsArray);
		return order;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
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

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static bool TryConfirmOrderCpu(
		ulong prime,
		ulong order,
		in MontgomeryDivisorData divisorData,
		in PrimeOrderCalculatorConfig config,
		out PartialFactorResult? reusableFactorization)
	{
		reusableFactorization = null;

		ArgumentOutOfRangeException.ThrowIfEqual(order, 0UL);
		if (!Pow2EqualsOneCpu(order, divisorData))
		{
			return false;
		}

		PartialFactorResult factorization = PartialFactorCpu(order, config);
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

			heapCandidateArray ??= FixedCapacityPools.ExclusiveUlongArray.Rent(ExponentHardLimit);
			if (ValidateOrderForFactor(heapCandidateArray.AsSpan(0, exponent), primeFactor, exponent, order, ref stepper))
			{
				violates = true;
				break;
			}
		}

		ThreadStaticPools.ReturnExponentStepperCpu(stepper);

		if (heapCandidateArray is not null)
		{
			FixedCapacityPools.ExclusiveUlongArray.Return(heapCandidateArray);
		}

		return !violates;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
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

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static bool TryHeuristicFinishCpu(
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
		PartialFactorResult orderFactors = cachedOrderFactors ?? PartialFactorCpu(order, config);
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

			ulong candidate = candidateSpan[0];
			powUsed++;

			bool equalsOne = powStepper.InitializeCpuIsUnity(candidate);
			if (equalsOne && TryConfirmCandidateCpu(prime, candidate, divisorData, config, ref powUsed, powBudget))
			{
				candidates.Clear();
				ThreadStaticPools.ReturnUlongList(candidates);
				ThreadStaticPools.ReturnExponentStepperCpu(powStepper);
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

				if (!TryConfirmCandidateCpu(prime, candidate, divisorData, config, ref powUsed, powBudget))
				{
					continue;
				}

				candidates.Clear();
				ThreadStaticPools.ReturnUlongList(candidates);
				ThreadStaticPools.ReturnExponentStepperCpu(powStepper);
				result = candidate;
				return true;

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
	private static void SortCandidates(ulong prime, ulong? previousOrder, List<ulong> candidates)
	{
		ulong previous = previousOrder ?? 0UL;
		bool hasPrevious = previousOrder.HasValue;
		ulong threshold1 = prime >> 3;
		ulong threshold2 = prime >> 2;
		ulong threshold3 = (prime * 3UL) >> 3;
		int previousGroup = hasPrevious ? GetGroup(previous, threshold1, threshold2, threshold3) : 1;

		candidates.Sort((x, y) => CompareCandidates(x, y, previous, previousGroup, hasPrevious, threshold1, threshold2, threshold3));
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static int CompareCandidates(ulong left, ulong right, ulong previous, int previousGroup, bool hasPrevious, ulong threshold1, ulong threshold2, ulong threshold3)
	{
		int leftGroup = GetGroup(left, threshold1, threshold2, threshold3);
		int rightGroup = GetGroup(right, threshold1, threshold2, threshold3);
		if (leftGroup == 0)
		{
			return rightGroup == 0 ? 0 : 1;
		}

		if (rightGroup == 0)
		{
			return -1;
		}

		bool leftIsGe = !hasPrevious || left >= previous;
		bool rightIsGe = !hasPrevious || right >= previous;
		int leftPrimary = ComputePrimary(leftGroup, leftIsGe, previousGroup);
		int rightPrimary = ComputePrimary(rightGroup, rightIsGe, previousGroup);

		int compare = leftPrimary.CompareTo(rightPrimary);
		if (compare != 0)
		{
			return compare;
		}

		long leftSecondary;
		long leftTertiary;
		if (leftGroup == 3)
		{
			leftSecondary = -(long)left;
			leftTertiary = -(long)left;
		}
		else
		{
			ulong leftReference = hasPrevious ? (left > previous ? left - previous : previous - left) : left;
			leftSecondary = (long)leftReference;
			leftTertiary = (long)left;
		}

		long rightSecondary;
		long rightTertiary;
		if (rightGroup == 3)
		{
			rightSecondary = -(long)right;
			rightTertiary = -(long)right;
		}
		else
		{
			ulong rightReference = hasPrevious ? (right > previous ? right - previous : previous - right) : right;
			rightSecondary = (long)rightReference;
			rightTertiary = (long)right;
		}

		compare = leftSecondary.CompareTo(rightSecondary);
		if (compare != 0)
		{
			return compare;
		}

		return leftTertiary.CompareTo(rightTertiary);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static void BuildCandidates(ulong order, ulong[] factors, int[] exponents, int count, List<ulong> candidates, int limit)
	{
		if (count == 0)
		{
			return;
		}

		Array.Sort(factors, exponents, 0, count);
		BuildCandidatesRecursive(order, factors, exponents, 0, 1UL, candidates, limit);
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
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

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static bool TryConfirmCandidateCpu(ulong prime, ulong candidate, in MontgomeryDivisorData divisorData, in PrimeOrderCalculatorConfig config, ref int powUsed, int powBudget)
	{
		PartialFactorResult factorization = PartialFactorCpu(candidate, config);

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

			heapCandidateArray ??= FixedCapacityPools.ExclusiveUlongArray.Rent(ExponentHardLimit);
			if (CheckCandidateViolation(heapCandidateArray.AsSpan(0, exponent), primeFactor, exponent, candidate, prime, ref powUsed, powBudget, ref stepper))
			{
				violates = true;
				break;
			}
		}

		ThreadStaticPools.ReturnExponentStepperCpu(stepper);

		if (heapCandidateArray is not null)
		{
			FixedCapacityPools.ExclusiveUlongArray.Return(heapCandidateArray);
		}

		factorization.Dispose();
		return !violates;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
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

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static bool Pow2EqualsOneCpu(ulong exponent, in MontgomeryDivisorData divisorData)
		=> exponent.Pow2MontgomeryModWindowedConvertToStandardCpu(divisorData) == 1UL;

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static PartialFactorResult PartialFactorCpu(ulong value, in PrimeOrderCalculatorConfig config)
	{
		if (value <= 1UL)
		{
			return PartialFactorResult.Empty;
		}

		// stackalloc is faster than pooling

		var result = PartialFactorResult.Rent();
		ulong[] factors = result.Factors;
		Span<ulong> primeSlots = new(factors);
		int[] exponents = result.Exponents;
		Span<int> exponentSlots = new(exponents);

		// We don't need to worry about leftovers, because we always use indexes within the calculated counts
		// primeSlots.Clear();
		// exponentSlots.Clear();

		List<PartialFactorPendingEntry> pending = result.PendingFactors;
		pending.Clear();

		PopulateSmallPrimeFactorsCpu(
			value,
			config.SmallFactorLimit,
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

				CollectFactorsCpu(primeSlots, exponentSlots, ref factorCount, pending, compositeStack, deadlineTimestamp, out bool limitReached);

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
				bool isPrime = PrimeTesterByLastDigit.IsPrimeCpu(composite);

				entry.WithPrimality(isPrime);
			}

			if (entry.IsPrime)
			{
				AddFactorToCollector(primeSlots, exponentSlots, ref factorCount, composite, 1);
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
			cofactorIsPrime = PrimeTesterByLastDigit.IsPrimeCpu(cofactor);
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

		// Check if it was it factored completely

		result.Cofactor = cofactor;
		result.FullyFactored = cofactor == 1UL;
		result.Count = factorCount;
		result.CofactorIsPrime = cofactorIsPrime;

		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static void CollectFactorsCpu(Span<ulong> primeSlots, Span<int> exponentSlots, ref int factorCount, List<PartialFactorPendingEntry> pending, FixedCapacityStack<ulong> compositeStack, long deadlineTimestamp, out bool pollardRhoDeadlineReached)
	{
		while (compositeStack.Count > 0)
		{
			ulong composite = compositeStack.Pop();
			bool isPrime = PrimeTesterByLastDigit.IsPrimeCpu(composite);

			if (isPrime)
			{
				AddFactorToCollector(primeSlots, exponentSlots, ref factorCount, composite, 1);
				continue;
			}

			pollardRhoDeadlineReached = Stopwatch.GetTimestamp() > deadlineTimestamp;
			if (pollardRhoDeadlineReached)
			{
				pending.Add(PartialFactorPendingEntry.Rent(composite, knownComposite: true));
				continue;
			}

			if (!TryPollardRhoCpu(composite, deadlineTimestamp, out ulong factor))
			{
				pending.Add(PartialFactorPendingEntry.Rent(composite, knownComposite: true));
				continue;
			}

			ulong quotient = composite / factor;
			compositeStack.Push(factor);
			compositeStack.Push(quotient);
		}

		pollardRhoDeadlineReached = false;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static void PopulateSmallPrimeFactorsCpu(
				ulong value,
				uint limit,
				Span<ulong> primeTargets,
				Span<int> exponentTargets,
				out int factorCount,
				out ulong remaining)
	{
		factorCount = 0;
		if (value <= 1UL)
		{
			remaining = value;
			return;
		}

		uint[] primes = PrimesGenerator.SmallPrimes;
		ulong[] squares = PrimesGenerator.SmallPrimesPow2;
		int primeCount = primes.Length;

		for (int i = 0; i < primeCount && value > 1UL; i++)
		{
			uint primeCandidate = primes[i];
			if (primeCandidate > limit)
			{
				break;
			}

			ulong primeValue = squares[i];
			// primeSquare will never == 0 in production code
			// if (primeSquare != 0 && primeSquare > remainingLocal)
			if (primeValue > value)
			{
				break;
			}

			primeValue = primeCandidate;
			// primeCandidate will never equal 0 in production code
			// if (primeCandidate == 0UL)
			// {
			// 	continue;
			// }

			if (value.ReduceCycleRemainder(primeValue) != 0UL)
			{
				continue;
			}

			int exponent = ExtractSmallPrimeExponent(ref value, primeValue);
			if (exponent == 0)
			{
				continue;
			}

			EnsureCollectorCapacity(
				primeTargets,
				factorCount + 1);

			primeTargets[factorCount] = primeValue;
			exponentTargets[factorCount] = exponent;
			factorCount++;
		}

		remaining = value;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static long CreateDeadlineTimestamp(int milliseconds)
	{
		if (milliseconds <= 0)
		{
			return long.MaxValue;
		}

		long startTimestamp = Stopwatch.GetTimestamp();
		long deadline = ConvertMillisecondsToStopwatchTicks(milliseconds);
		deadline += startTimestamp;
		if (deadline < startTimestamp)
		{
			return long.MaxValue;
		}

		return deadline;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static long ConvertMillisecondsToStopwatchTicks(int milliseconds)
	{
		long frequency = Stopwatch.Frequency;
		long baseTicks = (frequency / 1000L) * milliseconds;
		long remainder = (frequency % 1000L) * milliseconds;
		return baseTicks + (remainder / 1000L);
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static bool TryPollardRhoCpu(ulong n, long deadlineTimestamp, out ulong factor)
	{
		factor = 0UL;
		if ((n & 1UL) == 0UL)
		{
			factor = 2UL;
			return true;
		}

		long timestamp; // reused for deadline checks.
		while (true)
		{
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
				y = AdvancePolynomialTwiceCpu(y, c, n);
				ulong diff = x > y ? x - y : y - x;
				d = diff.BinaryGcd(n);
			}

			if (d != n)
			{
				factor = d;
				return true;
			}
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static ulong AdvancePolynomialCpu(ulong x, ulong c, ulong modulus)
	{
		UInt128 value = (UInt128)x * x + c;
		return (ulong)value.ReduceCycleRemainder(modulus);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static ulong AdvancePolynomialTwiceCpu(ulong x, ulong c, ulong modulus)
	{
		UInt128 value = (UInt128)x * x + c;
		x = (ulong)value.ReduceCycleRemainder(modulus);
		value = (UInt128)x * x + c;
		return (ulong)value.ReduceCycleRemainder(modulus);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static void AddFactorToCollector(
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

		for (int i = 0; i < count; i++)
		{
			if (primes[i] == prime)
			{
				exponents[i] += exponent;
				return;
			}
		}

		EnsureCollectorCapacity(
			primes,
			count + 1);

		primes[count] = prime;
		exponents[count] = exponent;
		count++;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static void EnsureCollectorCapacity(
				Span<ulong> primes,
				int requiredCapacity)
	{
		ArgumentOutOfRangeException.ThrowIfGreaterThan(requiredCapacity, primes.Length);
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static ulong CalculateByFactorizationCpu(ulong prime, in MontgomeryDivisorData divisorData)
	{
		ulong phi = prime - 1UL;
		Dictionary<ulong, int> counts = ThreadStaticPools.RentFactorCountDictionary();
		FactorCompletelyCpu(phi, counts, false);
		if (counts.Count == 0)
		{
			ThreadStaticPools.ReturnFactorCountDictionary(counts);
			return phi;
		}

		KeyValuePair<ulong, int>[] entries = [.. counts];
		Array.Sort(entries, static (a, b) => a.Key.CompareTo(b.Key));

		ulong order = phi;
		int entryCount = entries.Length;

		for (int i = 0; i < entryCount; i++)
		{
			ulong primeFactor = entries[i].Key;
			int exponent = entries[i].Value;
			for (int iteration = 0; iteration < exponent; iteration++)
			{
				ulong remainder = order.ReduceCycleRemainder(primeFactor);
				if (remainder != 0UL)
				{
					break;
				}

				// TODO: Benchmark performance with the division vs reusing remainder
				ulong candidate = order - primeFactor * remainder;
				// ulong candidate = order / primeFactor;

				if (candidate.Pow2MontgomeryModWindowedConvertToStandardCpu(divisorData) == 1UL)
				{
					order = candidate;
					continue;
				}

				break;
			}
		}

		ThreadStaticPools.ReturnFactorCountDictionary(counts);
		return order;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static void FactorCompletelyCpu(ulong value, Dictionary<ulong, int> counts, bool knownComposite)
	{
		if (value <= 1UL)
		{
			return;
		}

		bool isPrime = PrimeTesterByLastDigit.IsPrimeCpu(value);
		if (!knownComposite && isPrime)
		{
			AddFactor(counts, value, 1);
			return;
		}

		ulong factor = PollardRhoStrict(value);
		ulong quotient = value / factor;

		isPrime = PrimeTesterByLastDigit.IsPrimeCpu(factor);
		if (isPrime)
		{
			int exponent = 1;
			ulong remaining = quotient;
			while (remaining.ReduceCycleRemainder(factor) == 0UL)
			{
				remaining /= factor;
				exponent++;
			}

			AddFactor(counts, factor, exponent);

			if (remaining > 1UL)
			{
				FactorCompletelyCpu(remaining, counts, knownComposite: false);
			}

			return;
		}

		FactorCompletelyCpu(factor, counts, knownComposite: true);

		if (quotient == factor)
		{
			isPrime = false;
		}
		else
		{
			isPrime = PrimeTesterByLastDigit.IsPrimeCpu(quotient);
		}

		if (isPrime)
		{
			AddFactor(counts, quotient, 1);
		}
		else
		{
			FactorCompletelyCpu(quotient, counts, knownComposite: true);
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
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
				y = AdvancePolynomialTwiceCpu(y, c, n);
				ulong diff = x > y ? x - y : y - x;
				d = BinaryGcd(diff, n);
			}

			if (d != n)
			{
				return d;
			}
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static UInt128 Pow2ModWideCpu(in UInt128 exponent, in UInt128 modulus, in MontgomeryDivisorData? divisorData)
	{
		if (modulus == UInt128.One)
		{
			return UInt128.Zero;
		}

		return exponent.Pow2MontgomeryModWindowed(modulus);
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static bool TryConfirmCandidateWideCpu(in UInt128 prime, in UInt128 candidate, in PrimeOrderCalculatorConfig config, ref int powUsed, int powBudget, in MontgomeryDivisorData? divisorData)
	{
		PartialFactorResult128 factorization = PartialFactorWide(candidate, config);
		if (factorization.Factors is null)
		{
			return false;
		}

		if (!factorization.FullyFactored)
		{
			if (factorization.Cofactor <= UInt128.One)
			{
				return false;
			}

			if (!IsPrimeWide(factorization.Cofactor))
			{
				return false;
			}

			factorization = factorization.WithAdditionalPrime(factorization.Cofactor);
		}

		ReadOnlySpan<FactorEntry128> span = factorization.Factors;
		int length = factorization.Count;
		for (int i = 0; i < length; i++)
		{
			UInt128 primeFactor = span[i].Value;
			UInt128 reduced = candidate;
			for (int iteration = 0; iteration < span[i].Exponent; iteration++)
			{
				if (reduced.ReduceCycleRemainder(primeFactor) != UInt128.Zero)
				{
					break;
				}

				reduced /= primeFactor;
				if (powUsed >= powBudget && powBudget > 0)
				{
					return false;
				}

				powUsed++;
				if (Pow2ModWideCpu(reduced, prime, divisorData) == UInt128.One)
				{
					return false;
				}
			}
		}

		return true;
	}

}



