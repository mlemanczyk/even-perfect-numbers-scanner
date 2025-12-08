using System.Buffers;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ILGPU;
using PerfectNumbers.Core.Cpu;
using PerfectNumbers.Core.Gpu;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core;

internal static partial class PrimeOrderCalculator
{
	internal struct PendingEntry
	{
		public ulong Value;
		public bool KnownComposite;
		public bool HasKnownPrimality;
		public bool IsPrime;

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
		public void WithPrimality(bool isPrime)
		{
			KnownComposite = KnownComposite || !isPrime;
			HasKnownPrimality = true;
			IsPrime = isPrime;
		}
	}

	public static ulong CalculateCpu(
		ulong prime,
		ulong? previousOrder,
		in MontgomeryDivisorData divisorData,
		in PrimeOrderSearchConfig config)
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
			in PrimeOrderSearchConfig config)
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
			Queue<MontgomeryDivisorData> divisorPool = MontgomeryDivisorDataPool.Shared;
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

    private static UInt128 CalculateWideInternalCpu(in UInt128 prime, in UInt128? previousOrder, in MontgomeryDivisorData? divisorData, in PrimeOrderSearchConfig config)
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

	private static ulong RunHeuristicPipelineHybrid(
		PrimeOrderCalculatorAccelerator gpu,
		ulong prime,
		ulong? previousOrder,
		in PrimeOrderSearchConfig config,
		in MontgomeryDivisorData divisorData,
		ulong phi,
		PartialFactorResult phiFactors)
	{
		if (phiFactors.FullyFactored && TrySpecialMaxHybrid(gpu, phi, prime, phiFactors, divisorData))
		{
			return phi;
		}

		ulong candidateOrder = InitializeStartingOrderHybrid(gpu, prime, phi, divisorData);
		candidateOrder = ExponentLoweringCpu(candidateOrder, prime, phiFactors, divisorData);

		if (TryConfirmOrderHybrid(gpu, prime, candidateOrder, divisorData, config, out PartialFactorResult? orderFactors))
		{
			return candidateOrder;
		}

		if (config.Mode == PrimeOrderMode.Strict)
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

    private static UInt128 RunHeuristicPipelineWideCpu(
        in UInt128 prime,
        in UInt128? previousOrder,
        in MontgomeryDivisorData? divisorData,
        in PrimeOrderSearchConfig config,
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

        if (config.Mode == PrimeOrderMode.Strict)
        {
            return FinishStrictlyWideCpu(prime, divisorData);
        }

        if (TryHeuristicFinishWideCpu(prime, candidateOrder, previousOrder, divisorData, config, out UInt128 order))
        {
            return order;
        }

        return candidateOrder;
    }

    private static UInt128 FinishStrictlyWideCpu(in UInt128 prime, in MontgomeryDivisorData? divisorData)
    {
        UInt128 phi = prime - UInt128.One;
        Dictionary<UInt128, int> counts = new(capacity: 8);
        FactorCompletelyWide(phi, counts);
        if (counts.Count == 0)
        {
            return phi;
        }

        List<KeyValuePair<UInt128, int>> entries = new(counts);
        entries.Sort(static (a, b) => a.Key.CompareTo(b.Key));

        UInt128 order = phi;
        int entryCount = entries.Count;
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

        return order;
    }

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

    private static bool TryConfirmOrderWideCpu(in UInt128 prime, in UInt128 order, in MontgomeryDivisorData? divisorData, in PrimeOrderSearchConfig config)
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

    private static bool TryHeuristicFinishWideCpu(
        in UInt128 prime,
        in UInt128 order,
        in UInt128? previousOrder,
        in MontgomeryDivisorData? divisorData,
        in PrimeOrderSearchConfig config,
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

        int capacity = config.MaxPowChecks <= 0 ? 64 : config.MaxPowChecks * 4;
        List<UInt128> candidates = new(capacity);
        FactorEntry128[] factorArray = orderFactors.Factors!;
        BuildCandidatesWide(order, factorArray, orderFactors.Count, candidates, capacity);
        if (candidates.Count == 0)
        {
            return false;
        }

        SortCandidatesWide(prime, previousOrder, candidates);

        int powBudget = config.MaxPowChecks <= 0 ? candidates.Count : config.MaxPowChecks;
        int powUsed = 0;
        int candidateCount = candidates.Count;

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
            return true;
        }

        return false;
    }

	// private static ulong specialMaxHits;
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

	private static bool TryConfirmOrderHybrid(
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

		if (!Pow2EqualsOneHybrid(gpu, order, prime, divisorData))
		{
			return false;
		}

		PartialFactorResult factorization = PartialFactorHybrid(gpu, order, config);
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

			int capacity = config.MaxPowChecks <= 0 ? 64 : config.MaxPowChecks << 2;
			List<ulong> candidates = ThreadStaticPools.RentUlongList(capacity);
			candidates.Clear();
			ulong[] factorArray = orderFactors.Factors!;
			int[] exponentsArray = orderFactors.Exponents!;

			BuildCandidates(order, factorArray, exponentsArray, orderFactors.Count, candidates, capacity);
			if (candidates.Count == 0)
			{
				ThreadStaticPools.ReturnUlongList(candidates);
				return false;
			}

			SortCandidates(prime, previousOrder, candidates);

			int powBudget = config.MaxPowChecks <= 0 ? candidates.Count : config.MaxPowChecks;
			int powUsed = 0;
			int candidateCount = candidates.Count;
			Span<ulong> candidateSpan = CollectionsMarshal.AsSpan(candidates);
			ExponentRemainderStepperCpu powStepper = ThreadStaticPools.RentExponentStepperCpu(divisorData);
			bool powStepperInitialized = false;

			int index = 0;
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
					if (equalsOne && TryConfirmCandidateCpu(prime, candidate, divisorData, config, ref powUsed, powBudget))
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

	private static bool TryHeuristicFinishHybrid(
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
		PartialFactorResult orderFactors = cachedOrderFactors ?? PartialFactorHybrid(gpu, order, config);
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

			int capacity = config.MaxPowChecks <= 0 ? 64 : config.MaxPowChecks << 2;
			List<ulong> candidates = ThreadStaticPools.RentUlongList(capacity);
			candidates.Clear();
			ulong[] factorArray = orderFactors.Factors!;
			int[] exponentsArray = orderFactors.Exponents!;

			BuildCandidates(order, factorArray, exponentsArray, orderFactors.Count, candidates, capacity);
			if (candidates.Count == 0)
			{
				ThreadStaticPools.ReturnUlongList(candidates);
				return false;
			}

			SortCandidates(prime, previousOrder, candidates);

			int powBudget = config.MaxPowChecks <= 0 ? candidates.Count : config.MaxPowChecks;
			int powUsed = 0;
			int candidateCount = candidates.Count;
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

	private static bool TryHeuristicFinishGpu(
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
		PartialFactorResult orderFactors = cachedOrderFactors ?? PartialFactorGpu(gpu, order, config);
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

			int capacity = config.MaxPowChecks <= 0 ? 64 : config.MaxPowChecks << 2;
			List<ulong> candidates = ThreadStaticPools.RentUlongList(capacity);
			candidates.Clear();
			ulong[] factorArray = orderFactors.Factors!;
			int[] exponentsArray = orderFactors.Exponents!;

			BuildCandidates(order, factorArray, exponentsArray, orderFactors.Count, candidates, capacity);
			if (candidates.Count == 0)
			{
				ThreadStaticPools.ReturnUlongList(candidates);
				return false;
			}

			SortCandidates(prime, previousOrder, candidates);

			int powBudget = config.MaxPowChecks <= 0 ? candidates.Count : config.MaxPowChecks;
			int powUsed = 0;
			int candidateCount = candidates.Count;
			bool allowGpuBatch = true;
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
				ulong[]? gpuPool = null;
				Span<ulong> pooledGpuRemainders = default;
				bool gpuSuccess = false;
				bool gpuStackRemainders = false;
				GpuPow2ModStatus status = GpuPow2ModStatus.Unavailable;

				if (allowGpuBatch)
				{
					if (batchSize <= PerfectNumberConstants.DefaultSmallPrimeFactorSlotCount)
					{
						Span<ulong> localRemainders = stackGpuRemainders[..batchSize];
						status = PrimeOrderGpuHeuristics.TryPow2ModBatch(gpu, batch, prime, localRemainders, divisorData);
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
						status = PrimeOrderGpuHeuristics.TryPow2ModBatch(gpu, batch, prime, pooledRemainders, divisorData);
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

	private static bool TryConfirmCandidateCpu(ulong prime, ulong candidate, in MontgomeryDivisorData divisorData, in PrimeOrderSearchConfig config, ref int powUsed, int powBudget)
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

	private static bool TryConfirmCandidateHybrid(PrimeOrderCalculatorAccelerator gpu, ulong prime, ulong candidate, in MontgomeryDivisorData divisorData, in PrimeOrderSearchConfig config, ref int powUsed, int powBudget)
	{
		PartialFactorResult factorization = PartialFactorHybrid(gpu, candidate, config);

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

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool Pow2EqualsOneCpu(ulong exponent, ulong prime, in MontgomeryDivisorData divisorData)
		=> exponent.Pow2MontgomeryModWindowedConvertToStandardCpu(divisorData) == 1UL;

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

	// private static ulong _partialFactorHits;
	// private static ulong _partialFactorPendingHits;
	// private static ulong _partialFactorCofactorHits;

	private static bool _goToGpu = true;
	private static int _cpuCount;

	private static PartialFactorResult PartialFactorCpu(ulong value, in PrimeOrderSearchConfig config)
	{
		if (value <= 1UL)
		{
			return PartialFactorResult.Empty;
		}

		// stackalloc is faster than pooling
		Span<ulong> primeSlots = stackalloc ulong[PrimeOrderConstants.GpuSmallPrimeFactorSlots];
		Span<int> exponentSlots = stackalloc int[PrimeOrderConstants.GpuSmallPrimeFactorSlots];
		ulong[] primeSlotArray = [];
		int[] exponentSlotArray = [];

		// We don't need to worry about leftovers, because we always use indexes within the calculated counts
		// primeSlots.Clear();
		// exponentSlots.Clear();

		uint limit = config.SmallFactorLimit == 0 ? uint.MaxValue : config.SmallFactorLimit;
		ulong remaining = value;

		List<PendingEntry> pending = ThreadStaticPools.RentPrimeOrderPendingEntryList(2);

		PopulateSmallPrimeFactorsCpu(
			value,
			limit,
			ref primeSlots,
			ref exponentSlots,
			ref primeSlotArray,
			ref exponentSlotArray,
			out int factorCount,
			out remaining);

		bool limitReached;
		if (remaining > 1UL)
		{
			if (config.PollardRhoMilliseconds > 0)
			{
				long deadlineTimestamp = CreateDeadlineTimestamp(config.PollardRhoMilliseconds);
				Stack<ulong> compositeStack = ThreadStaticPools.RentUlongStack(4);
				compositeStack.Push(remaining);

				CollectFactorsCpu(ref primeSlots, ref exponentSlots, ref primeSlotArray, ref exponentSlotArray, ref factorCount, pending, compositeStack, deadlineTimestamp, out limitReached);

				if (limitReached)
				{
					while (compositeStack.Count > 0)
					{
						pending.Add(new PendingEntry(compositeStack.Pop(), knownComposite: false));
					}
				}

				compositeStack.Clear();
				ThreadStaticPools.ReturnUlongStack(compositeStack);
			}
			else
			{
				pending.Add(new PendingEntry(remaining, knownComposite: false));
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
				bool isPrime = PrimeTesterByLastDigit.IsPrimeCpu(composite);

				// Atomic.Add(ref _partialFactorPendingHits, 1UL);
				// Console.WriteLine($"Partial factor pending hits {Volatile.Read(ref _partialFactorPendingHits)}");

				entry.WithPrimality(isPrime);
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
			cofactorIsPrime = PrimeTesterByLastDigit.IsPrimeCpu(cofactor);

			// Atomic.Add(ref _partialFactorCofactorHits, 1UL);
			// Console.WriteLine($"Partial factor cofactor hits {Volatile.Read(ref _partialFactorCofactorHits)}");
		}

		PartialFactorResult result;
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

		ulong[] factorsArray = ulongPool.Rent(factorCount);
		int[] exponentsArray = intPool.Rent(factorCount);

		for (int i = 0; i < factorCount; i++)
		{
			factorsArray[i] = primeSlots[i];
			exponentsArray[i] = exponentSlots[i];

			// This will never happen on the execution path from production code
			// if (primeValue == 0UL || exponentValue == 0)
			// {
			// 	throw new Exception("Prime value or exponent equals zero");
			// 	continue;
			// }
		}

		Array.Sort(factorsArray, exponentsArray, 0, factorCount);

		// Check if it was it factored completely
		limitReached = cofactor == 1UL;

		result = PartialFactorResult.Rent(factorsArray, exponentsArray, cofactor, limitReached, factorCount, cofactorIsPrime);

	ReturnResult:
		if (primeSlotArray.Length > 0)
		{
			ThreadStaticPools.UlongPool.Return(primeSlotArray, clearArray: false);
			ThreadStaticPools.IntPool.Return(exponentSlotArray, clearArray: false);
		}

		pending.Clear();
		ThreadStaticPools.ReturnPrimeOrderPendingEntryList(pending);
		return result;
	}

	private static PartialFactorResult PartialFactorHybrid(PrimeOrderCalculatorAccelerator gpu, ulong value, in PrimeOrderSearchConfig config)
	{
		if (value <= 1UL)
		{
			return PartialFactorResult.Empty;
		}

		// stackalloc is faster than pooling
		Span<ulong> primeSlots = stackalloc ulong[PrimeOrderConstants.GpuSmallPrimeFactorSlots];
		Span<int> exponentSlots = stackalloc int[PrimeOrderConstants.GpuSmallPrimeFactorSlots];
		ulong[] primeSlotArray = [];
		int[] exponentSlotArray = [];
		// We don't need to worry about leftovers, because we always use indexes within the calculated counts
		// primeSlots.Clear();
		// exponentSlots.Clear();

		uint limit = config.SmallFactorLimit == 0 ? uint.MaxValue : config.SmallFactorLimit;
		ulong remaining = value;

		List<PendingEntry> pending = ThreadStaticPools.RentPrimeOrderPendingEntryList(2);

		PopulateSmallPrimeFactorsCpu(
			value,
			limit,
			ref primeSlots,
			ref exponentSlots,
			ref primeSlotArray,
			ref exponentSlotArray,
			out int factorCount,
			out remaining);

		bool limitReached;
		if (remaining > 1UL)
		{
			if (config.PollardRhoMilliseconds > 0)
			{
				long deadlineTimestamp = CreateDeadlineTimestamp(config.PollardRhoMilliseconds);
				Stack<ulong> compositeStack = ThreadStaticPools.RentUlongStack(4);
				compositeStack.Push(remaining);

				CollectFactorsHybrid(gpu, ref primeSlots, ref exponentSlots, primeSlotArray: ref primeSlotArray, ref exponentSlotArray, ref factorCount, pending, compositeStack, deadlineTimestamp, out limitReached);

				if (limitReached)
				{
					while (compositeStack.Count > 0)
					{
						pending.Add(new PendingEntry(compositeStack.Pop(), knownComposite: false));
					}
				}

				compositeStack.Clear();
				ThreadStaticPools.ReturnUlongStack(compositeStack);
			}
			else
			{
				pending.Add(new PendingEntry(remaining, knownComposite: false));
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
				bool isPrime = RunOnCpu() ? PrimeTesterByLastDigit.IsPrimeCpu(composite) : HeuristicCombinedPrimeTester.IsPrimeGpu(gpu, composite);

				// Atomic.Add(ref _partialFactorPendingHits, 1UL);
				// Console.WriteLine($"Partial factor pending hits {Volatile.Read(ref _partialFactorPendingHits)}");

				entry.WithPrimality(isPrime);
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
			cofactorIsPrime = RunOnCpu() ? PrimeTesterByLastDigit.IsPrimeCpu(cofactor) : HeuristicCombinedPrimeTester.IsPrimeGpu(gpu, cofactor);
		}

		PartialFactorResult result;
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

		ulong[] factorsArray = ulongPool.Rent(factorCount);
		int[] exponentsArray = intPool.Rent(factorCount);

		for (int i = 0; i < factorCount; i++)
		{
			factorsArray[i] = primeSlots[i];
			exponentsArray[i] = exponentSlots[i];

			// This will never happen on the execution path from production code
			// if (primeValue == 0UL || exponentValue == 0)
			// {
			// 	throw new Exception("Prime value or exponent equals zero");
			// 	continue;
			// }
		}

		Array.Sort(factorsArray, exponentsArray, 0, factorCount);

		// Check if it was factored completely
		limitReached = cofactor == 1UL;

		result = PartialFactorResult.Rent(factorsArray, exponentsArray, cofactor, limitReached, factorCount, cofactorIsPrime);

	ReturnResult:
		if (primeSlotArray.Length > 0)
		{
			ThreadStaticPools.UlongPool.Return(primeSlotArray, clearArray: false);
			ThreadStaticPools.IntPool.Return(exponentSlotArray!, clearArray: false);
		}

		pending.Clear();
		ThreadStaticPools.ReturnPrimeOrderPendingEntryList(pending);
		return result;
	}

	private static PartialFactorResult PartialFactorGpu(PrimeOrderCalculatorAccelerator gpu, ulong value, in PrimeOrderSearchConfig config)
	{
		if (value <= 1UL)
		{
			return PartialFactorResult.Empty;
		}

		// stackalloc is faster than pooling
		Span<ulong> primeSlots = stackalloc ulong[PrimeOrderConstants.GpuSmallPrimeFactorSlots];
		Span<int> exponentSlots = stackalloc int[PrimeOrderConstants.GpuSmallPrimeFactorSlots];
		ulong[] primeSlotArray = [];
		int[] exponentSlotArray = [];
		// We don't need to worry about leftovers, because we always use indexes within the calculated counts
		// primeSlots.Clear();
		// exponentSlots.Clear();

		uint limit = config.SmallFactorLimit == 0 ? uint.MaxValue : config.SmallFactorLimit;
		ulong remaining = value;

		bool gpuFactored = PrimeOrderGpuHeuristics.TryPartialFactor(gpu, value, limit, primeSlots, exponentSlots, out int factorCount, out remaining);

		List<PendingEntry> pending = ThreadStaticPools.RentPrimeOrderPendingEntryList(2);

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

		bool limitReached;
		if (remaining > 1UL)
		{
			if (config.PollardRhoMilliseconds > 0)
			{
				long deadlineTimestamp = CreateDeadlineTimestamp(config.PollardRhoMilliseconds);
				Stack<ulong> compositeStack = ThreadStaticPools.RentUlongStack(4);
				compositeStack.Push(remaining);

				CollectFactorsCpu(ref primeSlots, ref exponentSlots, ref primeSlotArray, ref exponentSlotArray, ref factorCount, pending, compositeStack, deadlineTimestamp, out limitReached);

				if (limitReached)
				{
					while (compositeStack.Count > 0)
					{
						pending.Add(new PendingEntry(compositeStack.Pop(), knownComposite: false));
					}
				}

				compositeStack.Clear();
				ThreadStaticPools.ReturnUlongStack(compositeStack);
			}
			else
			{
				pending.Add(new PendingEntry(remaining, knownComposite: false));
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
				bool isPrime = PrimeTesterByLastDigit.IsPrimeCpu(composite);

				// Atomic.Add(ref _partialFactorPendingHits, 1UL);
				// Console.WriteLine($"Partial factor pending hits {Volatile.Read(ref _partialFactorPendingHits)}");

				entry.WithPrimality(isPrime);
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
			cofactorIsPrime = PrimeTesterByLastDigit.IsPrimeCpu(cofactor);

			// Atomic.Add(ref _partialFactorCofactorHits, 1UL);
			// Console.WriteLine($"Partial factor cofactor hits {Volatile.Read(ref _partialFactorCofactorHits)}");
		}

		PartialFactorResult result;
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

		ulong[] factorsArray = ulongPool.Rent(factorCount);
		int[] exponentsArray = intPool.Rent(factorCount);

		for (int i = 0; i < factorCount; i++)
		{
			factorsArray[i] = primeSlots[i];
			exponentsArray[i] = exponentSlots[i];

			// This will never happen on the execution path from production code
			// if (primeValue == 0UL || exponentValue == 0)
			// {
			// 	throw new Exception("Prime value or exponent equals zero");
			// 	continue;
			// }
		}

		Array.Sort(factorsArray, exponentsArray, 0, factorCount);

		// Check if it was factored completely
		limitReached = cofactor == 1UL;

		result = PartialFactorResult.Rent(factorsArray, exponentsArray, cofactor, limitReached, factorCount, cofactorIsPrime);

	ReturnResult:
		if (primeSlotArray.Length > 0)
		{
			ThreadStaticPools.UlongPool.Return(primeSlotArray, clearArray: false);
			ThreadStaticPools.IntPool.Return(exponentSlotArray!, clearArray: false);
		}

		pending.Clear();
		ThreadStaticPools.ReturnPrimeOrderPendingEntryList(pending);
		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void CollectFactorsHybrid(PrimeOrderCalculatorAccelerator gpu, ref Span<ulong> primeSlots, ref Span<int> exponentSlots, ref ulong[] primeSlotArray, ref int[] exponentSlotArray, ref int factorCount, List<PendingEntry> pending, Stack<ulong> compositeStack, long deadlineTimestamp, out bool pollardRhoDeadlineReached)
	{
		while (compositeStack.Count > 0)
		{
			ulong composite = compositeStack.Pop();
			bool isPrime = RunOnCpu() ? PrimeTesterByLastDigit.IsPrimeCpu(composite) : HeuristicCombinedPrimeTester.IsPrimeGpu(gpu, composite);

			if (isPrime)
			{
				AddFactorToCollector(ref primeSlots, ref exponentSlots, ref primeSlotArray, ref exponentSlotArray, ref factorCount, composite, 1);
				continue;
			}

			pollardRhoDeadlineReached = Stopwatch.GetTimestamp() > deadlineTimestamp;
			if (pollardRhoDeadlineReached)
			{
				pending.Add(new PendingEntry(composite, knownComposite: true));
				continue;
			}

			if (!TryPollardRhoCpu(composite, deadlineTimestamp, out ulong factor))
			{
				pending.Add(new PendingEntry(composite, knownComposite: true));
				continue;
			}

			ulong quotient = composite / factor;
			compositeStack.Push(factor);
			compositeStack.Push(quotient);
		}

		pollardRhoDeadlineReached = false;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void CollectFactorsCpu(ref Span<ulong> primeSlots, ref Span<int> exponentSlots, ref ulong[] primeSlotArray, ref int[] exponentSlotArray, ref int factorCount, List<PendingEntry> pending, Stack<ulong> compositeStack, long deadlineTimestamp, out bool pollardRhoDeadlineReached)
	{
		while (compositeStack.Count > 0)
		{
			ulong composite = compositeStack.Pop();
			bool isPrime = PrimeTesterByLastDigit.IsPrimeCpu(composite);

			if (isPrime)
			{
				AddFactorToCollector(ref primeSlots, ref exponentSlots, ref primeSlotArray, ref exponentSlotArray, ref factorCount, composite, 1);
				continue;
			}

			pollardRhoDeadlineReached = Stopwatch.GetTimestamp() > deadlineTimestamp;
			if (pollardRhoDeadlineReached)
			{
				pending.Add(new PendingEntry(composite, knownComposite: true));
				continue;
			}

			if (!TryPollardRhoCpu(composite, deadlineTimestamp, out ulong factor))
			{
				pending.Add(new PendingEntry(composite, knownComposite: true));
				continue;
			}

			ulong quotient = composite / factor;
			compositeStack.Push(factor);
			compositeStack.Push(quotient);
		}

		pollardRhoDeadlineReached = false;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void CollectFactorsGpu(PrimeOrderCalculatorAccelerator gpu, ref Span<ulong> primeSlots, ref Span<int> exponentSlots, ref ulong[] primeSlotArray, ref int[] exponentSlotArray, ref int factorCount, List<PendingEntry> pending, Stack<ulong> compositeStack, long deadlineTimestamp, ref bool pollardRhoDeadlineReached)
	{
		while (compositeStack.Count > 0)
		{
			ulong composite = compositeStack.Pop();
			bool isPrime = HeuristicCombinedPrimeTester.IsPrimeGpu(gpu, composite);

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

			// if (!TryPollardRhoGpu(gpu, composite, out ulong factor))
			if (!TryPollardRhoCpu(composite, deadlineTimestamp, out ulong factor))
			{
				pending.Add(new PendingEntry(composite, knownComposite: true));
				continue;
			}

			ulong quotient = composite / factor;
			compositeStack.Push(factor);
			compositeStack.Push(quotient);
		}
	}

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
				if (Atomic.CompareExchange(ref _cpuCount, 0, 1) == 0)
				{
					return true;
				}

				if (Atomic.CompareExchange(ref _cpuCount, 1, 0) == 1)
				{
					return false;
				}

				Thread.Yield();
			}
		}

		int cpuCount = Atomic.Add(ref _cpuCount, 1);
		if (cpuCount == PerfectNumberConstants.GpuRatio)
		{
			Atomic.Add(ref _cpuCount, -PerfectNumberConstants.GpuRatio);
		}
		else if (cpuCount > PerfectNumberConstants.GpuRatio)
		{
			cpuCount -= PerfectNumberConstants.GpuRatio;
		}

		return cpuCount != PerfectNumberConstants.GpuRatio;
	}

	private static void PopulateSmallPrimeFactorsCpu(
				ulong value,
				uint limit,
				ref Span<ulong> primeTargets,
				ref Span<int> exponentTargets,
				ref ulong[] primeArray,
				ref int[] exponentArray,
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

			if (remainingLocal.ReduceCycleRemainder(primeValue) != 0UL)
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

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
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

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ulong AdvancePolynomialCpu(ulong x, ulong c, ulong modulus)
	{
		UInt128 value = (UInt128)x * x + c;
		return (ulong)value.ReduceCycleRemainder(modulus);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ulong AdvancePolynomialTwiceCpu(ulong x, ulong c, ulong modulus)
	{
		UInt128 value = (UInt128)x * x + c;
		var remainder = (ulong)value.ReduceCycleRemainder(modulus);
		value = (UInt128)remainder * remainder + c;
		return (ulong)value.ReduceCycleRemainder(modulus);
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
				ref ulong[] primeArray,
				ref int[] exponentArray,
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
				ref ulong[] primeArray,
				ref int[] exponentArray,
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

		primes[..count].CopyTo(newPrimeSpan);
		exponents[..count].CopyTo(newExponentSpan);

		if (primeArray.Length > 0)
		{
			ThreadStaticPools.UlongPool.Return(primeArray, clearArray: false);
			ThreadStaticPools.IntPool.Return(exponentArray!, clearArray: false);
		}

		primeArray = newPrimeArray;
		exponentArray = newExponentArray;
		primes = newPrimeSpan;
		exponents = newExponentSpan;
	}

	private static ulong CalculateByFactorizationCpu(ulong prime, in MontgomeryDivisorData divisorData)
	{
		ulong phi = prime - 1UL;
		Dictionary<ulong, int> counts = new(capacity: 8);
		FactorCompletelyCpu(phi, counts, false);
		if (counts.Count == 0)
		{
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

		return order;
	}

	// private static ulong _factorCompletelyHits;
	// private static ulong _factorCompletelyPollardRhoHits;

	private static void FactorCompletelyCpu(ulong value, Dictionary<ulong, int> counts, bool knownComposite)
	{
		if (value <= 1UL)
		{
			return;
		}

		// Atomic.Add(ref _factorCompletelyHits, 1UL);
		// Console.WriteLine($"Factor completely hits {Volatile.Read(ref _factorCompletelyHits)}");

		bool isPrime = PrimeTesterByLastDigit.IsPrimeCpu(value);
		if (!knownComposite && isPrime)
		{
			AddFactor(counts, value, 1);
			return;
		}

		ulong factor = PollardRhoStrict(value);
		ulong quotient = value / factor;

		// Atomic.Add(ref _factorCompletelyPollardRhoHits, 1UL);
		// Console.WriteLine($"Factor completely after PollardRho hits {Volatile.Read(ref _factorCompletelyPollardRhoHits)}");

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
			// Atomic.Add(ref _partialFactorCofactorHits, 1UL);
			// Console.WriteLine($"Partial factor cofactor hits {Volatile.Read(ref _partialFactorCofactorHits)}");

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

    private static UInt128 Pow2ModWideCpu(in UInt128 exponent, in UInt128 modulus, in MontgomeryDivisorData? divisorData)
    {
        if (modulus == UInt128.One)
        {
            return UInt128.Zero;
        }

        return exponent.Pow2MontgomeryModWindowed(modulus);
    }

    private static bool TryConfirmCandidateWideCpu(in UInt128 prime, in UInt128 candidate, in PrimeOrderSearchConfig config, ref int powUsed, int powBudget, in MontgomeryDivisorData? divisorData)
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

	private readonly struct CandidateKey(int primary, long secondary, long tertiary)
	{
		public readonly int Primary = primary;
		public readonly long Secondary = secondary;
		public readonly long Tertiary = tertiary;
	}
}

