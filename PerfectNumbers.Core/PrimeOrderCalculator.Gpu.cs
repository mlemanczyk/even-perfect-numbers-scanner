using System.Buffers;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core;

internal static partial class PrimeOrderCalculator
{
	private const int GpuSmallPrimeFactorSlots = 64;

	public static ulong CalculateGpu(
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

		if (PrimeOrderGpuHeuristics.TryCalculateOrder(gpu, prime, previousOrder, config, divisorData, out ulong gpuOrder))
		{
			return gpuOrder;
		}

		PartialFactorResult phiFactors = PartialFactorGpu(gpu, phi, config);

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

	public static UInt128 CalculateGpu(
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
			Queue<MontgomeryDivisorData> divisorPool = MontgomeryDivisorDataPool.Shared;
			divisorData = divisorPool.FromModulus(prime64);
			ulong order64 = CalculateGpu(gpu, prime64, previous, divisorData, config);
			divisorPool.Return(divisorData);
			result = order64 == 0UL ? UInt128.Zero : (UInt128)order64;
		}
		else
		{
			divisorData = MontgomeryDivisorData.Empty;
			result = CalculateWideInternalGpu(gpu, prime, previousOrder, divisorData, config);
		}

		return result;
	}
	
    private static UInt128 CalculateWideInternalGpu(PrimeOrderCalculatorAccelerator gpu, in UInt128 prime, in UInt128? previousOrder, in MontgomeryDivisorData? divisorData, in PrimeOrderCalculatorConfig config)
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
            result = FinishStrictlyWideGpu(gpu, prime, divisorData);
        }
        else
        {
            result = RunHeuristicPipelineWideGpu(gpu, prime, previousOrder, divisorData, config, phi, phiFactors);
        }

        return result;
    }

    private static UInt128 RunHeuristicPipelineWideGpu(
		PrimeOrderCalculatorAccelerator gpu,
        in UInt128 prime,
        in UInt128? previousOrder,
        in MontgomeryDivisorData? divisorData,
        in PrimeOrderCalculatorConfig config,
        in UInt128 phi,
        in PartialFactorResult128 phiFactors)
    {
        if (phiFactors.FullyFactored && TrySpecialMaxWideGpu(gpu, phi, prime, divisorData, phiFactors))
        {
            return phi;
        }

        UInt128 candidateOrder = InitializeStartingOrderWideGpu(gpu, prime, phi, divisorData);
        candidateOrder = ExponentLoweringWideGpu(gpu, candidateOrder, prime, divisorData, phiFactors);

        if (TryConfirmOrderWideGpu(gpu, prime, candidateOrder, divisorData, config))
        {
            return candidateOrder;
        }

        if (config.StrictMode)
        {
            return FinishStrictlyWideGpu(gpu, prime, divisorData);
        }

        if (TryHeuristicFinishWideGpu(gpu, prime, candidateOrder, previousOrder, divisorData, config, out UInt128 order))
        {
            return order;
        }

        return candidateOrder;
    }

    private static UInt128 FinishStrictlyWideGpu(PrimeOrderCalculatorAccelerator gpu, in UInt128 prime, in MontgomeryDivisorData? divisorData)
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
                if ((order % primeFactor) != UInt128.Zero)
                {
                    UInt128 candidate = order / primeFactor;
                    if (Pow2ModWideGpu(gpu, candidate, prime, divisorData) == UInt128.One)
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

    private static bool TrySpecialMaxWideGpu(PrimeOrderCalculatorAccelerator gpu, in UInt128 phi, in UInt128 prime, in MontgomeryDivisorData? divisorData, in PartialFactorResult128 factors)
    {
        ReadOnlySpan<FactorEntry128> factorSpan = factors.Factors;
        int length = factors.Count;
        for (int i = 0; i < length; i++)
        {
            UInt128 factor = factorSpan[i].Value;
            UInt128 reduced = phi / factor;
            if (Pow2ModWideGpu(gpu, reduced, prime, divisorData) == UInt128.One)
            {
                return false;
            }
        }

        return true;
    }

    private static UInt128 InitializeStartingOrderWideGpu(PrimeOrderCalculatorAccelerator gpu, in UInt128 prime, in UInt128 phi, in MontgomeryDivisorData? divisorData)
    {
        UInt128 order = phi;
        UInt128 mod8 = prime & (UInt128)7UL;
        if (mod8 == UInt128.One || mod8 == (UInt128)7UL)
        {
            UInt128 half = phi >> 1;
            if (Pow2ModWideGpu(gpu, half, prime, divisorData) == UInt128.One)
            {
                order = half;
            }
        }

        return order;
    }

    private static UInt128 ExponentLoweringWideGpu(PrimeOrderCalculatorAccelerator gpu, UInt128 order, in UInt128 prime, in MontgomeryDivisorData? divisorData, in PartialFactorResult128 factors)
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

		// TODO: Investigating using stepper to replace modulo operation
        for (int i = 0; i < length; i++)
        {
            UInt128 primeFactor = buffer[i].Value;
            int exponent = buffer[i].Exponent;
            for (int iteration = 0; iteration < exponent; iteration++)
            {
                if ((order % primeFactor) == UInt128.Zero)
                {
                    UInt128 reduced = order / primeFactor;
                    if (Pow2ModWideGpu(gpu, reduced, prime, divisorData) == UInt128.One)
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

    private static bool TryConfirmOrderWideGpu(PrimeOrderCalculatorAccelerator gpu, in UInt128 prime, in UInt128 order, in MontgomeryDivisorData? divisorData, in PrimeOrderCalculatorConfig config)
    {
        if (order == UInt128.Zero)
        {
            return false;
        }

        if (Pow2ModWideGpu(gpu, order, prime, divisorData) != UInt128.One)
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
				if (Pow2ModWideGpu(gpu, reduced, prime, divisorData) == UInt128.One)
				{
					return false;
				}
			}
        }

        return true;
    }

    private static bool TryHeuristicFinishWideGpu(
		PrimeOrderCalculatorAccelerator gpu,
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

            if (Pow2ModWideGpu(gpu, candidate, prime, divisorData) != UInt128.One)
            {
                continue;
            }

            if (!TryConfirmCandidateWideGpu(gpu, prime, candidate, config, ref powUsed, powBudget, divisorData))
            {
                continue;
            }

            result = candidate;
            return true;
        }

        return false;
    }

	private static ulong CalculateByFactorizationGpu(PrimeOrderCalculatorAccelerator gpu, ulong prime, in MontgomeryDivisorData divisorData)
	{
		ulong phi = prime - 1UL;
		Dictionary<ulong, int> counts = gpu.Pow2ModEntriesToTestOnHost;
		counts.Clear();
		
		FactorCompletelyCpu(phi, counts, false);
		if (counts.Count == 0)
		{
			return phi;
		}

		int entryCount = counts.Count;
		Span<KeyValuePair<ulong, int>> entries = stackalloc KeyValuePair<ulong, int>[entryCount];

		int stored = 0;
		foreach (var entry in counts)
		{
			entries[stored++] = entry;
		}
		
		entries.Sort(entries, static (a, b) => a.Key.CompareTo(b.Key));

		ulong order = phi;
		var acceleratorIndex = gpu.AcceleratorIndex;
		var kernel = gpu.CheckFactorsKernel;

		gpu.EnsureCapacity(entryCount, 1);
		var pow2ModEntriesToTestOnDeviceView = gpu.Pow2ModEntriesToTestOnDeviceView;

		AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		pow2ModEntriesToTestOnDeviceView.CopyFromCPU(stream, entries);

		kernel.Launch(stream, 1, entryCount, phi, pow2ModEntriesToTestOnDeviceView, divisorData.Modulus, divisorData.NPrime, divisorData.MontgomeryOne, divisorData.MontgomeryTwo, divisorData.MontgomeryTwoSquared, gpu.OutputUlongView);

		gpu.OutputUlongView.CopyToCPU(stream, ref order, 1);
		stream.Synchronize();

		AcceleratorStreamPool.Return(acceleratorIndex, stream);
		return order;
	}

	private static bool TryPopulateSmallPrimeFactorsGpu(PrimeOrderCalculatorAccelerator gpu, ulong value, uint limit, Dictionary<ulong, int> counts, out int factorCount, out ulong remaining)
	{
		var primeBufferArray = ThreadStaticPools.UlongPool.Rent(GpuSmallPrimeFactorSlots);
		var exponentBufferArray = ThreadStaticPools.IntPool.Rent(GpuSmallPrimeFactorSlots);
		Span<ulong> primeBuffer = primeBufferArray.AsSpan(0, GpuSmallPrimeFactorSlots);
		Span<int> exponentBuffer = exponentBufferArray.AsSpan(0, GpuSmallPrimeFactorSlots);
		remaining = value;

		gpu.EnsureSmallPrimeFactorSlotsCapacity(GpuSmallPrimeFactorSlots);
		int acceleratorIndex = gpu.AcceleratorIndex;
		var accelerator = gpu.Accelerator;

		var kernelLauncher = gpu.SmallPrimeFactorKernelLauncher;
		ArrayView1D<int, Stride1D.Dense> smallPrimeFactorCountSlotView = gpu.OutputIntView2;
		ArrayView1D<ulong, Stride1D.Dense> smallPrimeFactorRemainingSlotView = gpu.InputView;
		ArrayView1D<ulong, Stride1D.Dense> smallPrimeFactorPrimeSlotsView = gpu.OutputUlongView;
		ArrayView1D<int, Stride1D.Dense> smallPrimeFactorExponentSlotsView = gpu.OutputIntView;
		ArrayView1D<ulong, Stride1D.Dense> smallPrimeFactorsSquaresView = gpu.SmallPrimeFactorSquares;
		ArrayView1D<uint, Stride1D.Dense> smallPrimeFactorsPrimesView = gpu.SmallPrimeFactorPrimes;

		// GpuPrimeWorkLimiter.Acquire();
		var stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		kernelLauncher(
				stream,
				1,
				value,
				limit,
				smallPrimeFactorsPrimesView,
				smallPrimeFactorsSquaresView,
				(int)smallPrimeFactorsPrimesView.Length,
				smallPrimeFactorPrimeSlotsView,
				smallPrimeFactorExponentSlotsView,
				smallPrimeFactorCountSlotView,
				smallPrimeFactorRemainingSlotView);

		factorCount = 0;
		smallPrimeFactorCountSlotView.CopyToCPU(stream, ref factorCount, 1);
		stream.Synchronize();

		factorCount = Math.Min(factorCount, GpuSmallPrimeFactorSlots);

		if (factorCount > 0)
		{
			primeBuffer = primeBuffer[..factorCount];
			smallPrimeFactorPrimeSlotsView.CopyToCPU(stream, primeBuffer);

			exponentBuffer = exponentBuffer[..factorCount];
			smallPrimeFactorExponentSlotsView.CopyToCPU(stream, exponentBuffer);
		}

		smallPrimeFactorRemainingSlotView.CopyToCPU(stream, ref remaining, 1);
		stream.Synchronize();

		AcceleratorStreamPool.Return(acceleratorIndex, stream);
		// GpuPrimeWorkLimiter.Release();

		for (int i = 0; i < factorCount; i++)
		{
			ulong primeValue = primeBuffer[i];
			int exponent = exponentBuffer[i];
			counts.Add(primeValue, exponent);
		}

		return true;
	}

	private static bool EvaluateSpecialMaxCandidatesGpu(
			PrimeOrderCalculatorAccelerator gpu,
			ReadOnlySpan<ulong> factors,
			ulong phi,
			ulong prime,
			in MontgomeryDivisorData divisorData)
	{
		int factorCount = factors.Length;
		gpu.EnsureUlongInputOutputCapacity(factorCount);

		int acceleratorIndex = gpu.AcceleratorIndex;
		// GpuPrimeWorkLimiter.Acquire();
		ArrayView<ulong> specialMaxFactorsView = gpu.InputView;
		ArrayView<ulong> specialMaxResultView = gpu.OutputUlongView2;
		var kernelLauncher = gpu.SpecialMaxKernelLauncher;

		var stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		specialMaxFactorsView.SubView(0, factorCount).CopyFromCPU(stream, factors);

		kernelLauncher(
				stream,
				1,
				phi,
				specialMaxFactorsView,
				factorCount,
				divisorData.Modulus,
				specialMaxResultView);

		ulong result = 0UL;
		specialMaxResultView.CopyToCPU(stream, ref result, 1);
		stream.Synchronize();
		
		AcceleratorStreamPool.Return(acceleratorIndex, stream);
		// GpuPrimeWorkLimiter.Release();
		return result != 0;
	}

    private static UInt128 Pow2ModWideGpu(PrimeOrderCalculatorAccelerator gpu, in UInt128 exponent, in UInt128 modulus, in MontgomeryDivisorData? divisorData)
    {
        if (modulus == UInt128.One)
        {
            return UInt128.Zero;
        }

		if (modulus <= (UInt128)ulong.MaxValue && exponent <= (UInt128)ulong.MaxValue)
		{
			ulong prime64 = (ulong)modulus;
			ulong exponent64 = (ulong)exponent;
			GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2Mod(gpu, exponent64, prime64, out ulong remainder, divisorData);
			if (status == GpuPow2ModStatus.Success)
			{
				return remainder;
			}
		}

		GpuPow2ModStatus wideStatus = PrimeOrderGpuHeuristics.TryPow2Mod(gpu, exponent, modulus, out UInt128 wideRemainder);
		if (wideStatus == GpuPow2ModStatus.Success)
		{
			return wideRemainder;
		}

        return exponent.Pow2MontgomeryModWindowed(modulus);
    }

    private static bool TryConfirmCandidateWideGpu(PrimeOrderCalculatorAccelerator gpu, in UInt128 prime, in UInt128 candidate, in PrimeOrderCalculatorConfig config, ref int powUsed, int powBudget, in MontgomeryDivisorData? divisorData)
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
                if ((reduced % primeFactor) != UInt128.Zero)
                {
                    break;
                }

                reduced /= primeFactor;
                if (powUsed >= powBudget && powBudget > 0)
                {
                    return false;
                }

                powUsed++;
                if (Pow2ModWideGpu(gpu, reduced, prime, divisorData) == UInt128.One)
                {
                    return false;
                }
            }
        }

        return true;
    }

	private static bool TryPollardRhoGpu(PrimeOrderCalculatorAccelerator gpu, ulong n, out ulong factor)
	{
		Span<ulong> randomStateSpan = stackalloc ulong[1]; 
		randomStateSpan[0] = ThreadStaticDeterministicRandomGpu.Exclusive.State;

		Span<byte> factoredSpan = stackalloc byte[1];
		Span<ulong> factorSpan = stackalloc ulong[1];
		
		int acceleratorIndex = gpu.AcceleratorIndex;
		var kernel = gpu.PollardRhoKernel;

		AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		gpu.InputView.CopyFromCPU(stream, randomStateSpan);

		kernel.Launch(stream, 1, n, 1, gpu.InputView, gpu.OutputByteView, gpu.OutputUlongView);

		gpu.OutputByteView.CopyToCPU(stream, factoredSpan);
		gpu.OutputUlongView.CopyToCPU(stream, factorSpan);
		gpu.InputView.CopyToCPU(stream, randomStateSpan);
		stream.Synchronize();

		AcceleratorStreamPool.Return(acceleratorIndex, stream);
		ThreadStaticDeterministicRandomGpu.Exclusive.SetState(randomStateSpan[0]);

		bool factored = factoredSpan[0] != 0;
		factor = factored ? factorSpan[0] : 0UL;
		return factored;
	}
}

