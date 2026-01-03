using System.Buffers;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Cpu;
using PerfectNumbers.Core.Gpu;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core;

[DeviceDependentTemplate(typeof(ComputationDevice))]
public static partial class PrimeOrderCalculatorTemplate
{
	// TODO: Remove branching on the CPU / GPU paths
	// TODO: Split big non-static functions into smaller / extract static code to limit JIT time

    private static readonly ConcurrentDictionary<int, ulong> _dictionaryStats = new(20_480, 64);

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
				x = AdvancePolynomial(x, c, n);
				y = AdvancePolynomialTwice(y, c, n);
				ulong diff = x > y ? x - y : y - x;
				d = BinaryGcd(diff, n);
			}

			if (d != n)
			{
				return d;
			}
		}
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

    private static void PrintStats(int count)
    {
        _dictionaryStats.AddOrUpdate(count, 1UL, static (_, existing) => existing + 1UL);
        Console.WriteLine("Current stats:");
        foreach (KeyValuePair<int, ulong> entry in _dictionaryStats)
        {
            Console.WriteLine($"Count {entry.Key} = {entry.Value}");
        }
    }

    private static CandidateKey128 BuildKeyWide(
        in UInt128 value,
        in UInt128 prime,
        in UInt128 previous,
        int previousGroup,
        bool hasPrevious)
    {
        int group = GetGroupWide(value, prime);
        if (group == 0)
        {
            return new CandidateKey128(int.MaxValue, false, UInt128.Zero, false, UInt128.Zero);
        }

        bool isGe = !hasPrevious || value >= previous;
        int primary = ComputePrimary(group, isGe, previousGroup);

        if (group == 3)
        {
            return new CandidateKey128(primary, true, value, true, value);
        }

        UInt128 distance = hasPrevious ? (value > previous ? value - previous : previous - value) : value;
        return new CandidateKey128(primary, false, distance, false, value);
    }

    private static int CompareComponents(bool descendingX, in UInt128 valueX, bool descendingY, in UInt128 valueY)
    {
        if (descendingX == descendingY)
        {
            if (valueX == valueY)
            {
                return 0;
            }

            if (descendingX)
            {
                return valueX > valueY ? -1 : 1;
            }

            return valueX > valueY ? 1 : -1;
        }

        if (valueX == valueY)
        {
            return 0;
        }

        return valueX > valueY ? 1 : -1;
    }

    private static int GetGroupWide(in UInt128 value, in UInt128 prime)
    {
        UInt128 threshold = prime >> 3;
        if (value <= threshold)
        {
            return 1;
        }

        threshold = prime >> 2;
        if (value <= threshold)
        {
            return 2;
        }

        threshold = (UInt128)(((BigInteger)prime * 3) >> 3);
        if (value <= threshold)
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

    private static void BuildCandidatesWide(in UInt128 order, FactorEntry128[] factors, int count, List<UInt128> candidates, int limit)
    {
        if (count == 0)
        {
            return;
        }

        Span<FactorEntry128> buffer = factors.AsSpan(0, count);
        buffer.Sort(static (a, b) => a.Value.CompareTo(b.Value));
        BuildCandidatesRecursiveWide(order, buffer, 0, UInt128.One, candidates, limit);
    }

    private static void BuildCandidatesRecursiveWide(
        in UInt128 order,
        in ReadOnlySpan<FactorEntry128> factors,
        int index,
        in UInt128 divisorProduct,
        List<UInt128> candidates,
        int limit)
    {
        if (candidates.Count >= limit)
        {
            return;
        }

        if (index >= factors.Length)
        {
            if (divisorProduct == UInt128.One || divisorProduct == order)
            {
                return;
            }

            UInt128 candidate = order / divisorProduct;
            if (candidate > UInt128.One && candidate < order)
            {
                candidates.Add(candidate);
            }

            return;
        }

        FactorEntry128 factor = factors[index];
		index++;
        UInt128 primeFactor = factor.Value;
        UInt128 contribution = UInt128.One;
        for (int exponent = 0; exponent <= factor.Exponent; exponent++)
        {
            UInt128 nextDivisor = divisorProduct * contribution;
            if (nextDivisor > order)
            {
                break;
            }

            BuildCandidatesRecursiveWide(order, factors, index, nextDivisor, candidates, limit);
            if (candidates.Count >= limit)
            {
                return;
            }

            if (exponent == factor.Exponent)
            {
                break;
            }

            if (contribution > order / primeFactor)
            {
                break;
            }

            contribution *= primeFactor;
        }
    }

    private static PartialFactorResult128 PartialFactorWide(in UInt128 value, in PrimeOrderCalculatorConfig config)
    {
        if (value <= UInt128.One)
        {
            return PartialFactorResult128.Empty;
        }

        Dictionary<UInt128, int> counts = new(capacity: 8);
        uint limit = config.SmallFactorLimit == 0 ? uint.MaxValue : config.SmallFactorLimit;
        UInt128 remaining;

        // if (IsGpuPow2Allowed && value <= ulong.MaxValue)
        // {
        //     Dictionary<ulong, int> narrowCounts = new(capacity: 8);
        //     if (TryPopulateSmallPrimeFactorsGpu((ulong)value, limit, narrowCounts, out ulong narrowRemaining))
        //     {
        //         foreach (KeyValuePair<ulong, int> entry in narrowCounts)
        //         {
        //             counts[(UInt128)entry.Key] = entry.Value;
        //         }

        //         remaining = (UInt128)narrowRemaining;
        //     }
        //     else
        //     {
        //         remaining = PopulateSmallPrimeFactorsCpuWide(value, limit, counts);
        //     }
        // }
        // else
        // {
            remaining = PopulateSmallPrimeFactorsCpuWide(value, limit, counts);
        // }

        List<UInt128> pending = new();
        if (remaining > UInt128.One)
        {
            pending.Add(remaining);
        }

        if (config.PollardRhoMilliseconds > 0 && pending.Count > 0)
        {
            long deadlineTimestamp = CreateDeadlineTimestamp(config.PollardRhoMilliseconds);
            Stack<UInt128> stack = new();
            stack.Push(remaining);
            pending.Clear();

            long timestamp = 0L; // reused for deadline checks.
            while (stack.Count > 0)
            {
                UInt128 composite = stack.Pop();
                if (composite <= UInt128.One)
                {
                    continue;
                }

                if (IsPrimeWide(composite))
                {
                    AddFactor(counts, composite, 1);
                    continue;
                }

                timestamp = Stopwatch.GetTimestamp();
                if (timestamp > deadlineTimestamp)
                {
                    pending.Add(composite);
                    continue;
                }

                if (!TryPollardRhoWide(composite, deadlineTimestamp, out UInt128 factor))
                {
                    pending.Add(composite);
                    continue;
                }

                UInt128 quotient = composite / factor;
                stack.Push(factor);
                stack.Push(quotient);
            }
        }

        UInt128 cofactor = UInt128.One;
        int pendingCount = pending.Count;
        for (int i = 0; i < pendingCount; i++)
        {
            UInt128 composite = pending[i];
            if (IsPrimeWide(composite))
            {
                AddFactor(counts, composite, 1);
            }
            else
            {
                BigInteger product = (BigInteger)cofactor * (BigInteger)composite;
                cofactor = (UInt128)product;
            }
        }

        if (counts.Count == 0 && cofactor == value)
        {
            return new PartialFactorResult128(null, value, false, 0);
        }

		ArrayPool<FactorEntry128>? pool = null;
		FactorEntry128[]? rented = counts.Count > 0 ?  (pool = ThreadStaticPools.FactorEntry128Pool).Rent(counts.Count) : null;
        int index = 0;
        if (rented is not null)
        {
            foreach (KeyValuePair<UInt128, int> entry in counts)
            {
                rented[index] = new FactorEntry128(entry.Key, entry.Value);
                index++;
            }
        }

        FactorEntry128[]? resultArray = null;
        if (pool is not null)
        {
            Span<FactorEntry128> span = rented.AsSpan(0, index);
            span.Sort(static (a, b) => a.Value.CompareTo(b.Value));
            resultArray = new FactorEntry128[index];
            span.CopyTo(resultArray);
            pool.Return(rented!, clearArray: false);
        }

        bool fullyFactored = cofactor == UInt128.One;
        return new PartialFactorResult128(resultArray, cofactor, fullyFactored, index);
    }

    private static UInt128 PopulateSmallPrimeFactorsCpuWide(in UInt128 value, uint limit, Dictionary<UInt128, int> counts)
    {
        UInt128 remaining = value;
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

            UInt128 primeSquare = (UInt128)squares[i];
            if (primeSquare > remaining)
            {
                break;
            }

            UInt128 primeValue = primeCandidate;
            if ((remaining % primeValue) != UInt128.Zero)
            {
                continue;
            }

            int exponent = 0;
            do
            {
                remaining /= primeValue;
                exponent++;
            }
            while ((remaining % primeValue) == UInt128.Zero);

            counts[primeValue] = exponent;
        }

        return remaining;
    }
    private static bool TryPollardRhoWide(in UInt128 n, long deadlineTimestamp, out UInt128 factor)
    {
        factor = UInt128.Zero;
        if ((n & UInt128.One) == UInt128.Zero)
        {
            factor = (UInt128)2UL;
            return true;
        }

        UInt128 c = UInt128.One;
        UInt128 x = (UInt128)2UL;
        UInt128 y = x;
        long timestamp = 0L; // reused for deadline checks.

        while (true)
        {
            timestamp = Stopwatch.GetTimestamp();
            if (timestamp > deadlineTimestamp)
            {
                return false;
            }

            x = AdvancePolynomialWide(x, c, n);
            y = AdvancePolynomialWide(y, c, n);
            y = AdvancePolynomialWide(y, c, n);

            UInt128 diff = x > y ? x - y : y - x;
            UInt128 d = BinaryGcdWide(diff, n);
            if (d == n)
            {
                c += UInt128.One;
                x = (UInt128)2UL;
                y = x;
                continue;
            }

            if (d > UInt128.One)
            {
                factor = d;
                return true;
            }
        }
    }

    private static UInt128 AdvancePolynomialWide(in UInt128 x, in UInt128 c, in UInt128 modulus)
    {
        BigInteger value = (BigInteger)x;
        value = (value * value + (BigInteger)c) % (BigInteger)modulus;
        return (UInt128)value;
    }

    private static UInt128 BinaryGcdWide(UInt128 a, UInt128 b)
    {
        if (a == UInt128.Zero)
        {
            return b;
        }

        if (b == UInt128.Zero)
        {
            return a;
        }

        int shift = TrailingZeroCountWide(a | b);
        a >>= TrailingZeroCountWide(a);

        while (true)
        {
            b >>= TrailingZeroCountWide(b);
            if (a > b)
            {
                (a, b) = (b, a);
            }

            b -= a;
            if (b == UInt128.Zero)
            {
                return a << shift;
            }
        }
    }

    private static int TrailingZeroCountWide(in UInt128 value)
    {
        if (value == UInt128.Zero)
        {
            return 128;
        }

        ulong low = (ulong)value;
        if (low != 0UL)
        {
            return BitOperations.TrailingZeroCount(low);
        }

        ulong high = (ulong)(value >> 64);
        return 64 + BitOperations.TrailingZeroCount(high);
    }

    private static void FactorCompletelyWide(in UInt128 value, Dictionary<UInt128, int> counts)
    {
        if (value <= UInt128.One)
        {
            return;
        }

        if (IsPrimeWide(value))
        {
            AddFactor(counts, value, 1);
            return;
        }

        UInt128 factor = PollardRhoWide(value);
        FactorCompletelyWide(factor, counts);
        FactorCompletelyWide(value / factor, counts);
    }

    private static UInt128 PollardRhoWide(in UInt128 n)
    {
        if ((n & UInt128.One) == UInt128.Zero)
        {
            return (UInt128)2UL;
        }

        UInt128 c = UInt128.One;
        while (true)
        {
            UInt128 x = (UInt128)2UL;
            UInt128 y = x;
            UInt128 d = UInt128.One;

            while (d == UInt128.One)
            {
                x = AdvancePolynomialWide(x, c, n);
                y = AdvancePolynomialWide(y, c, n);
                y = AdvancePolynomialWide(y, c, n);

                UInt128 diff = x > y ? x - y : y - x;
                d = BinaryGcdWide(diff, n);
            }

            if (d != n)
            {
                return d;
            }

            c += UInt128.One;
        }
    }

    private static bool IsPrimeWide(in UInt128 value)
    {
        if (value <= ulong.MaxValue)
        {
            // return HeuristicPrimeTester.Exclusive.IsPrimeCpu((ulong)value, CancellationToken.None);
            return PrimeTester.IsPrimeCpu((ulong)value);
        }

        if ((value & UInt128.One) == UInt128.Zero)
        {
            return false;
        }

        uint[] smallPrimes = PrimesGenerator.SmallPrimes;
        int length = smallPrimes.Length;
        for (int i = 0; i < length; i++)
        {
            uint prime = smallPrimes[i];
            UInt128 primeValue = prime;
            UInt128 primeSquare = primeValue * primeValue;
            if (primeSquare > value)
            {
                break;
            }

            if ((value % primeValue) == UInt128.Zero)
            {
                return false;
            }
        }

        BigInteger n = (BigInteger)value;
        BigInteger d = n - BigInteger.One;
        int s = 0;
        while ((d & 1) == 0)
        {
            d >>= 1;
            s++;
        }

        int[] bases = new[] { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };
        for (int i = 0; i < bases.Length; i++)
        {
            int baseValue = bases[i];
            if (baseValue <= 1)
            {
                continue;
            }

            BigInteger a = baseValue;
            if (a >= n)
            {
                continue;
            }

            BigInteger x = BigInteger.ModPow(a, d, n);
            if (x == BigInteger.One || x == n - BigInteger.One)
            {
                continue;
            }

            bool witnessFound = true;
            for (int r = 1; r < s; r++)
            {
                x = BigInteger.ModPow(x, 2, n);
                if (x == n - BigInteger.One)
                {
                    witnessFound = false;
                    break;
                }
            }

            if (witnessFound)
            {
                return false;
            }
        }

        return true;
    }

    private static void AddFactor(Dictionary<UInt128, int> counts, in UInt128 prime, int exponent)
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

    private readonly struct CandidateKey128(int primary, bool secondaryDescending, in UInt128 secondary, bool tertiaryDescending, in UInt128 tertiary)
	{
		public int Primary { get; } = primary;

		public bool SecondaryDescending { get; } = secondaryDescending;

		public UInt128 Secondary { get; } = secondary;

		public bool TertiaryDescending { get; } = tertiaryDescending;

		public UInt128 Tertiary { get; } = tertiary;
	}

    private readonly struct PartialFactorResult128(FactorEntry128[]? factors, UInt128 cofactor, bool fullyFactored, int count)
	{
		public FactorEntry128[]? Factors { get; } = factors;

		public UInt128 Cofactor { get; } = cofactor;

		public bool FullyFactored { get; } = fullyFactored;

		public int Count { get; } = count;

		public static PartialFactorResult128 Empty => new(null, UInt128.One, true, 0);

        public PartialFactorResult128 WithAdditionalPrime(UInt128 prime)
        {
            if (Factors is null)
            {
                FactorEntry128[] local = new FactorEntry128[1];
                local[0] = new FactorEntry128(prime, 1);
                return new PartialFactorResult128(local, UInt128.One, true, 1);
            }

            FactorEntry128[] extended = new FactorEntry128[Count + 1];
            Array.Copy(Factors, extended, Count);
            extended[Count] = new FactorEntry128(prime, 1);
            return new PartialFactorResult128(extended, UInt128.One, true, Count + 1);
        }
    }

    public static ulong Calculate(
		#if DEVICE_HYBRID || DEVICE_GPU
			PrimeOrderCalculatorAccelerator gpu,
		#endif
        ulong prime,
        ulong? previousOrder,
        in MontgomeryDivisorData divisorData,
        in PrimeOrderCalculatorConfig config)
    {
        if (prime <= 3UL)
        {
            return prime == 3UL ? 2UL : 1UL;
        }

        ulong phi = prime - 1UL;

		#if DEVICE_GPU
			if (PrimeOrderGpuHeuristics.TryCalculateOrder(gpu, prime, previousOrder, config, divisorData, out ulong gpuOrder))
			{
				return gpuOrder;
			}

			PartialFactorResult phiFactors = PartialFactor(gpu, phi, divisorData, config);
		#elif DEVICE_HYBRID
			PartialFactorResult phiFactors = PartialFactor(gpu, phi, divisorData, config);
		#else
			PartialFactorResult phiFactors = PartialFactor(phi, divisorData, config);
		#endif

        ulong result;
		if (phiFactors.Factors is null)
        {
			result = CalculateByFactorization(prime, divisorData);
            phiFactors.Dispose();
            return result;
        }

		#if DEVICE_HYBRID || DEVICE_GPU
			result = RunHeuristicPipeline(gpu, prime, previousOrder, config, divisorData, phi, phiFactors);
		#else
			result = RunHeuristicPipeline(prime, previousOrder, config, divisorData, phi, phiFactors);
		#endif
        phiFactors.Dispose();
        return result;
    }

    public static UInt128 Calculate(
		#if DEVICE_HYBRID || DEVICE_GPU
			PrimeOrderCalculatorAccelerator gpu,
		#endif
        in UInt128 prime,
        in UInt128? previousOrder,
        in PrimeOrderCalculatorConfig config)
    {
        if (prime <= ulong.MaxValue)
        {
            ulong? previous = null;
            if (previousOrder.HasValue)
            {
                UInt128 previousValue = previousOrder.Value;
                previous = previousValue <= ulong.MaxValue ? (ulong)previousValue : ulong.MaxValue;
            }

            ulong prime64 = (ulong)prime;
            MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(prime64);
			#if DEVICE_HYBRID || DEVICE_GPU
				ulong order64 = Calculate(gpu, prime64, previous, divisorData, config);
			#else
				ulong order64 = Calculate(prime64, previous, divisorData, config);
			#endif
            return order64 == 0UL ? UInt128.Zero : (UInt128)order64;
        }

		#if DEVICE_GPU
			return CalculateWideInternal(gpu, prime, previousOrder, MontgomeryDivisorData.Empty, config);
		#else
			return CalculateWideInternal(prime, previousOrder, MontgomeryDivisorData.Empty, config);
        #endif
    }

    public static UInt128 Calculate(
        #if DEVICE_HYBRID || DEVICE_GPU
            PrimeOrderCalculatorAccelerator gpu,
        #endif
        in UInt128 prime,
        in UInt128? previousOrder,
        in MontgomeryDivisorData divisorData,
        in PrimeOrderCalculatorConfig config)
    {
        if (prime <= ulong.MaxValue)
        {
            ulong? previous = null;
            if (previousOrder.HasValue)
            {
                UInt128 previousValue = previousOrder.Value;
                previous = previousValue <= ulong.MaxValue ? (ulong)previousValue : ulong.MaxValue;
            }

            ulong prime64 = (ulong)prime;
            MontgomeryDivisorData effectiveDivisorData = divisorData.Equals(MontgomeryDivisorData.Empty) ? MontgomeryDivisorData.FromModulus(prime64) : divisorData;
            #if DEVICE_HYBRID || DEVICE_GPU
                ulong order64 = Calculate(gpu, prime64, previous, effectiveDivisorData, config);
            #else
                ulong order64 = Calculate(prime64, previous, effectiveDivisorData, config);
            #endif
            return order64 == 0UL ? UInt128.Zero : (UInt128)order64;
        }

        #if DEVICE_GPU
            return CalculateWideInternal(gpu, prime, previousOrder, divisorData, config);
        #else
            return CalculateWideInternal(prime, previousOrder, divisorData, config);
        #endif
    }

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static ulong CalculateByFactorization(ulong prime, in MontgomeryDivisorData divisorData)
	{
		ulong phi = prime - 1UL;
		Dictionary<ulong, int> counts = ThreadStaticPools.RentFactorCountDictionary();
		FactorCompletely(phi, counts, false);
		PrintStats(counts.Count);
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

				if (candidate.Pow2MontgomeryModWindowedKeepMontgomeryCpu(divisorData) == divisorData.MontgomeryOne)
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

    private static UInt128 CalculateWideInternal(
		#if DEVICE_GPU
			PrimeOrderCalculatorAccelerator gpu,
		#endif
        in UInt128 prime,
        in UInt128? previousOrder,
        in MontgomeryDivisorData divisorData,
        in PrimeOrderCalculatorConfig config)
    {
        if (prime <= UInt128.One)
        {
            return UInt128.One;
        }

        if (prime == UInt128Numbers.Three)
        {
            return UInt128Numbers.Two;
        }

        UInt128 phi = prime - UInt128.One;
        PartialFactorResult128 phiFactors = PartialFactorWide(phi, config);
        if (phiFactors.Factors is null)
        {
			#if DEVICE_GPU
				return FinishStrictlyWide(gpu, prime, divisorData);
			#else
				return FinishStrictlyWide(prime, divisorData);
			#endif
        }

		#if DEVICE_GPU
			UInt128 candidateOrder = InitializeStartingOrderWide(gpu, prime, phi, divisorData);
			candidateOrder = ExponentLoweringWide(gpu, candidateOrder, prime, divisorData, phiFactors);
		#else
			UInt128 candidateOrder = InitializeStartingOrderWide(prime, phi, divisorData);
			candidateOrder = ExponentLoweringWide(candidateOrder, prime, divisorData, phiFactors);
		#endif

		#if DEVICE_GPU
			if (TryConfirmOrderWide(gpu, prime, candidateOrder, divisorData, config))
		#else
			if (TryConfirmOrderWide(prime, candidateOrder, divisorData, config))
		#endif
        {
            return candidateOrder;
        }

        if (config.StrictMode)
        {
			#if DEVICE_GPU
				return FinishStrictlyWide(gpu, prime, divisorData);
			#else
				return FinishStrictlyWide(prime, divisorData);
			#endif
        }

		#if DEVICE_GPU
			if (TryHeuristicFinishWide(gpu, prime, candidateOrder, previousOrder, divisorData, config, out UInt128 order))
		#else
			if (TryHeuristicFinishWide(prime, candidateOrder, previousOrder, divisorData, config, out UInt128 order))
		#endif
        {
            return order;
        }

        return candidateOrder;
    }

    private static bool CheckCandidateViolation(
        Span<ulong> buffer,
        ulong primeFactor,
        int exponent,
        ulong candidate,
        ulong prime,
        ref int powUsed,
        int powBudget,
        ref ExponentRemainderStepperCpu stepper)
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

#if DEVICE_CPU || DEVICE_HYBRID
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void FactorCompletely(ulong value, Dictionary<ulong, int> counts, bool knownComposite)
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
			isPrime = PrimeTesterByLastDigit.IsPrimeCpu(quotient);
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
        ulong prime)
    {
        for (int i = 0; i < count; i++)
        {
            if (primes[i] == prime)
            {
                exponents[i]++;
                return;
            }
        }

        primes[count] = prime;
        exponents[count] = 1;
        count++;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static void AddFactorToDictionary(Dictionary<ulong, int> counts, ulong prime)
    {
        if (counts.TryGetValue(prime, out int existing))
        {
            counts[prime] = existing + 1;
        }
        else
        {
            counts[prime] = 1;
        }
    }
#endif

#if DEVICE_CPU || DEVICE_HYBRID
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void PopulateSmallPrimeFactors(
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
            if (primeCandidate > limit || squares[i] > value)
            {
                break;
            }

            ulong primeValue = primeCandidate;
            ulong quotient = value / primeValue;
            if (value - (quotient * primeValue) != 0UL)
            {
                continue;
            }

            int exponent = ExtractSmallPrimeExponent(ref value, primeValue, quotient);

            primeTargets[factorCount] = primeValue;
            exponentTargets[factorCount] = exponent;
            factorCount++;
        }

        remaining = value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static int ExtractSmallPrimeExponent(ref ulong value, ulong primeValue, ulong quotient)
    {
        ulong dividend = quotient;

        if (dividend < primeValue)
        {
            value = dividend;
            return 1;
        }

        quotient = dividend / primeValue;
        if (dividend - quotient * primeValue != 0UL)
        {
            value = dividend;
            return 1;
        }

        dividend = quotient;

        if (dividend < primeValue)
        {
            value = dividend;
            return 2;
        }

        quotient = dividend / primeValue;
        if (dividend - quotient * primeValue != 0UL)
        {
            value = dividend;
            return 2;
        }

		return ExtractSmallPrimeExponentWithLoop(ref value, primeValue, quotient);
    }

    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static int ExtractSmallPrimeExponentWithLoop(ref ulong value, ulong primeValue, ulong quotient)
	{
		ulong dividend = quotient;
        int exponent = 3;

        while (true)
        {
            quotient = dividend / primeValue;

            if (dividend - quotient * primeValue != 0UL)
            {
                break;
            }

            dividend = quotient;
            exponent++;
        }

        value = dividend;
        return exponent;
	}

#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static void AddFactorToDictionary(Dictionary<UInt128, int> counts, UInt128 prime)
    {
        if (counts.TryGetValue(prime, out int existing))
        {
            counts[prime] = existing + 1;
        }
        else
        {
            counts[prime] = 1;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static ulong AdvancePolynomial(ulong x, ulong c, ulong modulus) => unchecked(x * x % modulus + c) % modulus;

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static ulong AdvancePolynomialTwice(ulong x, ulong c, ulong modulus)
    {
		x = unchecked(x * x % modulus + c) % modulus;
        return unchecked(x * x % modulus + c) % modulus;
    }

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static void CollectFactors(
		#if DEVICE_HYBRID || DEVICE_GPU
			PrimeOrderCalculatorAccelerator gpu,
		#endif
		Span<ulong> primeSlots,
		Span<int> exponentSlots,
		ref int factorCount,
		List<PartialFactorPendingEntry> pending,
		FixedCapacityStack<ulong> compositeStack,
		long deadlineTimestamp,
		out bool pollardRhoDeadlineReached)
	{
		bool limitReached = false;
		while (compositeStack.Count > 0)
		{
			ulong composite = compositeStack.Pop();
			#if DEVICE_GPU
				bool isPrime = HeuristicCombinedPrimeTester.IsPrimeGpu(gpu, composite);
			#elif DEVICE_HYBRID
				bool isPrime = RunOnCpu() ? PrimeTesterByLastDigit.IsPrimeCpu(composite) : HeuristicCombinedPrimeTester.IsPrimeGpu(gpu, composite);
			#else
				bool isPrime = PrimeTesterByLastDigit.IsPrimeCpu(composite);
			#endif

			if (isPrime)
			{
				AddFactorToCollector(primeSlots, exponentSlots, ref factorCount, composite);
				continue;
			}

			limitReached = Stopwatch.GetTimestamp() > deadlineTimestamp;
			if (limitReached)
			{
				pending.Add(new (composite, knownComposite: true));
				continue;
			}

			#if DEVICE_GPU
				if (!TryPollardRho(gpu, composite, out ulong factor))
			#else
				if (!TryPollardRho(composite, deadlineTimestamp, out ulong factor))
			#endif
			{
				pending.Add(new (composite, knownComposite: true));
				continue;
			}

			ulong quotient = composite / factor;
			compositeStack.Push(factor);
			compositeStack.Push(quotient);
		}

		pollardRhoDeadlineReached = limitReached;
	}

#if DEVICE_GPU || DEVICE_HYBRID
	private static void CollectFactors(
		PrimeOrderCalculatorAccelerator gpu,
		ref Span<ulong> primeSlots,
		ref Span<int> exponentSlots,
		ref ulong[] factors,
		ref int[] exponents,
		ref int factorCount,
		List<PartialFactorPendingEntry> pending,
		FixedCapacityStack<ulong> compositeStack,
		long deadlineTimestamp,
		ref bool pollardRhoDeadlineReached)
	{
		CollectFactors(gpu, primeSlots, exponentSlots, ref factorCount, pending, compositeStack, deadlineTimestamp, out bool reached);
		pollardRhoDeadlineReached |= reached;
	}
#endif


#if DEVICE_HYBRID || DEVICE_GPU
    private static bool EvaluateSpecialMaxCandidates(
        PrimeOrderCalculatorAccelerator gpu,
        ReadOnlySpan<ulong> factors,
        ulong phi,
        ulong prime,
        in MontgomeryDivisorData divisorData)
    {
        int factorCount = factors.Length;
        gpu.EnsureUlongInputOutputCapacity(factorCount);

        int acceleratorIndex = gpu.AcceleratorIndex;
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
        return result != 0;
    }
#endif

#if DEVICE_CPU || DEVICE_HYBRID
    private static bool EvaluateSpecialMaxCandidates(
        Span<ulong> buffer,
        ReadOnlySpan<ulong> factors,
        ulong phi,
        ulong prime,
        in MontgomeryDivisorData divisorData)
    {
        int factorCount = factors.Length;
        for (int i = 0; i < factorCount; i++)
        {
            buffer[i] = phi / factors[i];
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
#endif

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static UInt128 ExponentLoweringWide(
		#if DEVICE_GPU
			PrimeOrderCalculatorAccelerator gpu,
		#endif
		UInt128 order,
		in UInt128 prime,
		in MontgomeryDivisorData divisorData,
		in PartialFactorResult128 factors)
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
				#if DEVICE_GPU
					if ((order % primeFactor) == UInt128.Zero)
				#else
					if (order.ReduceCycleRemainder(primeFactor) == UInt128.Zero)
				#endif
				{
					UInt128 reduced = order / primeFactor;
					#if DEVICE_GPU
						if (Pow2ModWide(gpu, reduced, prime, divisorData) == UInt128.One)
					#else
						if (Pow2ModWide(reduced, prime, divisorData) == UInt128.One)
					#endif
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
    private static UInt128 FinishStrictlyWide(
        #if DEVICE_GPU
            PrimeOrderCalculatorAccelerator gpu,
        #endif
        in UInt128 prime,
        in MontgomeryDivisorData divisorData)
    {
        UInt128 phi = prime - UInt128.One;
        FixedCapacityStack<Dictionary<UInt128, int>> uInt128IntDictionaryPool = ThreadStaticPools.UInt128IntDictionaryPool;
        Dictionary<UInt128, int> counts = uInt128IntDictionaryPool.Rent(capacity: 8);
        FactorCompletelyWide(phi, counts);
        int entryCount = counts.Count;
        PrintStats(entryCount);
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
				#if DEVICE_GPU
					if ((order % primeFactor) != UInt128.Zero)
				#else
					if (order.ReduceCycleRemainder(primeFactor) != UInt128.Zero)
				#endif
				{
					UInt128 candidate = order / primeFactor;
					#if DEVICE_GPU
						if (Pow2ModWide(gpu, candidate, prime, divisorData) == UInt128.One)
					#else
						if (Pow2ModWide(candidate, prime, divisorData) == UInt128.One)
					#endif
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

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	#if DEVICE_HYBRID || DEVICE_GPU
		private static ulong InitializeStartingOrder(PrimeOrderCalculatorAccelerator gpu, ulong prime, ulong phi, in MontgomeryDivisorData divisorData)
	#else
		private static ulong InitializeStartingOrder(ulong prime, ulong phi, in MontgomeryDivisorData divisorData)
	#endif
	{
		ulong order = phi;
		if ((prime & 7UL) == 1UL || (prime & 7UL) == 7UL)
		{
			ulong half = phi >> 1;
			#if DEVICE_HYBRID
				if (Pow2EqualsOne(gpu, half, prime, divisorData))
			#else
				if (Pow2EqualsOne(half, divisorData))
			#endif
			{
				order = half;
			}
		}

		return order;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static UInt128 InitializeStartingOrderWide(
		#if DEVICE_GPU
			PrimeOrderCalculatorAccelerator gpu,
		#endif
		in UInt128 prime,
		in UInt128 phi,
		in MontgomeryDivisorData divisorData)
    {
        UInt128 order = phi;
        UInt128 mod8 = prime & (UInt128)7UL;
        if (mod8 == UInt128.One || mod8 == (UInt128)7UL)
        {
			UInt128 half = phi >> 1;
			#if DEVICE_GPU
				if (Pow2ModWide(gpu, half, prime, divisorData) == UInt128.One)
			#else
				if (Pow2ModWide(half, prime, divisorData) == UInt128.One)
			#endif
			{
				order = half;
			}
		}

        return order;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static bool Pow2EqualsOne(ulong exponent, in MontgomeryDivisorData divisorData)
    {
        return exponent.Pow2MontgomeryModWindowedKeepMontgomeryCpu(divisorData) == divisorData.MontgomeryOne;
    }

#if DEVICE_HYBRID || DEVICE_GPU
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static bool Pow2EqualsOne(
            PrimeOrderCalculatorAccelerator gpu,
        ulong exponent,
        in MontgomeryDivisorData divisorData)
    {
        // TODO: swap to GPU pow2 when available; fallback to CPU for now.
        return exponent.Pow2MontgomeryModWindowedKeepMontgomeryCpu(divisorData) == divisorData.MontgomeryOne;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static bool Pow2EqualsOne(
            PrimeOrderCalculatorAccelerator gpu,
        ulong exponent,
        ulong prime,
        in MontgomeryDivisorData divisorData)
    {
        return exponent.Pow2MontgomeryModWindowedKeepMontgomeryCpu(divisorData) == divisorData.MontgomeryOne;
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static ulong ExponentLowering(
        ulong order,
        PartialFactorResult factors,
        in MontgomeryDivisorData divisorData)
    {
        int factorCount = factors.Count;
        int newFactorCount = factorCount + 1;
        Span<ulong> factorsSpan = new(factors.Factors, 0, newFactorCount);
        Span<int> exponentsSpan = new(factors.Exponents, 0, newFactorCount);

        if (!factors.FullyFactored && factors.Cofactor > 1UL && factors.CofactorIsPrime)
        {
            factorsSpan[factorCount] = factors.Cofactor;
            exponentsSpan[factorCount] = 1;
            factors.Count = factorCount = newFactorCount;
            factors.Cofactor = 1UL;
            factors.FullyFactored = true;
            factors.CofactorIsPrime = false;
        }

        SortFactorExponentSpans(factorsSpan, exponentsSpan, factorCount);

        ExponentRemainderStepperCpu stepper = factors.ExponentRemainderStepper;

        Span<ulong> stackCandidates = new(factors.StackCandidates);
        Span<bool> stackEvaluations = new(factors.StackEvaluations);

        for (int i = 0; i < factorCount; i++)
        {
            ulong primeFactor = factorsSpan[i];
            int exponent = exponentsSpan[i];
            ProcessExponentLoweringPrime(stackCandidates[..exponent], stackEvaluations[..exponent], ref order, primeFactor, exponent, ref stepper);
        }

        return order;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void SortFactorExponentSpans(Span<ulong> factorsSpan, Span<int> exponentsSpan, int length)
    {
        if (length <= 1)
        {
            return;
        }

        for (int i = 1; i < length; i++)
        {
            int current = i;
            int previous = current - 1;
            while (current > 0 && factorsSpan[current] < factorsSpan[previous])
            {
                (factorsSpan[current], factorsSpan[previous]) = (factorsSpan[previous], factorsSpan[current]);
                (exponentsSpan[current], exponentsSpan[previous]) = (exponentsSpan[previous], exponentsSpan[current]);
                current--;
                previous--;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ProcessExponentLoweringPrime(
        Span<ulong> candidateBuffer,
        Span<bool> evaluationBuffer,
        ref ulong order,
        ulong primeFactor,
        int exponent,
        ref ExponentRemainderStepperCpu stepper)
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

#if DEVICE_GPU
	private static PartialFactorResult PartialFactor(
		PrimeOrderCalculatorAccelerator gpu,
		ulong value,
		in MontgomeryDivisorData divisorData,
		in PrimeOrderCalculatorConfig config)
	{
		if (value <= 1UL)
		{
			return PartialFactorResult.Empty;
		}

		// stackalloc is faster than pooling

		PartialFactorResult result = PartialFactorResult.Rent(divisorData);
		ulong[] factors = result.Factors;
		Span<ulong> primeSlots = new(factors);
		int[] exponents = result.Exponents;
		Span<int> exponentSlots = new(exponents);

		// TODO: Check if this is true for GPU path
		// We don't need to worry about leftovers, because we always use indexes within the calculated counts
		// primeSlots.Clear();
		// exponentSlots.Clear();

		List<PartialFactorPendingEntry> pending = result.PendingFactors;
		pending.Clear();

		Span<int> exponentBuffer = exponentSlots[..PrimeOrderConstants.ExponentSlotLimit];
		Span<ulong> primeBuffer = primeSlots[..PrimeOrderConstants.PrimeSlotLimit];
		Span<ulong> compositeSlots = stackalloc ulong[PrimeOrderConstants.MaxCompositeSlots];
		Span<ulong> factorScratch = stackalloc ulong[PrimeOrderConstants.ScratchSlots];

		FixedCapacityStack<ulong> compositeStack = result.CompositeStack;
		compositeStack.Clear();

		ulong working = value;
		bool pollardRhoDeadlineReached = false;
		int factorCount;
		CollectSmallFactors(gpu, config, exponentBuffer, primeBuffer, ref working, ref pollardRhoDeadlineReached, out factorCount);

		Span<ulong> primeSlotsBuffer = primeSlots[..factorCount];
		Span<int> exponentSlotsBuffer = exponentSlots[..factorCount];
		primeBuffer[..factorCount].CopyTo(primeSlotsBuffer);
		exponentBuffer[..factorCount].CopyTo(exponentSlotsBuffer);

		if (working > 1UL && !pollardRhoDeadlineReached)
		{
		    compositeStack.Push(working);
		    long deadlineTimestamp = Stopwatch.GetTimestamp() + (config.MaxSeconds * Stopwatch.Frequency);
		    CollectFactors(gpu, ref primeSlotsBuffer, ref exponentSlotsBuffer, ref factors, ref exponents, ref factorCount, pending, compositeStack, deadlineTimestamp, ref pollardRhoDeadlineReached);
		}

		primeSlotsBuffer = primeSlots[..factorCount];
		exponentSlotsBuffer = exponentSlots[..factorCount];
		primeSlotsBuffer.Sort(exponentSlotsBuffer);

		result.Count = factorCount;
		exponentSlotsBuffer.CopyTo(result.Exponents);
		result.PendingFactors.Clear();
		result.PendingFactors.AddRange(pending);
		result.Cofactor = pollardRhoDeadlineReached ? working : 1UL;
		result.CofactorIsPrime = false;

		return result;
	}
#else
	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static PartialFactorResult PartialFactor(
		#if DEVICE_HYBRID
			PrimeOrderCalculatorAccelerator gpu,
		#endif
		ulong value,
		in MontgomeryDivisorData divisorData,
		in PrimeOrderCalculatorConfig config)
	{
		if (value <= 1UL)
		{
			return PartialFactorResult.Empty;
		}

		// stackalloc is faster than pooling

		PartialFactorResult result = PartialFactorResult.Rent(divisorData);
		ulong[] factors = result.Factors;
		Span<ulong> primeSlots = new(factors);
		int[] exponents = result.Exponents;
		Span<int> exponentSlots = new(exponents);

		// We don't need to worry about leftovers, because we always use indexes within the calculated counts
		// primeSlots.Clear();
		// exponentSlots.Clear();

		List<PartialFactorPendingEntry> pending = result.PendingFactors;
		pending.Clear();
		// #if DEVICE_HYBRID
		// 	bool pollardRhoDeadlineReached = false;
		// 	CollectSmallFactors(gpu, config, exponentSlots, primeSlots, ref value, ref pollardRhoDeadlineReached, out int factorCount);
		// 	if (factorCount == 0 && !pollardRhoDeadlineReached)
		// 	{
		// 		PopulateSmallPrimeFactors(
		// 			value,
		// 			config.SmallFactorLimit,
		// 			primeSlots,
		// 			exponentSlots,
		// 			out factorCount,
		// 			out value);
		// 	}
		// #elif DEVICE_CPU
			PopulateSmallPrimeFactors(
				value,
				config.SmallFactorLimit,
				primeSlots,
				exponentSlots,
				out int factorCount,
				out value);
		// #endif


		bool cofactorContainsComposite;
		if (value > 1UL)
		{
			if (config.PollardRhoMilliseconds > 0)
			{
				FixedCapacityStack<ulong> compositeStack = result.CompositeStack;
				compositeStack.Clear();
				compositeStack.Push(value);
				long deadlineTimestamp = CreateDeadlineTimestamp(config.PollardRhoMilliseconds);

				#if DEVICE_HYBRID
					CollectFactors(gpu, primeSlots, exponentSlots, ref factorCount, pending, compositeStack, deadlineTimestamp, out cofactorContainsComposite);
				#else
					CollectFactors(primeSlots, exponentSlots, ref factorCount, pending, compositeStack, deadlineTimestamp, out cofactorContainsComposite);
				#endif

				if (cofactorContainsComposite)
				{
					while (compositeStack.Count > 0)
					{
						pending.Add(new(compositeStack.Pop(), knownComposite: false));
					}
				}
			}
			else
			{
				pending.Add(new (value, knownComposite: false));
			}
		}

		ulong cofactor = 1UL;
		bool isPrime;
		int pendingCount = pending.Count;
		cofactorContainsComposite = false;

		for (int index = 0; index < pendingCount; index++)
		{
			PartialFactorPendingEntry entry = pending[index];
			// Reuse "value" parameter for testing if a number is composite to limit registry pressure.
			value = entry.Value;

			if (entry.KnownComposite)
			{
				cofactor = checked(cofactor * value);
				cofactorContainsComposite = true;
				continue;
			}

			if (!entry.HasKnownPrimality)
			{
				#if DEVICE_HYBRID
					isPrime = RunOnCpu() ? PrimeTesterByLastDigit.IsPrimeCpu(value) : HeuristicCombinedPrimeTester.IsPrimeGpu(gpu, value);
				#else
					isPrime = PrimeTesterByLastDigit.IsPrimeCpu(value);
				#endif

				entry.WithPrimality(isPrime);
				pending[index] = entry;
			}

			if (entry.IsPrime)
			{
				AddFactorToCollector(primeSlots, exponentSlots, ref factorCount, value);
			}
			// composite is never smaller on the execution path
			// else if (composite > 1UL)
			else
			{
				cofactor = checked(cofactor * value);
				cofactorContainsComposite = true;
			}
		}

		if (cofactorContainsComposite || cofactor <= 1UL)
		{
			isPrime = false;
		}
		else
		{
			#if DEVICE_HYBRID
				isPrime = RunOnCpu() ? PrimeTesterByLastDigit.IsPrimeCpu(cofactor) : HeuristicCombinedPrimeTester.IsPrimeGpu(gpu, cofactor);
			#else
				isPrime = PrimeTesterByLastDigit.IsPrimeCpu(cofactor);
			#endif
		}

		// This will never happen in production code. We'll always get at least 1 factor
		// if (factorCount == 0)
		// {
		// 	if (cofactor == value)
		// 		(...)
		// }

		Array.Sort(factors, exponents, 0, factorCount);

		result.Cofactor = cofactor;
		result.FullyFactored = cofactor == 1UL;
		result.Count = factorCount;
		result.CofactorIsPrime = isPrime;

		return result;
	}
#endif

    private static UInt128 Pow2ModWide(
		#if DEVICE_GPU
			PrimeOrderCalculatorAccelerator gpu,
		#endif
        in UInt128 exponent,
        in UInt128 modulus,
        in MontgomeryDivisorData divisorData)
    {
        if (modulus == UInt128.One)
        {
            return UInt128.Zero;
        }

		#if DEVICE_GPU
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

		#endif
        return exponent.Pow2MontgomeryModWindowed(modulus);
    }

    private static ulong RunHeuristicPipeline(
		#if DEVICE_HYBRID || DEVICE_GPU
			PrimeOrderCalculatorAccelerator gpu,
		#endif
        ulong prime,
        ulong? previousOrder,
        in PrimeOrderCalculatorConfig config,
        in MontgomeryDivisorData divisorData,
        ulong phi,
        PartialFactorResult phiFactors)
    {
		#if DEVICE_HYBRID || DEVICE_GPU
			if (phiFactors.FullyFactored && TrySpecialMax(gpu, phi, prime, phiFactors, divisorData))
		#else
			if (phiFactors.FullyFactored && TrySpecialMax(phi, prime, phiFactors, divisorData))
		#endif
        {
            return phi;
        }

		#if DEVICE_HYBRID || DEVICE_GPU
			ulong candidateOrder = InitializeStartingOrder(gpu, prime, phi, divisorData);
		#else
			ulong candidateOrder = InitializeStartingOrder(prime, phi, divisorData);
		#endif
        candidateOrder = ExponentLowering(candidateOrder, phiFactors, divisorData);

		#if DEVICE_HYBRID || DEVICE_GPU
			if (TryConfirmOrder(gpu, prime, candidateOrder, divisorData, config, out PartialFactorResult? orderFactors))
		#else
			if (TryConfirmOrder(prime, candidateOrder, divisorData, config, out PartialFactorResult? orderFactors))
		#endif
        {
            return candidateOrder;
        }

        if (config.StrictMode)
        {
            orderFactors?.Dispose();
			return CalculateByFactorization(prime, divisorData);
        }

		#if DEVICE_HYBRID || DEVICE_GPU
			if (TryHeuristicFinish(gpu, prime, candidateOrder, previousOrder, divisorData, config, orderFactors, out ulong order))
		#else
			if (TryHeuristicFinish(prime, candidateOrder, previousOrder, divisorData, config, orderFactors, out ulong order))
		#endif
        {
            return order;
        }

        return candidateOrder;
    }

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static UInt128 RunHeuristicPipelineWide(
		#if DEVICE_GPU
			PrimeOrderCalculatorAccelerator gpu,
		#endif
		in UInt128 prime,
		in UInt128? previousOrder,
		in MontgomeryDivisorData divisorData,
		in PrimeOrderCalculatorConfig config,
		in UInt128 phi,
		in PartialFactorResult128 phiFactors)
	{
		#if DEVICE_GPU
			if (phiFactors.FullyFactored && TrySpecialMaxWide(gpu, phi, prime, divisorData, phiFactors))
		#else
			if (phiFactors.FullyFactored && TrySpecialMaxWide(phi, prime, divisorData, phiFactors))
		#endif
		{
			return phi;
		}

		#if DEVICE_GPU
			UInt128 candidateOrder = InitializeStartingOrderWide(gpu, prime, phi, divisorData);
			candidateOrder = ExponentLoweringWide(gpu, candidateOrder, prime, divisorData, phiFactors);

			if (TryConfirmOrderWide(gpu, prime, candidateOrder, divisorData, config))
		#else
			UInt128 candidateOrder = InitializeStartingOrderWide(prime, phi, divisorData);
			candidateOrder = ExponentLoweringWide(candidateOrder, prime, divisorData, phiFactors);

			if (TryConfirmOrderWide(prime, candidateOrder, divisorData, config))
		#endif
		{
			return candidateOrder;
		}

		if (config.StrictMode)
		{
			#if DEVICE_GPU
				return FinishStrictlyWide(gpu, prime, divisorData);
			#else
				return FinishStrictlyWide(prime, divisorData);
			#endif
		}

		#if DEVICE_GPU
			if (TryHeuristicFinishWide(gpu, prime, candidateOrder, previousOrder, divisorData, config, out UInt128 order))
		#else
			if (TryHeuristicFinishWide(prime, candidateOrder, previousOrder, divisorData, config, out UInt128 order))
		#endif
		{
			return order;
		}

		return candidateOrder;
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
        int rightPrimary = ComputePrimary(rightGroup, rightIsGe, previousGroup);

        int compare = ComputePrimary(leftGroup, leftIsGe, previousGroup);
        compare = compare.CompareTo(rightPrimary);
        if (compare != 0)
        {
            return compare;
        }

        ulong leftDistance = hasPrevious ? (left > previous ? left - previous : previous - left) : left;
        ulong rightDistance = hasPrevious ? (right > previous ? right - previous : previous - right) : right;

        bool leftGroupDescending = leftGroup == 3;
        if (leftDistance == rightDistance)
        {
            return leftGroupDescending ? (left > right ? -1 : (left < right ? 1 : 0)) : left.CompareTo(right);
        }

        if (leftGroupDescending)
        {
            return leftDistance > rightDistance ? -1 : 1;
        }

        return leftDistance.CompareTo(rightDistance);
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

    private static void SortCandidatesWide(UInt128 prime, in UInt128? previousOrder, List<UInt128> candidates)
    {
        UInt128 previous = previousOrder ?? UInt128.Zero;
        int previousGroup = previousOrder.HasValue ? GetGroupWide(previousOrder.Value, prime) : 1;
        bool hasPrevious = previousOrder.HasValue;

        candidates.Sort((x, y) =>
        {
            CandidateKey128 keyX = BuildKeyWide(x, prime, previous, previousGroup, hasPrevious);
            CandidateKey128 keyY = BuildKeyWide(y, prime, previous, previousGroup, hasPrevious);

            int primary = keyX.Primary.CompareTo(keyY.Primary);
            if (primary != 0)
            {
                return primary;
            }

            int secondary = CompareComponents(keyX.SecondaryDescending, keyX.Secondary, keyY.SecondaryDescending, keyY.Secondary);
            if (secondary != 0)
            {
                return secondary;
            }

            return CompareComponents(keyX.TertiaryDescending, keyX.Tertiary, keyY.TertiaryDescending, keyY.Tertiary);
        });
    }

    private static bool TryConfirmCandidate(
		#if DEVICE_HYBRID || DEVICE_GPU
			PrimeOrderCalculatorAccelerator gpu,
		#endif
        ulong prime,
        ulong candidate,
        in MontgomeryDivisorData divisorData,
        in PrimeOrderCalculatorConfig config,
        ref int powUsed,
        int powBudget)
    {
		#if DEVICE_HYBRID || DEVICE_GPU
			PartialFactorResult factorization = PartialFactor(gpu, candidate, divisorData, config);
		#else
			PartialFactorResult factorization = PartialFactor(candidate, divisorData, config);
		#endif

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

    private static bool TryConfirmCandidateWide(
		#if DEVICE_GPU
			PrimeOrderCalculatorAccelerator gpu,
		#endif
        in UInt128 prime,
        in UInt128 candidate,
        in PrimeOrderCalculatorConfig config,
        ref int powUsed,
        int powBudget,
        in MontgomeryDivisorData divisorData)
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
				#if DEVICE_GPU
					if ((reduced % primeFactor) != UInt128.Zero)
				#else
					if (reduced.ReduceCycleRemainder(primeFactor) != UInt128.Zero)
				#endif
                {
                    break;
                }

                reduced /= primeFactor;
                if (powUsed >= powBudget && powBudget > 0)
                {
                    return false;
                }

                powUsed++;
				#if DEVICE_GPU
					if (Pow2ModWide(gpu, reduced, prime, divisorData) == UInt128.One)
				#else
					if (Pow2ModWide(reduced, prime, divisorData) == UInt128.One)
				#endif
                {
                    return false;
                }
            }
        }

        return true;
    }

    private static bool TryConfirmOrder(
		#if DEVICE_HYBRID || DEVICE_GPU
			PrimeOrderCalculatorAccelerator gpu,
		#endif
        ulong prime,
        ulong candidateOrder,
        in MontgomeryDivisorData divisorData,
        in PrimeOrderCalculatorConfig config,
        out PartialFactorResult? orderFactors)
    {
        orderFactors = null;
		ArgumentOutOfRangeException.ThrowIfEqual(candidateOrder, 0UL);

		#if DEVICE_HYBRID
			if (!Pow2EqualsOne(gpu, candidateOrder, prime, divisorData))
		#else
			if (!Pow2EqualsOne(candidateOrder, divisorData))
		#endif
        {
            return false;
        }

		#if DEVICE_HYBRID || DEVICE_GPU
			PartialFactorResult factorization = PartialFactor(gpu, candidateOrder, divisorData, config);
		#else
			PartialFactorResult factorization = PartialFactor(candidateOrder, divisorData, config);
		#endif
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

        if (!ValidateOrderAgainstFactors(prime, candidateOrder, divisorData, factorization))
        {
            orderFactors = factorization;
            return false;
        }

        factorization.Dispose();
        return true;
    }

    private static bool TryConfirmOrderWide(
		#if DEVICE_GPU
			PrimeOrderCalculatorAccelerator gpu,
		#endif
        in UInt128 prime,
        in UInt128 order,
        in MontgomeryDivisorData divisorData,
        in PrimeOrderCalculatorConfig config)
    {
        if (order == UInt128.Zero)
        {
            return false;
        }

		#if DEVICE_GPU
			if (Pow2ModWide(gpu, order, prime, divisorData) != UInt128.One)
		#else
			if (Pow2ModWide(order, prime, divisorData) != UInt128.One)
		#endif
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
				#if DEVICE_GPU
					if (Pow2ModWide(gpu, reduced, prime, divisorData) == UInt128.One)
				#else
					if (Pow2ModWide(reduced, prime, divisorData) == UInt128.One)
				#endif
                {
                    return false;
                }
            }
        }

        return true;
    }

    private static bool TryHeuristicFinish(
		#if DEVICE_HYBRID || DEVICE_GPU
			PrimeOrderCalculatorAccelerator gpu,
		#endif
        ulong prime,
        ulong candidateOrder,
        ulong? previousOrder,
        in MontgomeryDivisorData divisorData,
        in PrimeOrderCalculatorConfig config,
        PartialFactorResult? orderFactors,
        out ulong order)
    {
        order = 0UL;
        if (candidateOrder <= 1UL)
        {
            orderFactors?.Dispose();
            return false;
        }

		#if DEVICE_GPU || DEVICE_HYBRID
			PartialFactorResult localOrderFactors = orderFactors ?? PartialFactor(gpu, candidateOrder, divisorData, config);
		#else
			PartialFactorResult localOrderFactors = orderFactors ?? PartialFactor(candidateOrder, divisorData, config);
		#endif
        try
        {
            if (localOrderFactors.Factors is null)
            {
                return false;
            }

            if (!localOrderFactors.FullyFactored)
            {
                if (localOrderFactors.Cofactor <= 1UL || !localOrderFactors.CofactorIsPrime)
                {
                    return false;
                }

                localOrderFactors.WithAdditionalPrime(localOrderFactors.Cofactor);
            }

            int capacity = config.MaxPowChecks <= 0 ? 64 : config.MaxPowChecksCapacity;
            List<ulong> candidates = ThreadStaticPools.RentUlongList(capacity);
            candidates.Clear();
            ulong[] factorArray = localOrderFactors.Factors!;
            int[] exponentsArray = localOrderFactors.Exponents!;

            BuildCandidates(candidateOrder, factorArray, exponentsArray, localOrderFactors.Count, candidates, capacity);
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
			#if DEVICE_GPU || DEVICE_HYBRID
                bool allowGpuBatch = true;
                Span<ulong> stackGpuRemainders = stackalloc ulong[PerfectNumberConstants.DefaultSmallPrimeFactorSlotCount];
                ArrayPool<ulong> pool = ThreadStaticPools.UlongPool;
			#endif

            while (index < candidateCount && powUsed < powBudget)
            {
                int remaining = candidateCount - index;
                int budgetRemaining = powBudget - powUsed;
                int batchSize = budgetRemaining <= 0 ? remaining : Math.Min(remaining, budgetRemaining);

				#if DEVICE_GPU || DEVICE_HYBRID
                    batchSize = Math.Min(batchSize, PrimeOrderConstants.MaxGpuBatchSize);
				#endif

                if (batchSize <= 0)
                {
                    break;
                }

                ReadOnlySpan<ulong> batch = candidateSpan.Slice(index, batchSize);

				#if DEVICE_GPU || DEVICE_HYBRID
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
				#endif

                for (int i = 0; i < batchSize && powUsed < powBudget; i++)
                {
                    ulong candidate = batch[i];

					#if DEVICE_GPU || DEVICE_HYBRID
                        bool equalsOne;
                        if (gpuSuccess)
                        {
                            powUsed++;
                            ulong remainderValue = gpuStackRemainders ? stackGpuRemainders[i] : pooledGpuRemainders[i];
                            equalsOne = remainderValue == 1UL;
                        }
                        else
					#endif
                        {
                            powUsed++;
                            if (!powStepperInitialized)
                            {
                                powStepperInitialized = true;
                                if (!powStepper.InitializeCpuIsUnity(candidate))
                                {
                                    continue;
                                }
                            }
                            else if (!powStepper.ComputeNextIsUnity(candidate))
                            {
                                continue;
                            }
							#if DEVICE_GPU || DEVICE_HYBRID
                                equalsOne = true;
							#endif
                            }

                    if (powUsed > powBudget && powBudget > 0)
                    {
                        break;
                    }

					#if DEVICE_GPU || DEVICE_HYBRID
                        if (!TryConfirmCandidate(gpu, prime, candidate, divisorData, config, ref powUsed, powBudget))
					#else
                        if (!TryConfirmCandidate(prime, candidate, divisorData, config, ref powUsed, powBudget))
					#endif
                    {
                        continue;
                    }

					#if DEVICE_GPU || DEVICE_HYBRID
                        if (gpuPool is not null)
                        {
                            pool.Return(gpuPool, clearArray: false);
                        }
					#endif

                    candidates.Clear();
                    ThreadStaticPools.ReturnUlongList(candidates);
                    ThreadStaticPools.ReturnExponentStepperCpu(powStepper);
                    order = candidate;
                    return true;
                }

				#if DEVICE_GPU || DEVICE_HYBRID
                    if (gpuPool is not null)
                    {
                        pool.Return(gpuPool, clearArray: false);
                    }
				#endif

                index += batchSize;
            }

            candidates.Clear();
            ThreadStaticPools.ReturnUlongList(candidates);
            ThreadStaticPools.ReturnExponentStepperCpu(powStepper);

            return false;
        }
        finally
        {
            localOrderFactors.Dispose();
        }
    }

    private static bool TryHeuristicFinishWide(
		#if DEVICE_GPU
			PrimeOrderCalculatorAccelerator gpu,
		#endif
        in UInt128 prime,
        in UInt128 order,
        in UInt128? previousOrder,
        in MontgomeryDivisorData divisorData,
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

			#if DEVICE_GPU
				if (Pow2ModWide(gpu, candidate, prime, divisorData) != UInt128.One)
			#else
				if (Pow2ModWide(candidate, prime, divisorData) != UInt128.One)
			#endif
            {
                continue;
            }

			#if DEVICE_GPU
				if (!TryConfirmCandidateWide(gpu, prime, candidate, config, ref powUsed, powBudget, divisorData))
			#else
				if (!TryConfirmCandidateWide(prime, candidate, config, ref powUsed, powBudget, divisorData))
			#endif
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

#if DEVICE_GPU
    private static bool TryPollardRho(
        PrimeOrderCalculatorAccelerator gpu,
        ulong n,
        out ulong factor)
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
#else
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static bool TryPollardRho(
        ulong n,
        long deadlineTimestamp,
        out ulong factor)
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

                x = AdvancePolynomial(x, c, n);
                y = AdvancePolynomialTwice(y, c, n);
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
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static long CreateDeadlineTimestamp(int milliseconds)
    {
        if (milliseconds <= 0)
        {
            return long.MaxValue;
        }

        long ticksPerMillisecond = Stopwatch.Frequency / 1000;
        return Stopwatch.GetTimestamp() + (ticksPerMillisecond * milliseconds);
    }

#if DEVICE_HYBRID
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
#endif

    private static bool TrySpecialMax(
        #if DEVICE_HYBRID || DEVICE_GPU
            PrimeOrderCalculatorAccelerator gpu,
        #endif
        ulong phi,
        ulong prime,
        PartialFactorResult factors,
        in MontgomeryDivisorData divisorData)
    {
        int length = factors.Count;
        ReadOnlySpan<ulong> factorSpan = new(factors.Factors, 0, length);
        Span<ulong> stackBuffer = stackalloc ulong[length];

		#if DEVICE_HYBRID
			if (length <= 8)
			{
				return EvaluateSpecialMaxCandidates(stackBuffer, factorSpan, phi, prime, divisorData);
			}

			return EvaluateSpecialMaxCandidates(gpu, factorSpan, phi, prime, divisorData);
		#elif DEVICE_GPU
			return EvaluateSpecialMaxCandidates(gpu, factorSpan, phi, prime, divisorData);
		#else
			return EvaluateSpecialMaxCandidates(stackBuffer, factorSpan, phi, prime, divisorData);
		#endif
    }

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static bool TrySpecialMaxWide(
		#if DEVICE_GPU
			PrimeOrderCalculatorAccelerator gpu,
		#endif
		in UInt128 phi,
		in UInt128 prime,
		in MontgomeryDivisorData divisorData,
		in PartialFactorResult128 factors)
	{
		ReadOnlySpan<FactorEntry128> factorSpan = factors.Factors;
		int length = factors.Count;
		for (int i = 0; i < length; i++)
		{
			UInt128 factor = factorSpan[i].Value;
			UInt128 reduced = phi / factor;
			#if DEVICE_GPU
				if (Pow2ModWide(gpu, reduced, prime, divisorData) == UInt128.One)
			#else
				if (Pow2ModWide(reduced, prime, divisorData) == UInt128.One)
			#endif
			{
				return false;
			}
		}

		return true;
	}

#if DEVICE_GPU //|| DEVICE_HYBRID
    private const int GpuSmallPrimeFactorSlots = 64;

	// GPU specific. It doesn't make sense to consolidate into one method.
    private static bool TryPopulateSmallPrimeFactors(
        PrimeOrderCalculatorAccelerator gpu,
        ulong value,
        uint limit,
        Dictionary<ulong, int> counts,
        out int factorCount,
        out ulong remaining)
    {
        var primeBufferArray = ThreadStaticPools.UlongPool.Rent(GpuSmallPrimeFactorSlots);
        var exponentBufferArray = ThreadStaticPools.IntPool.Rent(GpuSmallPrimeFactorSlots);
        Span<ulong> primeBuffer = primeBufferArray.AsSpan(0, GpuSmallPrimeFactorSlots);
        Span<int> exponentBuffer = exponentBufferArray.AsSpan(0, GpuSmallPrimeFactorSlots);
        remaining = value;

        gpu.EnsureSmallPrimeFactorSlotsCapacity(GpuSmallPrimeFactorSlots);
        int acceleratorIndex = gpu.AcceleratorIndex;

        var kernelLauncher = gpu.SmallPrimeFactorKernelLauncher;
        ArrayView1D<int, Stride1D.Dense> smallPrimeFactorCountSlotView = gpu.OutputIntView2;
        ArrayView1D<ulong, Stride1D.Dense> smallPrimeFactorRemainingSlotView = gpu.InputView;
        ArrayView1D<ulong, Stride1D.Dense> smallPrimeFactorPrimeSlotsView = gpu.OutputUlongView;
        ArrayView1D<int, Stride1D.Dense> smallPrimeFactorExponentSlotsView = gpu.OutputIntView;
        ArrayView1D<ulong, Stride1D.Dense> smallPrimeFactorsSquaresView = gpu.SmallPrimeFactorSquares;
        ArrayView1D<uint, Stride1D.Dense> smallPrimeFactorsPrimesView = gpu.SmallPrimeFactorPrimes;

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

        for (int i = 0; i < factorCount; i++)
        {
            ulong primeValue = primeBuffer[i];
            int exponent = exponentBuffer[i];
            counts.Add(primeValue, exponent);
        }

        ThreadStaticPools.UlongPool.Return(primeBufferArray, clearArray: false);
        ThreadStaticPools.IntPool.Return(exponentBufferArray, clearArray: false);
        return true;
    }

	// GPU specific. It doesn't make sense to consolidate into one method.
    private static void CollectSmallFactors(
        PrimeOrderCalculatorAccelerator gpu,
        in PrimeOrderCalculatorConfig config,
        Span<int> exponentBuffer,
        Span<ulong> primeBuffer,
        ref ulong working,
        ref bool pollardRhoDeadlineReached,
        out int factorCount)
    {
        Dictionary<ulong, int> counts = ThreadStaticPools.UlongIntDictionaryPool.Rent(capacity: 64);
        counts.Clear();

        bool populated = TryPopulateSmallPrimeFactors(gpu, working, (uint)PrimeNumberConstants.GpuSmallPrimeBound, counts, out factorCount, out ulong remaining);
        pollardRhoDeadlineReached = false;
        if (!populated)
        {
            factorCount = 0;
            working = remaining;
            ThreadStaticPools.UlongIntDictionaryPool.Return(counts);
            return;
        }

        if (remaining > 1UL)
        {
            if (counts.Count < GpuSmallPrimeFactorSlots)
            {
                counts.Add(remaining, 1);
                working = 1UL;
            }
            else
            {
                pollardRhoDeadlineReached = true;
                working = remaining;
            }
        }

        int collectedCount = 0;
        foreach (KeyValuePair<ulong, int> pair in counts)
        {
            if (collectedCount >= PrimeOrderConstants.ExponentSlotLimit)
            {
                pollardRhoDeadlineReached = true;
                break;
            }

            primeBuffer[collectedCount] = pair.Key;
            exponentBuffer[collectedCount] = pair.Value;
            collectedCount++;
        }

        factorCount = collectedCount;
        ThreadStaticPools.UlongIntDictionaryPool.Return(counts);
    }

#endif

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

        Span<ulong> stackBuffer = stackalloc ulong[StackExponentCapacity];

        ExponentRemainderStepperCpu stepper = ThreadStaticPools.RentExponentStepperCpu(divisorData);
        bool violates = false;

        for (int i = 0; i < length; i++)
        {
            ulong primeFactor = factorsSpan[i];
            int exponent = exponentsSpan[i];

            if (ValidateOrderForFactor(stackBuffer[..exponent], primeFactor, exponent, order, ref stepper))
            {
                violates = true;
                break;
            }
        }

        ThreadStaticPools.ReturnExponentStepperCpu(stepper);

        return !violates;
    }

    private static bool ValidateOrderForFactor(
        Span<ulong> buffer,
        ulong primeFactor,
        int exponent,
        ulong order,
        ref ExponentRemainderStepperCpu stepper)
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
}
