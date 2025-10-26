using System.Buffers;
using System.Diagnostics;
using System.Numerics;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

internal static partial class PrimeOrderCalculator
{
        [ThreadStatic]
        private static bool s_pow2ModeInitialized;

        [ThreadStatic]
        private static bool s_allowGpuPow2;

        [ThreadStatic]
        private static PrimeOrderHeuristicDevice s_deviceMode;

        [ThreadStatic]
        private static bool s_debugLoggingEnabled;

        private static bool IsGpuHeuristicDevice => s_pow2ModeInitialized && s_deviceMode == PrimeOrderHeuristicDevice.Gpu;

        private readonly struct Pow2ModeScope
        {
                private readonly bool _previousInitialized;
                private readonly bool _previousAllow;
                private readonly PrimeOrderHeuristicDevice _previousDevice;

                public Pow2ModeScope(PrimeOrderHeuristicDevice device)
                {
                        _previousInitialized = s_pow2ModeInitialized;
                        _previousAllow = s_allowGpuPow2;
                        _previousDevice = s_deviceMode;
                        s_pow2ModeInitialized = true;
                        s_allowGpuPow2 = device == PrimeOrderHeuristicDevice.Gpu;
                        s_deviceMode = device;
                }

                public void Dispose()
                {
                        s_allowGpuPow2 = _previousAllow;
                        s_pow2ModeInitialized = _previousInitialized;
                        s_deviceMode = _previousDevice;
                }
        }

        private readonly struct DebugLoggingScope
        {
                private readonly bool _previous;

                public DebugLoggingScope(bool enabled)
                {
                        _previous = s_debugLoggingEnabled;
                        s_debugLoggingEnabled = enabled;
                }

                public void Dispose()
                {
                        s_debugLoggingEnabled = _previous;
                }
        }

        private static Pow2ModeScope UsePow2Mode(PrimeOrderHeuristicDevice device) => new(device);

        private static DebugLoggingScope UseDebugLogging(bool enabled) => new(enabled);

        [Conditional("DEBUG")]
        private static void DebugLog(string message)
        {
                if (s_debugLoggingEnabled)
                {
                        Console.WriteLine(message);
                }
        }

        [Conditional("DEBUG")]
        private static void DebugLog(Func<string> messageFactory)
        {
                if (s_debugLoggingEnabled)
                {
                        Console.WriteLine(messageFactory());
                }
        }
    private static UInt128 CalculateWideInternal(in UInt128 prime, in UInt128? previousOrder, in MontgomeryDivisorData divisorData, in PrimeOrderSearchConfig config)
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
        var debugScope = UseDebugLogging(!IsGpuHeuristicDevice);
        PartialFactorResult128 phiFactors = PartialFactorWide(phi, config);
        UInt128 result;
        if (phiFactors.Factors is null)
        {
            result = FinishStrictlyWide(prime, divisorData);
        }
        else
        {
            result = RunHeuristicPipelineWide(prime, previousOrder, divisorData, config, phi, phiFactors);
        }

        debugScope.Dispose();
        return result;
    }

    private static UInt128 RunHeuristicPipelineWide(
        in UInt128 prime,
        in UInt128? previousOrder,
        in MontgomeryDivisorData divisorData,
        in PrimeOrderSearchConfig config,
        in UInt128 phi,
        in PartialFactorResult128 phiFactors)
    {
        if (phiFactors.FullyFactored && TrySpecialMaxWide(phi, prime, divisorData, phiFactors))
        {
            return phi;
        }

        UInt128 candidateOrder = InitializeStartingOrderWide(prime, phi, divisorData);
        candidateOrder = ExponentLoweringWide(candidateOrder, prime, divisorData, phiFactors);

        if (TryConfirmOrderWide(prime, candidateOrder, divisorData, config))
        {
            return candidateOrder;
        }

        if (config.Mode == PrimeOrderMode.Strict)
        {
            return FinishStrictlyWide(prime, divisorData);
        }

        if (TryHeuristicFinishWide(prime, candidateOrder, previousOrder, divisorData, config, out UInt128 order))
        {
            return order;
        }

        return candidateOrder;
    }

    private static UInt128 FinishStrictlyWide(in UInt128 prime, in MontgomeryDivisorData divisorData)
    {
        return CalculateByFactorizationWide(prime, divisorData);
    }

    private static bool TrySpecialMaxWide(in UInt128 phi, in UInt128 prime, in MontgomeryDivisorData divisorData, in PartialFactorResult128 factors)
    {
        ReadOnlySpan<FactorEntry128> factorSpan = factors.Factors;
        int length = factors.Count;
        for (int i = 0; i < length; i++)
        {
            UInt128 factor = factorSpan[i].Value;
            UInt128 reduced = phi / factor;
            if (Pow2ModWide(reduced, prime, divisorData) == UInt128.One)
            {
                return false;
            }
        }

        return true;
    }

    private static UInt128 InitializeStartingOrderWide(in UInt128 prime, in UInt128 phi, in MontgomeryDivisorData divisorData)
    {
        UInt128 order = phi;
        UInt128 mod8 = prime & (UInt128)7UL;
        if (mod8 == UInt128.One || mod8 == (UInt128)7UL)
        {
            UInt128 half = phi >> 1;
            if (Pow2ModWide(half, prime, divisorData) == UInt128.One)
            {
                order = half;
            }
        }

        return order;
    }

    private static UInt128 ExponentLoweringWide(UInt128 order, in UInt128 prime, in MontgomeryDivisorData divisorData, in PartialFactorResult128 factors)
    {
		ArrayPool<FactorEntry128> pool = ThreadStaticPools.FactorEntry128Pool;
		ReadOnlySpan<FactorEntry128> factorSpan = factors.Factors;
		int length = factors.Count;
		// TODO: This should never trigger from production code - check
		// if (length == 0)
		// {
		//     return order;
		// }
		FactorEntry128[]? tempArray = pool.Rent(length + 1);
        try
        {

            Span<FactorEntry128> buffer = tempArray.AsSpan(0, length);
            factorSpan.CopyTo(buffer);

            if (!factors.FullyFactored && factors.Cofactor > UInt128.One && IsPrimeWide(factors.Cofactor))
            {
                buffer[length] = new FactorEntry128(factors.Cofactor, 1);
                length++;
            }

            buffer.Slice(0, length).Sort(static (a, b) => a.Value.CompareTo(b.Value));

            for (int i = 0; i < length; i++)
            {
                UInt128 primeFactor = buffer[i].Value;
                int exponent = buffer[i].Exponent;
                for (int iteration = 0; iteration < exponent; iteration++)
                {
                    if ((order % primeFactor) == UInt128.Zero)
                    {
                        UInt128 reduced = order / primeFactor;
                        if (Pow2ModWide(reduced, prime, divisorData) == UInt128.One)
                        {
                            order = reduced;
                            continue;
                        }
                    }

                    break;
                }
            }

            return order;
        }
        finally
        {
			pool.Return(tempArray, clearArray: false);
        }
    }

    private static bool TryConfirmOrderWide(in UInt128 prime, in UInt128 order, in MontgomeryDivisorData divisorData, in PrimeOrderSearchConfig config)
    {
        if (order == UInt128.Zero)
        {
            return false;
        }

        if (Pow2ModWide(order, prime, divisorData) != UInt128.One)
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
                if ((reduced % primeFactor) != UInt128.Zero)
                {
                    break;
                }

                reduced /= primeFactor;
                if (Pow2ModWide(reduced, prime, divisorData) == UInt128.One)
                {
                    return false;
                }
            }
        }

        return true;
    }

    private static bool TryHeuristicFinishWide(
        in UInt128 prime,
        in UInt128 order,
        in UInt128? previousOrder,
        in MontgomeryDivisorData divisorData,
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

            if (Pow2ModWide(candidate, prime, divisorData) != UInt128.One)
            {
                continue;
            }

            if (!TryConfirmCandidateWide(prime, candidate, config, ref powUsed, powBudget, divisorData))
            {
                continue;
            }

            result = candidate;
            return true;
        }

        return false;
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
        UInt128 threshold1 = prime >> 3;
        if (value <= threshold1)
        {
            return 1;
        }

        UInt128 threshold2 = prime >> 2;
        if (value <= threshold2)
        {
            return 2;
        }

        UInt128 threshold3 = (UInt128)(((BigInteger)prime * 3) >> 3);
        if (value <= threshold3)
        {
            return 3;
        }

        return 0;
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
        UInt128 primeFactor = factor.Value;
        UInt128 contribution = UInt128.One;
        for (int exponent = 0; exponent <= factor.Exponent; exponent++)
        {
            UInt128 nextDivisor = divisorProduct * contribution;
            if (nextDivisor > order)
            {
                break;
            }

            BuildCandidatesRecursiveWide(order, factors, index + 1, nextDivisor, candidates, limit);
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

    private static bool TryConfirmCandidateWide(in UInt128 prime, in UInt128 candidate, in PrimeOrderSearchConfig config, ref int powUsed, int powBudget, in MontgomeryDivisorData divisorData)
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
                if (Pow2ModWide(reduced, prime, divisorData) == UInt128.One)
                {
                    return false;
                }
            }
        }

        return true;
    }

    private static PartialFactorResult128 PartialFactorWide(in UInt128 value, in PrimeOrderSearchConfig config)
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

    private static UInt128 CalculateByFactorizationWide(in UInt128 prime, in MontgomeryDivisorData divisorData)
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
                    if (Pow2ModWide(candidate, prime, divisorData) == UInt128.One)
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
            return PrimeTester.IsPrimeCpu((ulong)value, CancellationToken.None);
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

    private static UInt128 Pow2ModWide(in UInt128 exponent, in UInt128 modulus, in MontgomeryDivisorData divisorData)
    {
        if (modulus == UInt128.One)
        {
            return UInt128.Zero;
        }

        if (IsGpuPow2Allowed)
        {
            if (modulus <= (UInt128)ulong.MaxValue && exponent <= (UInt128)ulong.MaxValue)
            {
                ulong prime64 = (ulong)modulus;
                ulong exponent64 = (ulong)exponent;
                GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2Mod(exponent64, prime64, out ulong remainder, divisorData);
                if (status == GpuPow2ModStatus.Success)
                {
                    return remainder;
                }
            }

            GpuPow2ModStatus wideStatus = PrimeOrderGpuHeuristics.TryPow2Mod(exponent, modulus, out UInt128 wideRemainder);
            if (wideStatus == GpuPow2ModStatus.Success)
            {
                return wideRemainder;
            }
        }

        return exponent.Pow2MontgomeryModWindowed(modulus);
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
}
