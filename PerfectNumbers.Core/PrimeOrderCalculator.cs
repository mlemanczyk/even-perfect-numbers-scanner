using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Threading;

namespace PerfectNumbers.Core;

internal static partial class PrimeOrderCalculator
{
    internal enum PrimeOrderStatus
    {
        Found,
        HeuristicUnresolved,
    }

    internal readonly struct PrimeOrderResult
    {
        public PrimeOrderResult(PrimeOrderStatus status, ulong order)
        {
            Status = status;
            Order = order;
        }

        public PrimeOrderStatus Status { get; }

        public ulong Order { get; }
    }

    internal enum PrimeOrderMode
    {
        Heuristic,
        Strict,
    }

    internal readonly struct PrimeOrderSearchConfig
    {
        public PrimeOrderSearchConfig(uint smallFactorLimit, int pollardRhoMilliseconds, int maxPowChecks, PrimeOrderMode mode)
        {
            SmallFactorLimit = smallFactorLimit;
            PollardRhoMilliseconds = pollardRhoMilliseconds;
            MaxPowChecks = maxPowChecks;
            Mode = mode;
        }

        public uint SmallFactorLimit { get; }

        public int PollardRhoMilliseconds { get; }

        public int MaxPowChecks { get; }

        public PrimeOrderMode Mode { get; }

        public static PrimeOrderSearchConfig HeuristicDefault => new(100_000, 128, 24, PrimeOrderMode.Heuristic);

        public static PrimeOrderSearchConfig StrictDefault => new(1_000_000, 0, 0, PrimeOrderMode.Strict);
    }

    public static PrimeOrderResult Calculate(ulong prime, ulong? previousOrder, PrimeOrderSearchConfig config)
    {
        if (prime <= 3UL)
        {
            return new PrimeOrderResult(PrimeOrderStatus.Found, prime == 3UL ? 2UL : 1UL);
        }

        ulong phi = prime - 1UL;
        MontgomeryDivisorData divisorData = MontgomeryDivisorDataCache.Get(prime);

#if DEBUG
        Console.WriteLine("Partial factoring φ(p)");
#endif
        PartialFactorResult phiFactors = PartialFactor(phi, config);

        if (phiFactors.Factors is null)
        {
#if DEBUG
            Console.WriteLine("No factors found");
#endif
            return FinishStrictly(prime, config.Mode);
        }

#if DEBUG
        Console.WriteLine("Trying special max check");
#endif
        if (phiFactors.FullyFactored && TrySpecialMax(phi, prime, phiFactors, divisorData))
        {
            return new PrimeOrderResult(PrimeOrderStatus.Found, phi);
        }

#if DEBUG
        Console.WriteLine("Initializing starting order");
#endif
        ulong candidateOrder = InitializeStartingOrder(prime, phi, divisorData);
        candidateOrder = ExponentLowering(candidateOrder, prime, phiFactors, divisorData);

#if DEBUG
        Console.WriteLine("Trying to confirm order");
#endif
        if (TryConfirmOrder(prime, candidateOrder, divisorData, config))
        {
            return new PrimeOrderResult(PrimeOrderStatus.Found, candidateOrder);
        }

        if (config.Mode == PrimeOrderMode.Strict)
        {
            return FinishStrictly(prime, PrimeOrderMode.Strict);
        }

        if (TryHeuristicFinish(prime, candidateOrder, previousOrder, divisorData, config, phiFactors, out ulong order))
        {
            return new PrimeOrderResult(PrimeOrderStatus.Found, order);
        }

#if DEBUG
        Console.WriteLine("Heuristic unresolved, finishing strictly");
#endif
        return FinishStrictly(prime, config.Mode);
    }

    private static PrimeOrderResult FinishStrictly(ulong prime, PrimeOrderMode mode)
    {
        ulong strictOrder = CalculateByDoubling(prime);
        return new PrimeOrderResult(mode == PrimeOrderMode.Strict ? PrimeOrderStatus.Found : PrimeOrderStatus.HeuristicUnresolved, strictOrder);
    }

    private static bool TrySpecialMax(ulong phi, ulong prime, PartialFactorResult factors, in MontgomeryDivisorData divisorData)
    {
        ReadOnlySpan<FactorEntry> factorSpan = factors.Factors;
        int length = factors.Count;
        for (int i = 0; i < length; i++)
        {
            ulong factor = factorSpan[i].Value;
            ulong reduced = phi / factor;
            if (reduced.Pow2MontgomeryModWindowed(divisorData, keepMontgomery: false) == 1UL)
            {
                return false;
            }
        }

        return true;
    }

    private static ulong InitializeStartingOrder(ulong prime, ulong phi, in MontgomeryDivisorData divisorData)
    {
        ulong order = phi;
        if ((prime & 7UL) == 1UL || (prime & 7UL) == 7UL)
        {
            ulong half = phi >> 1;
            if (half.Pow2MontgomeryModWindowed(divisorData, keepMontgomery: false) == 1UL)
            {
                order = half;
            }
        }

        return order;
    }

    private static ulong ExponentLowering(ulong order, ulong prime, PartialFactorResult factors, in MontgomeryDivisorData divisorData)
    {
        FactorEntry[]? tempArray = null;
        try
        {
            ReadOnlySpan<FactorEntry> factorSpan = factors.Factors;
            int length = factors.Count;
            tempArray = ArrayPool<FactorEntry>.Shared.Rent(length + 1);
            Span<FactorEntry> buffer = tempArray;
            factorSpan.CopyTo(buffer);

            bool isPrime = Open.Numeric.Primes.Prime.Numbers.IsPrime(factors.Cofactor);
            // bool isPrime = PrimeTester.IsPrimeInternal(factors.Cofactor, CancellationToken.None);
            if (!factors.FullyFactored && factors.Cofactor > 1UL && isPrime)
            {
                buffer[length] = new FactorEntry(factors.Cofactor, 1);
                length++;
            }

            buffer.Slice(0, length).Sort(static (a, b) => a.Value.CompareTo(b.Value));

            for (int i = 0; i < length; i++)
            {
                ulong primeFactor = buffer[i].Value;
                int exponent = buffer[i].Exponent;
                for (int iteration = 0; iteration < exponent; iteration++)
                {
                    if ((order % primeFactor) != 0UL)
                    {
                        break;
                    }

                    ulong reduced = order / primeFactor;
                    if (reduced.Pow2MontgomeryModWindowed(divisorData, keepMontgomery: false) == 1UL)
                    {
                        order = reduced;
                        continue;
                    }

                    break;
                }
            }

            return order;
        }
        finally
        {
            if (tempArray is not null)
            {
                ArrayPool<FactorEntry>.Shared.Return(tempArray, clearArray: false);
            }
        }
    }

    private static bool TryConfirmOrder(ulong prime, ulong order, in MontgomeryDivisorData divisorData, PrimeOrderSearchConfig config)
    {
        if (order == 0UL)
        {
            return false;
        }

        // Calculating `a^order ≡ 1 (mod p)` is a prerequisite for `order` being the actual order of 2 modulo `p`.
#if DEBUG
        Console.WriteLine("Verifying a^order ≡ 1 (mod p)");
#endif
        if (order.Pow2MontgomeryModWindowed(divisorData, keepMontgomery: false) != 1UL)
        {
            return false;
        }

#if DEBUG
        Console.WriteLine("Partial factoring order");
#endif

        // TODO: Do we do partial factoring of order multiple times?
        PartialFactorResult factorization = PartialFactor(order, config);
        if (factorization.Factors is null)
        {
            return false;
        }

        if (!factorization.FullyFactored)
        {
            if (factorization.Cofactor <= 1UL)
            {
#if DEBUG
                Console.WriteLine("Cofactor <= 1. No factors found");
#endif
                return false;
            }

            // TODO: Use Open.Numerics.Primality for this final check once it's available.
#if DEBUG
            Console.WriteLine("Cofactor > 1. Testing primality of cofactor");
#endif
            bool isPrime = Open.Numeric.Primes.Prime.Numbers.IsPrime(factorization.Cofactor);
            // bool isPrime = PrimeTester.IsPrimeInternal(factorization.Cofactor, CancellationToken.None);
            if (!isPrime)
            {
                return false;
            }

#if DEBUG
            Console.WriteLine("Adding cofactor as prime factor");
#endif
            factorization = factorization.WithAdditionalPrime(factorization.Cofactor);
        }

        ReadOnlySpan<FactorEntry> span = factorization.Factors;
#if DEBUG
        Console.WriteLine("Verifying prime-power reductions");
#endif
        int length = factorization.Count;
        for (int i = 0; i < length; i++)
        {
            ulong primeFactor = span[i].Value;
            ulong reduced = order;
            for (int iteration = 0; iteration < span[i].Exponent; iteration++)
            {
                if ((reduced % primeFactor) != 0UL)
                {
                    break;
                }

                reduced /= primeFactor;
                if (reduced.Pow2MontgomeryModWindowed(divisorData, keepMontgomery: false) == 1UL)
                {
                    return false;
                }
            }
        }

        return true;
    }

    private static bool TryHeuristicFinish(
        ulong prime,
        ulong order,
        ulong? previousOrder,
        in MontgomeryDivisorData divisorData,
        PrimeOrderSearchConfig config,
        PartialFactorResult phiFactors,
        out ulong result)
    {
        result = 0UL;
        if (order <= 1UL)
        {
            return false;
        }

        // TODO: Do we do partial factoring of order multiple times?
#if DEBUG
        Console.WriteLine("Trying heuristic. Partial factoring order");
#endif
        PartialFactorResult orderFactors = PartialFactor(order, config);
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

            var isPrime = Open.Numeric.Primes.Prime.Numbers.IsPrime(orderFactors.Cofactor);
            // if (!PrimeTester.IsPrimeInternal(orderFactors.Cofactor, CancellationToken.None))
            if (!isPrime)
            {
                return false;
            }

            orderFactors = orderFactors.WithAdditionalPrime(orderFactors.Cofactor);
        }

        int capacity = config.MaxPowChecks <= 0 ? 64 : config.MaxPowChecks * 4;
        List<ulong> candidates = new(capacity);
        FactorEntry[] factorArray = orderFactors.Factors!;
#if DEBUG
        Console.WriteLine("Building candidates list");
#endif
        BuildCandidates(order, factorArray, orderFactors.Count, candidates, capacity);
        if (candidates.Count == 0)
        {
            return false;
        }

#if DEBUG
        Console.WriteLine("Sorting candidates");
#endif
        SortCandidates(prime, previousOrder, candidates);

        int powBudget = config.MaxPowChecks <= 0 ? candidates.Count : config.MaxPowChecks;
        int powUsed = 0;
        int candidateCount = candidates.Count;

#if DEBUG
        Console.WriteLine($"Checking candidates ({candidateCount} candidates, {powBudget} pow budget)");
#endif
        for (int i = 0; i < candidateCount; i++)
        {
            if (powUsed >= powBudget)
            {
                break;
            }

            ulong candidate = candidates[i];
            powUsed++;

            if (candidate.Pow2MontgomeryModWindowed(divisorData, keepMontgomery: false) != 1UL)
            {
                continue;
            }

            if (!TryConfirmCandidate(prime, candidate, divisorData, config, ref powUsed, powBudget))
            {
                continue;
            }

            result = candidate;
            return true;
        }

#if DEBUG
        Console.WriteLine("No candidate confirmed");
#endif
        return false;
    }

    private static void SortCandidates(ulong prime, ulong? previousOrder, List<ulong> candidates)
    {
        ulong previous = previousOrder ?? 0UL;
        int previousGroup = previousOrder.HasValue ? GetGroup(previousOrder.Value, prime) : 1;
        bool hasPrevious = previousOrder.HasValue;

        candidates.Sort((x, y) =>
        {
            CandidateKey keyX = BuildKey(x, prime, previous, previousGroup, hasPrevious);
            CandidateKey keyY = BuildKey(y, prime, previous, previousGroup, hasPrevious);
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

    private static CandidateKey BuildKey(ulong value, ulong prime, ulong previous, int previousGroup, bool hasPrevious)
    {
        int group = GetGroup(value, prime);
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

    private static void BuildCandidatesRecursive(ulong order, ReadOnlySpan<FactorEntry> factors, int index, ulong divisorProduct, List<ulong> candidates, int limit)
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
        ulong primeFactor = factor.Value;
        ulong contribution = 1UL;
        for (int exponent = 0; exponent <= factor.Exponent; exponent++)
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

    private static bool TryConfirmCandidate(ulong prime, ulong candidate, in MontgomeryDivisorData divisorData, PrimeOrderSearchConfig config, ref int powUsed, int powBudget)
    {
        PartialFactorResult factorization = PartialFactor(candidate, config);
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

            bool isPrime = Open.Numeric.Primes.Prime.Numbers.IsPrime(factorization.Cofactor);
            // bool isPrime = PrimeTester.IsPrimeInternal(factorization.Cofactor, CancellationToken.None);
            if (!isPrime)
            {
                return false;
            }

            factorization = factorization.WithAdditionalPrime(factorization.Cofactor);
        }

        ReadOnlySpan<FactorEntry> span = factorization.Factors;
        int length = factorization.Count;
        for (int i = 0; i < length; i++)
        {
            ulong primeFactor = span[i].Value;
            ulong reduced = candidate;
            for (int iteration = 0; iteration < span[i].Exponent; iteration++)
            {
                if ((reduced % primeFactor) != 0UL)
                {
                    break;
                }

                reduced /= primeFactor;
                if (powUsed >= powBudget && powBudget > 0)
                {
                    return false;
                }

                powUsed++;
                if (reduced.Pow2MontgomeryModWindowed(divisorData, keepMontgomery: false) == 1UL)
                {
                    return false;
                }
            }
        }

        return true;
    }

    private static PartialFactorResult PartialFactor(ulong value, PrimeOrderSearchConfig config)
    {
        if (value <= 1UL)
        {
            return PartialFactorResult.Empty;
        }

        Dictionary<ulong, int> counts = new(capacity: 8);
        ulong remaining = value;
        uint[] primes = PrimesGenerator.SmallPrimes;
        ulong[] squares = PrimesGenerator.SmallPrimesPow2;
        int primeCount = primes.Length;
        uint limit = config.SmallFactorLimit == 0 ? uint.MaxValue : config.SmallFactorLimit;

        for (int i = 0; i < primeCount; i++)
        {
            uint primeCandidate = primes[i];

#if DEBUG
            Console.WriteLine($"Trying small prime {primeCandidate}");
#endif
            if (primeCandidate > limit)
            {
                break;
            }

            if (squares[i] > remaining)
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

            counts[primeValue] = exponent;
        }

        List<ulong> pending = new();
        if (remaining > 1UL)
        {
            pending.Add(remaining);
        }

        if (config.PollardRhoMilliseconds > 0 && pending.Count > 0)
        {
#if DEBUG
            Console.WriteLine("Processing pending composites with Pollard's Rho");
#endif
            Stopwatch stopwatch = Stopwatch.StartNew();
            long budgetTicks = TimeSpan.FromMilliseconds(config.PollardRhoMilliseconds).Ticks;
            Stack<ulong> stack = new();
            stack.Push(remaining);
            pending.Clear();

            while (stack.Count > 0)
            {
                ulong composite = stack.Pop();
                if (composite == 1UL)
                {
                    continue;
                }

                bool isPrime = Open.Numeric.Primes.Prime.Numbers.IsPrime(composite);
                // if (PrimeTester.IsPrimeInternal(composite, CancellationToken.None))
                if (isPrime)
                {
                    AddFactor(counts, composite, 1);
                    continue;
                }

                if (stopwatch.ElapsedTicks > budgetTicks)
                {
                    pending.Add(composite);
                    continue;
                }

                if (!TryPollardRho(composite, stopwatch, budgetTicks, out ulong factor))
                {
                    pending.Add(composite);
                    continue;
                }

                ulong quotient = composite / factor;
                stack.Push(factor);
                stack.Push(quotient);
            }
        }

        ulong cofactor = 1UL;
        int pendingCount = pending.Count;
#if DEBUG
        Console.WriteLine($"Processing {pendingCount} pending composites with Open.Numeric.Primes");
#endif
        for (int i = 0; i < pendingCount; i++)
        {
            ulong composite = pending[i];
            bool isPrime = Open.Numeric.Primes.Prime.Numbers.IsPrime(composite);
            // if (PrimeTester.IsPrimeInternal(composite, CancellationToken.None))
            if (isPrime)
            {
                AddFactor(counts, composite, 1);
            }
            else
            {
                cofactor = checked(cofactor * composite);
            }
        }

        if (counts.Count == 0 && cofactor == value)
        {
#if DEBUG
            Console.WriteLine("cofactor is the same as value, no factors found");
#endif
            return new PartialFactorResult(null, value, false, 0);
        }

        FactorEntry[] factors = ArrayPool<FactorEntry>.Shared.Rent(counts.Count);
        int index = 0;
#if DEBUG
        Console.WriteLine($"Collecting {counts.Count} prime factors");
#endif
        foreach (KeyValuePair<ulong, int> entry in counts)
        {
            factors[index] = new FactorEntry(entry.Key, entry.Value);
            index++;
        }

        Span<FactorEntry> span = factors.AsSpan(0, index);
        span.Sort(static (a, b) => a.Value.CompareTo(b.Value));

        FactorEntry[] resultArray = new FactorEntry[index];
        span.CopyTo(resultArray);
        ArrayPool<FactorEntry>.Shared.Return(factors, clearArray: false);

        bool fullyFactored = cofactor == 1UL;
        return new PartialFactorResult(resultArray, cofactor, fullyFactored, index);
    }

    private static bool TryPollardRho(ulong n, Stopwatch stopwatch, long budgetTicks, out ulong factor)
    {
        factor = 0UL;
        if ((n & 1UL) == 0UL)
        {
            factor = 2UL;
            return true;
        }

        Span<byte> buffer = stackalloc byte[8];

        while (true)
        {
            if (stopwatch.ElapsedTicks > budgetTicks)
            {
                return false;
            }

            RandomNumberGenerator.Fill(buffer);
            ulong c = (BinaryPrimitives.ReadUInt64LittleEndian(buffer) % (n - 1UL)) + 1UL;
            RandomNumberGenerator.Fill(buffer);
            ulong x = (BinaryPrimitives.ReadUInt64LittleEndian(buffer) % (n - 2UL)) + 2UL;
            ulong y = x;
            ulong d = 1UL;

            while (d == 1UL)
            {
                if (stopwatch.ElapsedTicks > budgetTicks)
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

    private static ulong CalculateByDoubling(ulong prime)
    {
        ulong order = 1UL;
        ulong pow = 2UL;

        while (pow != 1UL)
        {
            pow <<= 1;
            if (pow >= prime)
            {
                pow -= prime;
            }

            order++;
        }

        return order;
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
    }

    private readonly struct FactorEntry
    {
        public FactorEntry(ulong value, int exponent)
        {
            Value = value;
            Exponent = exponent;
        }

        public ulong Value { get; }

        public int Exponent { get; }
    }

    private readonly struct PartialFactorResult
    {
        public PartialFactorResult(FactorEntry[]? factors, ulong cofactor, bool fullyFactored, int count)
        {
            Factors = factors;
            Cofactor = cofactor;
            FullyFactored = fullyFactored;
            Count = count;
        }

        public FactorEntry[]? Factors { get; }

        public ulong Cofactor { get; }

        public bool FullyFactored { get; }

        public int Count { get; }

        public static PartialFactorResult Empty => new(null, 1UL, true, 0);

        public PartialFactorResult WithAdditionalPrime(ulong prime)
        {
            if (Factors is null)
            {
                FactorEntry[] local = [new FactorEntry(prime, 1)];
                return new PartialFactorResult(local, 1UL, true, 1);
            }

            FactorEntry[] extended = new FactorEntry[Count + 1];
            Array.Copy(Factors, 0, extended, 0, Count);
            extended[Count] = new FactorEntry(prime, 1);
            Array.Sort(extended, static (a, b) => a.Value.CompareTo(b.Value));
            return new PartialFactorResult(extended, 1UL, true, Count + 1);
        }
    }
}
