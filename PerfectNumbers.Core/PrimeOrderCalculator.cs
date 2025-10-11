using System.Buffers;
using System.Buffers.Binary;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

internal static partial class PrimeOrderCalculator
{
    internal enum PrimeOrderMode
    {
        Heuristic,
        Strict,
    }

    internal readonly struct PrimeOrderSearchConfig(uint smallFactorLimit, int pollardRhoMilliseconds, int maxPowChecks, PrimeOrderMode mode)
    {
        public static PrimeOrderSearchConfig HeuristicDefault => new(smallFactorLimit: 100_000, pollardRhoMilliseconds: 24, maxPowChecks: 24, PrimeOrderMode.Heuristic);
        public static PrimeOrderSearchConfig StrictDefault => new(smallFactorLimit: 1_000_000, pollardRhoMilliseconds: 0, maxPowChecks: 0, PrimeOrderMode.Strict);

        public readonly uint SmallFactorLimit = smallFactorLimit;
        public readonly int PollardRhoMilliseconds = pollardRhoMilliseconds;
        public readonly int MaxPowChecks = maxPowChecks;
        public readonly PrimeOrderMode Mode = mode;
    }

    internal enum PrimeOrderHeuristicDevice
    {
        Cpu,
        Gpu,
    }

    [ThreadStatic]
    private static bool s_pow2ModeInitialized;

    [ThreadStatic]
    private static bool s_allowGpuPow2;

    [ThreadStatic]
    private static bool s_allowGpuCycle;

    private static bool IsGpuPow2Allowed => s_pow2ModeInitialized && s_allowGpuPow2;

    private static bool IsGpuCycleAllowed => s_pow2ModeInitialized && s_allowGpuCycle;

    [ThreadStatic]
    private static PrimeOrderHeuristicDevice s_deviceMode;

    [ThreadStatic]
    private static bool s_debugLoggingEnabled;

    [ThreadStatic]
    private static List<ulong>? s_partialFactorPending;

    [ThreadStatic]
    private static Stack<ulong>? s_partialFactorCompositeStack;

    [ThreadStatic]
    private static Dictionary<ulong, int>? s_partialFactorCounts;

    [ThreadStatic]
    private static ulong s_cachedDivisorCycleModulus;

    [ThreadStatic]
    private static ulong s_cachedDivisorCycleLength;

    [ThreadStatic]
    private static bool s_hasCachedDivisorCycle;

    [ThreadStatic]
    private static int s_divisorCycleSuppressionDepth;

    [ThreadStatic]
    private static MutableUInt128 s_pollardRhoPolynomialBuffer;

    private static bool IsGpuHeuristicDevice => s_pow2ModeInitialized && s_deviceMode == PrimeOrderHeuristicDevice.Gpu;

    private readonly struct Pow2ModeScope
    {
        private readonly bool _previousInitialized;
        private readonly bool _previousAllowPow2;
        private readonly bool _previousAllowCycle;
        private readonly PrimeOrderHeuristicDevice _previousDevice;

        public Pow2ModeScope(PrimeOrderHeuristicDevice device)
        {
            _previousInitialized = s_pow2ModeInitialized;
            _previousAllowPow2 = s_allowGpuPow2;
            _previousAllowCycle = s_allowGpuCycle;
            _previousDevice = s_deviceMode;
            s_pow2ModeInitialized = true;
            bool gpuAvailable = !GpuContextPool.ForceCpu;
            bool allowCpuPow2 = gpuAvailable && device == PrimeOrderHeuristicDevice.Cpu;
            bool allowCpuCycle = gpuAvailable && device == PrimeOrderHeuristicDevice.Cpu && GpuKernelPool.CpuCycleUsesGpu;
            s_allowGpuPow2 = device == PrimeOrderHeuristicDevice.Gpu || allowCpuPow2;
            s_allowGpuCycle = device == PrimeOrderHeuristicDevice.Gpu || allowCpuCycle;
            s_deviceMode = device;
        }

        public void Dispose()
        {
            s_allowGpuPow2 = _previousAllowPow2;
            s_allowGpuCycle = _previousAllowCycle;
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

    private static bool IsDivisorCycleSuppressed => s_divisorCycleSuppressionDepth > 0;

    private readonly struct DivisorCycleSuppressionScope
    {
        private readonly bool _suppress;

        private DivisorCycleSuppressionScope(bool suppress)
        {
            _suppress = suppress;
        }

        public static readonly DivisorCycleSuppressionScope Suppressed = new(true);

        public static readonly DivisorCycleSuppressionScope Unsuppressed = new(false);

        public void Enter()
        {
            if (_suppress)
            {
                s_divisorCycleSuppressionDepth++;
            }
        }

        public void Exit()
        {
            if (_suppress)
            {
                s_divisorCycleSuppressionDepth--;
            }
        }
    }

    private static Pow2ModeScope UsePow2Mode(PrimeOrderHeuristicDevice device) => new(device);

    private static DebugLoggingScope UseDebugLogging(bool enabled) => new(enabled);

    private static DivisorCycleSuppressionScope SuppressDivisorCycleUsage(bool suppress) =>
        suppress ? DivisorCycleSuppressionScope.Suppressed : DivisorCycleSuppressionScope.Unsuppressed;

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

    public static ulong Calculate(
        ulong prime,
        ulong? previousOrder,
        in MontgomeryDivisorData divisorData,
        in PrimeOrderSearchConfig config,
        PrimeOrderHeuristicDevice device)
    {
        var scope = UsePow2Mode(device);
        ulong order = CalculateInternal(prime, previousOrder, divisorData, config);
        scope.Dispose();
        return order;
    }

    private static ulong CalculateInternal(ulong prime, ulong? previousOrder, in MontgomeryDivisorData divisorData, in PrimeOrderSearchConfig config)
    {
        // EvenPerfectBitScanner only reaches this helper with divisors of the form 2 * exponent * k + 1 where exponent >= 5,
        // so the small-prime return path is unreachable in production.
        // if (prime <= 3UL)
        // {
        //     return prime == 3UL ? 2UL : 1UL;
        // }

        ulong phi = prime - 1UL;

        // The by-divisor scanner currently defaults to the CPU heuristic device, but --mersenne-device may request GPU heuristics.
        if (IsGpuHeuristicDevice && PrimeOrderGpuHeuristics.TryCalculateOrder(prime, previousOrder, config, divisorData, out ulong gpuOrder))
        {
            return gpuOrder;
        }

        PartialFactorResult phiFactors = PartialFactor(phi, config);

        ulong result;
        // The by-divisor scan always yields phi = prime - 1 with at least the factor 2 captured during small-prime sieving,
        // so PartialFactor never returns a null factor list on this call path.
        // if (phiFactors.Factors is null)
        // {
        //     result = CalculateByFactorizationCpu(prime, divisorData, config);
        //     phiFactors.Dispose();
        //     return result;
        // }

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
        var cycleScope = SuppressDivisorCycleUsage(config.Mode == PrimeOrderMode.Heuristic && !IsGpuHeuristicDevice);
        cycleScope.Enter();
        try
        {
            if (phiFactors.FullyFactored && TrySpecialMaxCpu(phi, prime, phiFactors, divisorData))
            {
                return phi;
            }

            ulong candidateOrder = InitializeStartingOrderCpu(prime, phi, divisorData);
            candidateOrder = ExponentLoweringCpu(candidateOrder, prime, phiFactors, divisorData);

            PartialFactorResult orderFactors = PartialFactor(candidateOrder, config);
            if (orderFactors.Factors is null)
            {
                orderFactors.Dispose();
                return candidateOrder;
            }

            // Reuse the computed factorization for both the confirmation check and the heuristic pass
            // to avoid factoring the same order twice on the by-divisor CPU pipeline.
            if (TryConfirmOrderCpu(prime, candidateOrder, divisorData, ref orderFactors))
            {
                orderFactors.Dispose();
                return candidateOrder;
            }

            if (config.Mode == PrimeOrderMode.Strict)
            {
                orderFactors.Dispose();
                return CalculateByFactorizationCpu(prime, divisorData, config);
            }

            if (TryHeuristicFinishCpu(prime, candidateOrder, previousOrder, divisorData, config, ref orderFactors, phiFactors, out ulong order))
            {
                orderFactors.Dispose();
                return order;
            }

            orderFactors.Dispose();
            return candidateOrder;
        }
        finally
        {
            cycleScope.Exit();
        }
    }

    private static bool TrySpecialMaxCpu(ulong phi, ulong prime, PartialFactorResult factors, in MontgomeryDivisorData divisorData)
    {
        ReadOnlySpan<FactorEntry> factorSpan = factors.Factors;
        int length = factors.Count;
        for (int i = 0; i < length; i++)
        {
            ulong factor = factorSpan[i].Value;
            ulong reduced = phi / factor;
            if (Pow2EqualsOneCpu(reduced, prime, divisorData))
            {
                return false;
            }
        }

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

    [ThreadStatic]
    private static ArrayPool<FactorEntry>? _factorPool;
    private static ArrayPool<FactorEntry> FactorPool => _factorPool ??= ArrayPool<FactorEntry>.Create();

    private const int FactorEntryStackThreshold = 32;

    [ThreadStatic]
    private static ArrayPool<ulong>? _ulongPool;
    private static ArrayPool<ulong> ULongPool => _ulongPool ??= ArrayPool<ulong>.Create();

    private static ulong ExponentLoweringCpu(ulong order, ulong prime, PartialFactorResult factors, in MontgomeryDivisorData divisorData)
    {
        ArrayPool<FactorEntry> factorPool = FactorPool;

        FactorEntry[]? tempArray = null;
        try
        {
            ReadOnlySpan<FactorEntry> factorSpan = factors.Factors;
            int length = factors.Count;
            int capacity = length + 1;

            Span<FactorEntry> buffer;
            if (capacity <= FactorEntryStackThreshold)
            {
                buffer = stackalloc FactorEntry[capacity];
            }
            else
            {
                tempArray = factorPool.Rent(capacity);
                buffer = tempArray.AsSpan(0, capacity);
            }

            factorSpan.CopyTo(buffer);

            bool isPrime = Open.Numeric.Primes.Prime.Numbers.IsPrime(factors.Cofactor);
            // bool isPrime = PrimeTester.IsPrimeInternal(factors.Cofactor, CancellationToken.None);
            if (!factors.FullyFactored && factors.Cofactor > 1UL && isPrime)
            {
                buffer[length] = new FactorEntry(factors.Cofactor);
                length++;
            }

            buffer = buffer.Slice(0, length); // Reuse the same span restricted to populated entries.
            buffer.Sort(static (a, b) => a.Value.CompareTo(b.Value));

            for (int i = 0; i < buffer.Length; i++)
            {
                FactorEntry entry = buffer[i];
                ulong primeFactor = entry.Value;
                int exponent = entry.Exponent;
                for (int iteration = 0; iteration < exponent; iteration++)
                {
                    if ((order % primeFactor) != 0UL)
                    {
                        break;
                    }

                    ulong reduced = order / primeFactor;
                    if (Pow2EqualsOneCpu(reduced, prime, divisorData))
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
                factorPool.Return(tempArray, clearArray: false);
            }
        }
    }

    private static bool TryConfirmOrderCpu(ulong prime, ulong order, in MontgomeryDivisorData divisorData, ref PartialFactorResult orderFactors)
    {
        // The candidate order is always a positive divisor of phi in this flow, so zero never appears here.
        // if (order == 0UL)
        // {
        //     return false;
        // }

        // Calculating `a^order ≡ 1 (mod p)` is a prerequisite for `order` being the actual order of 2 modulo `p`.
        // DebugLog("Verifying a^order ≡ 1 (mod p)");
        if (!Pow2EqualsOneCpu(order, prime, divisorData))
        {
            return false;
        }

        // PartialFactor returns a null factor span when it cannot peel off any divisors, so keep the guard for composite candidate orders.
        if (orderFactors.Factors is null)
        {
            return false;
        }

        if (!orderFactors.FullyFactored)
        {
            // A leftover cofactor of 0 or 1 indicates the partial factorization produced no useful prime divisor.
            if (orderFactors.Cofactor <= 1UL)
            {
                return false;
            }

            // DebugLog("Cofactor > 1. Testing primality of cofactor");
            bool isPrime = Open.Numeric.Primes.Prime.Numbers.IsPrime(orderFactors.Cofactor);
            // bool isPrime = PrimeTester.IsPrimeInternal(orderFactors.Cofactor, CancellationToken.None);
            if (!isPrime)
            {
                return false;
            }

            // DebugLog("Adding cofactor as prime factor");
            PartialFactorResult extended = orderFactors.WithAdditionalPrime(orderFactors.Cofactor);
            orderFactors.Dispose();
            orderFactors = extended;
        }

        ReadOnlySpan<FactorEntry> span = orderFactors.Factors!;
        // DebugLog("Verifying prime-power reductions");
        int length = orderFactors.Count;
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
                if (Pow2EqualsOneCpu(reduced, prime, divisorData))
                {
                    return false;
                }
            }
        }

        return true;
    }

    private static bool TryHeuristicFinishCpu(
        ulong prime,
        ulong order,
        ulong? previousOrder,
        in MontgomeryDivisorData divisorData,
        in PrimeOrderSearchConfig config,
        ref PartialFactorResult orderFactors,
        PartialFactorResult phiFactors,
        out ulong result)
    {
        result = 0UL;
        // The candidate order stays at least 2 for odd divisors above 3, so this guard never triggers in the EvenPerfectBitScanner flow.
        // if (order <= 1UL)
        // {
        //     return false;
        // }

        // PartialFactor returns a null factor span when it cannot peel off any divisors, so the heuristic pass must bail in that case.
        if (orderFactors.Factors is null)
        {
            return false;
        }

        if (!orderFactors.FullyFactored)
        {
            // The partial factoring helper reports a cofactor of 0 or 1 when nothing usable remains, so short-circuit the heuristic stage.
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

            PartialFactorResult extended = orderFactors.WithAdditionalPrime(orderFactors.Cofactor);
            orderFactors.Dispose();
            orderFactors = extended;
        }

        int capacity = config.MaxPowChecks <= 0 ? 64 : config.MaxPowChecks * 4;
        List<ulong> candidates = new(capacity);
        FactorEntry[] factorArray = orderFactors.Factors!;
        // DebugLog("Building candidates list");
        BuildCandidates(order, factorArray, orderFactors.Count, candidates, capacity);
        if (candidates.Count == 0)
        {
            return false;
        }

        // DebugLog("Sorting candidates");
        SortCandidates(prime, previousOrder, candidates);

        int powBudget = config.MaxPowChecks <= 0 ? candidates.Count : config.MaxPowChecks;
        int powUsed = 0;
        int candidateCount = candidates.Count;
        bool allowGpuBatch = true;
        Span<ulong> candidateSpan = CollectionsMarshal.AsSpan(candidates);

        // DebugLog(() => $"Checking candidates ({candidateCount} candidates, {powBudget} pow budget)");
        int index = 0;
        const int MaxGpuBatchSize = 1024;
        const int StackGpuBatchSize = 1024;
        Span<ulong> stackGpuRemainders = stackalloc ulong[StackGpuBatchSize];
        ArrayPool<ulong> uLongPool = ULongPool;
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
                    Span<ulong> localRemainders = stackGpuRemainders.Slice(0, batchSize);
                    status = PrimeOrderGpuHeuristics.TryPow2ModBatch(batch, prime, localRemainders, divisorData);
                    if (status == GpuPow2ModStatus.Success)
                    {
                        gpuSuccess = true;
                        gpuStackRemainders = true;
                    }
                }
                else
                {
                    gpuPool = uLongPool.Rent(batchSize);
                    Span<ulong> pooledRemainders = gpuPool.AsSpan(0, batchSize);
                    status = PrimeOrderGpuHeuristics.TryPow2ModBatch(batch, prime, pooledRemainders, divisorData);
                    if (status == GpuPow2ModStatus.Success)
                    {
                        pooledGpuRemainders = pooledRemainders;
                        gpuSuccess = true;
                    }
                    else
                    {
                        uLongPool.Return(gpuPool, clearArray: false);
                        gpuPool = null;
                    }
                }

                if (!gpuSuccess && (status == GpuPow2ModStatus.Overflow || status == GpuPow2ModStatus.Unavailable))
                {
                    allowGpuBatch = false;
                }
            }

            for (int i = 0; i < batchSize && powUsed < powBudget; i++)
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
                    equalsOne = Pow2EqualsOneCpu(candidate, prime, divisorData);
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
                    uLongPool.Return(gpuPool, clearArray: false);
                }

                result = candidate;
                return true;
            }

            if (gpuPool is not null)
            {
                uLongPool.Return(gpuPool, clearArray: false);
            }

            index += batchSize;
        }

        // DebugLog("No candidate confirmed");
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

                bool isPrime = Open.Numeric.Primes.Prime.Numbers.IsPrime(factorization.Cofactor);
                // bool isPrime = PrimeTester.IsPrimeInternal(factorization.Cofactor, CancellationToken.None);
                if (!isPrime)
                {
                    return false;
                }

                PartialFactorResult extended = factorization.WithAdditionalPrime(factorization.Cofactor);
                factorization.Dispose();
                factorization = extended;
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
                    if (Pow2EqualsOneCpu(reduced, prime, divisorData))
                    {
                        return false;
                    }
                }
            }

            return true;
        }
        finally
        {
            factorization.Dispose();
        }
    }

    private static bool TryGetDivisorCycle(ulong prime, in MontgomeryDivisorData divisorData, out ulong cycleLength)
    {
        if (prime <= 1UL || IsDivisorCycleSuppressed)
        {
            cycleLength = 0UL;
            return false;
        }

        if (s_hasCachedDivisorCycle && s_cachedDivisorCycleModulus == prime)
        {
            cycleLength = s_cachedDivisorCycleLength;
            return cycleLength != 0UL;
        }

        var suppression = SuppressDivisorCycleUsage(true);
        suppression.Enter();
        try
        {
            cycleLength = MersenneDivisorCycles.CalculateCycleLength(prime, divisorData);
            s_cachedDivisorCycleModulus = prime;
            s_cachedDivisorCycleLength = cycleLength;
            s_hasCachedDivisorCycle = true;
            return cycleLength != 0UL;
        }
        finally
        {
            suppression.Exit();
        }
    }

    private static bool Pow2EqualsOneCpu(ulong exponent, ulong prime, in MontgomeryDivisorData divisorData)
    {
        if (IsGpuPow2Allowed)
        {
            GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2Mod(exponent, prime, out ulong remainder, divisorData);
            if (status == GpuPow2ModStatus.Success)
            {
                return remainder == 1UL;
            }
        }

        if (TryGetDivisorCycle(prime, divisorData, out ulong cycleLength) && cycleLength != 0UL)
        {
            return exponent.Pow2MontgomeryModWithCycleCpu(cycleLength, divisorData) == 1UL;
        }

        return exponent.Pow2MontgomeryModWindowedCpu(divisorData, keepMontgomery: false) == 1UL;
    }

    private static List<ulong> AcquirePendingCompositeList(int capacityHint)
    {
        List<ulong>? list = s_partialFactorPending;
        if (list is null)
        {
            list = new List<ulong>(capacityHint);
            s_partialFactorPending = list;
            return list;
        }

        list.Clear();
        if (list.Capacity < capacityHint)
        {
            list.Capacity = capacityHint;
        }

        return list;
    }

    private static Stack<ulong> AcquireCompositeStack(int capacityHint)
    {
        Stack<ulong>? stack = s_partialFactorCompositeStack;
        if (stack is null)
        {
            stack = new Stack<ulong>(capacityHint);
            s_partialFactorCompositeStack = stack;
            return stack;
        }

        stack.Clear();
        stack.EnsureCapacity(capacityHint);
        return stack;
    }

    private static Dictionary<ulong, int> AcquireFactorCountDictionary(int capacityHint)
    {
        Dictionary<ulong, int>? dictionary = s_partialFactorCounts;
        if (dictionary is null)
        {
            dictionary = new Dictionary<ulong, int>(capacityHint);
            s_partialFactorCounts = dictionary;
            return dictionary;
        }

        dictionary.Clear();
        dictionary.EnsureCapacity(capacityHint);
        return dictionary;
    }

    private static PartialFactorResult PartialFactor(ulong value, in PrimeOrderSearchConfig config)
    {
        if (value <= 1UL)
        {
            return PartialFactorResult.Empty;
        }

        const int FactorSlotCount = GpuSmallPrimeFactorSlots;
        Span<ulong> primeSlots = stackalloc ulong[FactorSlotCount];
        Span<int> exponentSlots = stackalloc int[FactorSlotCount];
        primeSlots.Clear();
        exponentSlots.Clear();

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

        List<ulong> pending = AcquirePendingCompositeList(2);
        Stack<ulong>? compositeStack = null;

        try
        {
            if (!gpuFactored)
            {
                primeSlots.Clear();
                exponentSlots.Clear();
                if (!TryPopulateSmallPrimeFactorsCpu(value, limit, primeSlots, exponentSlots, out factorCount, out remaining))
                {
                    useDictionary = true;
                    counts = AcquireFactorCountDictionary(Math.Max(factorCount, 8));
                    remaining = PopulateSmallPrimeFactorsCpu(value, limit, counts);
                }
            }

            if (remaining > 1UL)
            {
                pending.Add(remaining);
            }

            if (config.PollardRhoMilliseconds > 0 && pending.Count > 0)
            {
                Stopwatch stopwatch = Stopwatch.StartNew();
                long budgetTicks = TimeSpan.FromMilliseconds(config.PollardRhoMilliseconds).Ticks;
                compositeStack = AcquireCompositeStack(Math.Max(pending.Count * 2, 4));
                compositeStack.Push(remaining);
                pending.Clear();

                while (compositeStack.Count > 0)
                {
                    ulong composite = compositeStack.Pop();
                    if (composite <= 1UL)
                    {
                        continue;
                    }

                    bool isPrime = Open.Numeric.Primes.Prime.Numbers.IsPrime(composite);
                    if (isPrime)
                    {
                        AddFactorToCollector(ref useDictionary, ref counts, primeSlots, exponentSlots, ref factorCount, composite, 1);
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
                    compositeStack.Push(factor);
                    compositeStack.Push(quotient);
                }
            }

            ulong cofactor = 1UL;
            int pendingCount = pending.Count;
            for (int i = 0; i < pendingCount; i++)
            {
                ulong composite = pending[i];
                bool isPrime = Open.Numeric.Primes.Prime.Numbers.IsPrime(composite);
                if (isPrime)
                {
                    AddFactorToCollector(ref useDictionary, ref counts, primeSlots, exponentSlots, ref factorCount, composite, 1);
                }
                else if (composite > 1UL)
                {
                    cofactor = checked(cofactor * composite);
                }
            }

            if (useDictionary)
            {
                if ((counts is null || counts.Count == 0) && cofactor == value)
                {
                    return PartialFactorResult.Rent(null, value, false, 0);
                }

                if (counts is null || counts.Count == 0)
                {
                    return PartialFactorResult.Rent(null, cofactor, cofactor == 1UL, 0);
                }

                int dictionaryCount = counts.Count;
                FactorEntry[] factors = dictionaryCount >= PerfectNumberConstants.PooledArrayThreshold
                    ? ThreadLocalArrayPool<FactorEntry>.Shared.Rent(dictionaryCount)
                    : new FactorEntry[dictionaryCount];
                int index = 0;
                foreach (KeyValuePair<ulong, int> entry in counts)
                {
                    factors[index] = new FactorEntry(entry.Key, entry.Value);
                    index++;
                }

                Span<FactorEntry> span = factors.AsSpan(0, index);
                span.Sort(static (a, b) => a.Value.CompareTo(b.Value));

                bool fullyFactored = cofactor == 1UL;
                return PartialFactorResult.Rent(factors, cofactor, fullyFactored, index);
            }

            int actualCount = 0;
            for (int i = 0; i < factorCount; i++)
            {
                if (primeSlots[i] != 0UL && exponentSlots[i] != 0)
                {
                    actualCount++;
                }
            }

            if (actualCount == 0)
            {
                if (cofactor == value)
                {
                    return PartialFactorResult.Rent(null, value, false, 0);
                }

                return PartialFactorResult.Rent(null, cofactor, cofactor == 1UL, 0);
            }

            FactorEntry[] array = actualCount >= PerfectNumberConstants.PooledArrayThreshold
                ? ThreadLocalArrayPool<FactorEntry>.Shared.Rent(actualCount)
                : new FactorEntry[actualCount];
            int arrayIndex = 0;
            for (int i = 0; i < factorCount; i++)
            {
                ulong primeValue = primeSlots[i];
                int exponentValue = exponentSlots[i];
                if (primeValue == 0UL || exponentValue == 0)
                {
                    continue;
                }

                array[arrayIndex] = new FactorEntry(primeValue, exponentValue);
                arrayIndex++;
            }

            Span<FactorEntry> arraySpan = array.AsSpan(0, arrayIndex);
            arraySpan.Sort(static (a, b) => a.Value.CompareTo(b.Value));
            bool fullyFactoredArray = cofactor == 1UL;
            return PartialFactorResult.Rent(array, cofactor, fullyFactoredArray, arrayIndex);
        }
        finally
        {
            pending.Clear();
            counts?.Clear();
            compositeStack?.Clear();
        }
    }

    private const int GpuSmallPrimeFactorSlots = 64;

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

        if (value <= 1UL)
        {
            return true;
        }

        int capacity = Math.Min(primeTargets.Length, exponentTargets.Length);
        if (capacity == 0)
        {
            return false;
        }

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
            if (primeSquare != 0UL && primeSquare > remainingLocal)
            {
                break;
            }

            ulong primeValue = primeCandidate;
            if ((remainingLocal % primeValue) != 0UL)
            {
                continue;
            }

            int exponent = 0;
            do
            {
                remainingLocal /= primeValue;
                exponent++;
            }
            while ((remainingLocal % primeValue) == 0UL);

            if (factorCount >= capacity)
            {
                return false;
            }

            primeTargets[factorCount] = primeValue;
            exponentTargets[factorCount] = exponent;
            factorCount++;
        }

        remaining = remainingLocal;
        return true;
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

        return remaining;
    }

    private static bool TryPopulateSmallPrimeFactorsGpu(ulong value, uint limit, Dictionary<ulong, int> counts, out ulong remaining)
    {
        Span<ulong> primeBuffer = stackalloc ulong[GpuSmallPrimeFactorSlots];
        Span<int> exponentBuffer = stackalloc int[GpuSmallPrimeFactorSlots];
        primeBuffer.Clear();
        exponentBuffer.Clear();

        if (!PrimeOrderGpuHeuristics.TryPartialFactor(value, limit, primeBuffer, exponentBuffer, out int factorCount, out ulong gpuRemaining, out _))
        {
            remaining = value;
            return false;
        }

        remaining = gpuRemaining;
        for (int i = 0; i < factorCount; i++)
        {
            ulong primeValue = primeBuffer[i];
            int exponent = exponentBuffer[i];
            if (primeValue == 0UL || exponent == 0)
            {
                continue;
            }

            counts[primeValue] = exponent;
        }

        return true;
    }
    private static bool TryPollardRho(ulong n, Stopwatch stopwatch, long budgetTicks, out ulong factor)
    {
        factor = 0UL;
        if ((n & 1UL) == 0UL)
        {
            factor = 2UL;
            return true;
        }

        while (true)
        {
            if (stopwatch.ElapsedTicks > budgetTicks)
            {
                return false;
            }

            ulong c = (DeterministicRandom.NextUInt64() % (n - 1UL)) + 1UL;
            ulong x = (DeterministicRandom.NextUInt64() % (n - 2UL)) + 2UL;
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
        ref MutableUInt128 polynomial = ref s_pollardRhoPolynomialBuffer;
        polynomial.Set(x);
        polynomial.Multiply(x);
        polynomial.Add(c);
        return polynomial.Mod(modulus);
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

        counts ??= AcquireFactorCountDictionary(Math.Max(count, 8));
        CopyFactorsToDictionary(primes, exponents, count, counts);
        count = 0;
        useDictionary = true;
        AddFactor(counts, prime, exponent);
    }

    private static void CopyFactorsToDictionary(
        ReadOnlySpan<ulong> primes,
        ReadOnlySpan<int> exponents,
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

    private static ulong CalculateByFactorizationCpu(ulong prime, in MontgomeryDivisorData divisorData, in PrimeOrderSearchConfig config)
    {
        if (!IsGpuHeuristicDevice && IsGpuCycleAllowed)
        {
            PrimeOrderSearchConfig strictConfig = config.Mode == PrimeOrderMode.Strict
                ? config
                : new PrimeOrderSearchConfig(config.SmallFactorLimit, config.PollardRhoMilliseconds, config.MaxPowChecks, PrimeOrderMode.Strict);
            if (PrimeOrderGpuHeuristics.TryCalculateOrder(prime, previousOrder: null, strictConfig, divisorData, out ulong gpuOrder))
            {
                return gpuOrder;
            }
        }

        ulong phi = prime - 1UL;
        Dictionary<ulong, int> counts = new(capacity: 8);
        FactorCompletely(phi, counts);
        if (counts.Count == 0)
        {
            return phi;
        }

        List<KeyValuePair<ulong, int>> entries = new(counts);
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
        if (value <= 1UL)
        {
            return;
        }

        if (Open.Numeric.Primes.Prime.Numbers.IsPrime(value))
        {
            AddFactor(counts, value, 1);
            return;
        }

        ulong factor = PollardRhoStrict(value);
        FactorCompletely(factor, counts);
        FactorCompletely(value / factor, counts);
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
