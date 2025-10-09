using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using PerfectNumbers.Core.Gpu;
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

    private enum HeuristicFailureReason
    {
        PhiPartialFactorizationFailed,
        StrictModeRequested,
        HeuristicPipelineUnresolved,
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

    internal enum PrimeOrderHeuristicDevice
    {
        Cpu,
        Gpu,
    }

    [ThreadStatic]
    private static bool s_pow2ModeInitialized;

    [ThreadStatic]
    private static bool s_allowGpuPow2;

    private static bool IsGpuPow2Allowed => s_pow2ModeInitialized && s_allowGpuPow2;


    [ThreadStatic]
    private static PrimeOrderHeuristicDevice s_deviceMode;

    [ThreadStatic]
    private static bool s_debugLoggingEnabled;

    private static bool IsGpuHeuristicDevice => s_pow2ModeInitialized && s_deviceMode == PrimeOrderHeuristicDevice.Gpu;
    private readonly struct Pow2ModeScope : IDisposable
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

    private readonly struct DebugLoggingScope : IDisposable
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

    public static PrimeOrderResult Calculate(ulong prime, ulong? previousOrder, in MontgomeryDivisorData divisorData, PrimeOrderSearchConfig config)
        => Calculate(prime, previousOrder, divisorData, config, PrimeOrderHeuristicDevice.Gpu);

    public static PrimeOrderResult Calculate(
        ulong prime,
        ulong? previousOrder,
        in MontgomeryDivisorData divisorData,
        PrimeOrderSearchConfig config,
        PrimeOrderHeuristicDevice device)
    {
        using var scope = UsePow2Mode(device);
        return CalculateInternal(prime, previousOrder, divisorData, config);
    }

    private static PrimeOrderResult CalculateInternal(ulong prime, ulong? previousOrder, MontgomeryDivisorData divisorData, PrimeOrderSearchConfig config)
    {
        if (prime <= 3UL)
        {
            return new PrimeOrderResult(PrimeOrderStatus.Found, prime == 3UL ? 2UL : 1UL);
        }

        ulong phi = prime - 1UL;

        if (IsGpuHeuristicDevice && PrimeOrderGpuHeuristics.TryCalculateOrder(prime, previousOrder, config, divisorData, out PrimeOrderResult gpuResult))
        {
            return gpuResult;
        }

        using var debugScope = UseDebugLogging(!IsGpuHeuristicDevice);
        // DebugLog("Partial factoring φ(p)");

        PartialFactorResult phiFactors = PartialFactor(phi, config);

        if (phiFactors.Factors is null)
        {
            // DebugLog("No factors found");
            HeuristicFailureLog.Record(prime, null, HeuristicFailureReason.PhiPartialFactorizationFailed);
            return FinishStrictly(prime, divisorData, config.Mode);
        }

        return RunHeuristicPipeline(prime, previousOrder, config, divisorData, phi, phiFactors);
    }

    private static PrimeOrderResult RunHeuristicPipeline(
        ulong prime,
        ulong? previousOrder,
        PrimeOrderSearchConfig config,
        in MontgomeryDivisorData divisorData,
        ulong phi,
        PartialFactorResult phiFactors)
    {
        // DebugLog("Trying special max check");
        if (phiFactors.FullyFactored && TrySpecialMax(phi, prime, phiFactors, divisorData))
        {
            return new PrimeOrderResult(PrimeOrderStatus.Found, phi);
        }

        // DebugLog("Initializing starting order");
        ulong candidateOrder = InitializeStartingOrder(prime, phi, divisorData);
        candidateOrder = ExponentLowering(candidateOrder, prime, phiFactors, divisorData);

        // DebugLog("Trying to confirm order");
        if (TryConfirmOrder(prime, candidateOrder, divisorData, config))
        {
            return new PrimeOrderResult(PrimeOrderStatus.Found, candidateOrder);
        }

        if (config.Mode == PrimeOrderMode.Strict)
        {
            HeuristicFailureLog.Record(prime, candidateOrder, HeuristicFailureReason.StrictModeRequested);
            return FinishStrictly(prime, divisorData, PrimeOrderMode.Strict);
        }

        if (TryHeuristicFinish(prime, candidateOrder, previousOrder, divisorData, config, phiFactors, out ulong order))
        {
            return new PrimeOrderResult(PrimeOrderStatus.Found, order);
        }

        // DebugLog("Heuristic unresolved, finishing strictly");
        HeuristicFailureLog.Record(prime, candidateOrder, HeuristicFailureReason.HeuristicPipelineUnresolved);
        return FinishStrictly(prime, divisorData, config.Mode);
    }

    private static PrimeOrderResult FinishStrictly(ulong prime, in MontgomeryDivisorData divisorData, PrimeOrderMode mode)
    {
        ulong strictOrder = CalculateByFactorization(prime, divisorData);
        PrimeOrderStatus status = mode == PrimeOrderMode.Strict ? PrimeOrderStatus.Found : PrimeOrderStatus.HeuristicUnresolved;
        return new PrimeOrderResult(status, strictOrder);
    }

    private static bool TrySpecialMax(ulong phi, ulong prime, PartialFactorResult factors, in MontgomeryDivisorData divisorData)
    {
        ReadOnlySpan<FactorEntry> factorSpan = factors.Factors;
        int length = factors.Count;
        for (int i = 0; i < length; i++)
        {
            ulong factor = factorSpan[i].Value;
            ulong reduced = phi / factor;
            if (Pow2EqualsOne(reduced, prime, divisorData))
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
            if (Pow2EqualsOne(half, prime, divisorData))
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
                    if (Pow2EqualsOne(reduced, prime, divisorData))
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
        // DebugLog("Verifying a^order ≡ 1 (mod p)");
        if (!Pow2EqualsOne(order, prime, divisorData))
        {
            return false;
        }

        // DebugLog("Partial factoring order");

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
            // DebugLog("Cofactor <= 1. No factors found");
                return false;
            }

            // DebugLog("Cofactor > 1. Testing primality of cofactor");
            bool isPrime = Open.Numeric.Primes.Prime.Numbers.IsPrime(factorization.Cofactor);
            // bool isPrime = PrimeTester.IsPrimeInternal(factorization.Cofactor, CancellationToken.None);
            if (!isPrime)
            {
                return false;
            }

            // DebugLog("Adding cofactor as prime factor");
            factorization = factorization.WithAdditionalPrime(factorization.Cofactor);
        }

        ReadOnlySpan<FactorEntry> span = factorization.Factors;
        // DebugLog("Verifying prime-power reductions");
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
                if (Pow2EqualsOne(reduced, prime, divisorData))
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
        // DebugLog("Trying heuristic. Partial factoring order");
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
                    gpuPool = ArrayPool<ulong>.Shared.Rent(batchSize);
                    Span<ulong> pooledRemainders = gpuPool.AsSpan(0, batchSize);
                    status = PrimeOrderGpuHeuristics.TryPow2ModBatch(batch, prime, pooledRemainders, divisorData);
                    if (status == GpuPow2ModStatus.Success)
                    {
                        pooledGpuRemainders = pooledRemainders;
                        gpuSuccess = true;
                    }
                    else
                    {
                        ArrayPool<ulong>.Shared.Return(gpuPool, clearArray: false);
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
                    equalsOne = Pow2EqualsOne(candidate, prime, divisorData);
                }
                if (!equalsOne)
                {
                    continue;
                }

                if (!TryConfirmCandidate(prime, candidate, divisorData, config, ref powUsed, powBudget))
                {
                    continue;
                }

                if (gpuPool is not null)
                {
                    ArrayPool<ulong>.Shared.Return(gpuPool, clearArray: false);
                }

                result = candidate;
                return true;
            }

            if (gpuPool is not null)
            {
                ArrayPool<ulong>.Shared.Return(gpuPool, clearArray: false);
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
                if (Pow2EqualsOne(reduced, prime, divisorData))
                {
                    return false;
                }
            }
        }

        return true;
    }

    private static bool Pow2EqualsOne(ulong exponent, ulong prime, in MontgomeryDivisorData divisorData)
    {
        if (IsGpuPow2Allowed)
        {
            GpuPow2ModStatus status = PrimeOrderGpuHeuristics.TryPow2Mod(exponent, prime, out ulong remainder, divisorData);
            if (status == GpuPow2ModStatus.Success)
            {
                return remainder == 1UL;
            }
        }

        return exponent.Pow2MontgomeryModWindowed(divisorData, keepMontgomery: false) == 1UL;
    }

    private static class HeuristicFailureLog
    {
        private const string LogFileName = "prime-order-heuristic-fallbacks.log";
        private const int FlushThreshold = 4096;
        private const int MaxBufferedEntries = 16;
        private static readonly ConcurrentQueue<StringBuilder> s_builderPool = new();
        private static readonly ThreadLocal<StringBuilder?> s_threadBuilder = new(() => AcquireBuilder(), trackAllValues: true);
        private static readonly ThreadLocal<int> s_threadEntryCount = new(() => 0, trackAllValues: true);
        private static readonly object s_fileLock = new();
        private static readonly string s_logPath = Path.Combine(AppContext.BaseDirectory, LogFileName);
        private static readonly UTF8Encoding s_utf8NoBom = new(false);

        static HeuristicFailureLog()
        {
            AppDomain.CurrentDomain.ProcessExit += static (_, _) => FlushAll(force: true);
        }

        public static void Record(ulong prime, ulong? candidateOrder, HeuristicFailureReason reason)
        {
            string primeText = prime.ToString(CultureInfo.InvariantCulture);
            string? candidateText = candidateOrder.HasValue ? candidateOrder.Value.ToString(CultureInfo.InvariantCulture) : null;
            RecordInternal(primeText, candidateText, reason);
        }

        public static void Record(UInt128 prime, UInt128? candidateOrder, HeuristicFailureReason reason)
        {
            string primeText = prime.ToString(CultureInfo.InvariantCulture);
            string? candidateText = candidateOrder.HasValue ? candidateOrder.Value.ToString(CultureInfo.InvariantCulture) : null;
            RecordInternal(primeText, candidateText, reason);
        }

        private static void RecordInternal(string primeText, string? candidateText, HeuristicFailureReason reason)
        {
            StringBuilder? existing = s_threadBuilder.Value;
            StringBuilder builder = existing ?? AcquireBuilder();
            int count = s_threadEntryCount.Value + 1;
            bool forceFlush = count >= MaxBufferedEntries;
            builder.Append(DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture));
            builder.Append(" | prime=");
            builder.Append(primeText);
            builder.Append(" | reason=");
            builder.Append(reason);
            if (candidateText is not null)
            {
                builder.Append(" | candidateOrder=");
                builder.Append(candidateText);
            }

            builder.AppendLine();
            FlushBuilderIfNeeded(builder, forceFlush);
            if (builder.Length == 0)
            {
                count = 0;
            }

            s_threadEntryCount.Value = count;
            s_threadBuilder.Value = builder;
        }

        private static void FlushAll(bool force)
        {
            foreach (StringBuilder? builder in s_threadBuilder.Values)
            {
                if (builder is not null)
                {
                    FlushBuilderIfNeeded(builder, force);
                }
            }
        }

        private static void FlushBuilderIfNeeded(StringBuilder builder, bool force)
        {
            if (builder.Length == 0)
            {
                return;
            }

            if (!force && builder.Length < FlushThreshold)
            {
                return;
            }

            string text = builder.ToString();
            builder.Clear();
            lock (s_fileLock)
            {
                using FileStream stream = new FileStream(s_logPath, FileMode.Append, FileAccess.Write, FileShare.ReadWrite);
                using StreamWriter writer = new StreamWriter(stream, s_utf8NoBom);
                writer.Write(text);
            }
        }

        private static StringBuilder AcquireBuilder()
        {
            if (s_builderPool.TryDequeue(out StringBuilder? builder))
            {
                builder.Clear();
                return builder;
            }

            return new StringBuilder(FlushThreshold);
        }
    }

    private static PartialFactorResult PartialFactor(ulong value, PrimeOrderSearchConfig config)
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
        ulong remaining;

        bool gpuFactored = false;
        if (IsGpuHeuristicDevice)
        {
            gpuFactored = PrimeOrderGpuHeuristics.TryPartialFactor(value, limit, primeSlots, exponentSlots, out factorCount, out remaining, out _);
        }
        else
        {
            remaining = value;
        }

        if (!gpuFactored)
        {
            primeSlots.Clear();
            exponentSlots.Clear();
            if (!TryPopulateSmallPrimeFactorsCpu(value, limit, primeSlots, exponentSlots, out factorCount, out remaining))
            {
                useDictionary = true;
                counts = new Dictionary<ulong, int>(capacity: 8);
                remaining = PopulateSmallPrimeFactorsCpu(value, limit, counts);
            }
        }

        List<ulong> pending = new();
        if (remaining > 1UL)
        {
            pending.Add(remaining);
        }

        if (config.PollardRhoMilliseconds > 0 && pending.Count > 0)
        {
            // DebugLog("Processing pending composites with Pollard's Rho");
            Stopwatch stopwatch = Stopwatch.StartNew();
            long budgetTicks = TimeSpan.FromMilliseconds(config.PollardRhoMilliseconds).Ticks;
            Stack<ulong> stack = new();
            stack.Push(remaining);
            pending.Clear();

            while (stack.Count > 0)
            {
                ulong composite = stack.Pop();
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
                stack.Push(factor);
                stack.Push(quotient);
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
                // DebugLog("cofactor is the same as value, no factors found");
                return new PartialFactorResult(null, value, false, 0);
            }

            if (counts is null || counts.Count == 0)
            {
                return new PartialFactorResult(null, cofactor, cofactor == 1UL, 0);
            }

            FactorEntry[] factors = ArrayPool<FactorEntry>.Shared.Rent(counts.Count);
            int index = 0;
            // DebugLog(() => $"Collecting {counts.Count} prime factors");
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
                // DebugLog("cofactor is the same as value, no factors found");
                return new PartialFactorResult(null, value, false, 0);
            }

            return new PartialFactorResult(null, cofactor, cofactor == 1UL, 0);
        }

        FactorEntry[] array = new FactorEntry[actualCount];
        int arrayIndex = 0;
        // DebugLog(() => $"Collecting {actualCount} prime factors");
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

        Array.Sort(array, static (a, b) => a.Value.CompareTo(b.Value));
        bool fullyFactoredArray = cofactor == 1UL;
        return new PartialFactorResult(array, cofactor, fullyFactoredArray, array.Length);
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

        counts ??= new Dictionary<ulong, int>(Math.Max(count, 8));
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

    private static ulong CalculateByFactorization(ulong prime, in MontgomeryDivisorData divisorData)
    {
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
                if (Pow2EqualsOne(candidate, prime, divisorData))
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

        Span<byte> buffer = stackalloc byte[8];
        while (true)
        {
            RandomNumberGenerator.Fill(buffer);
            ulong c = (BinaryPrimitives.ReadUInt64LittleEndian(buffer) % (n - 1UL)) + 1UL;

            RandomNumberGenerator.Fill(buffer);
            ulong x = (BinaryPrimitives.ReadUInt64LittleEndian(buffer) % (n - 2UL)) + 2UL;
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

    private readonly struct FactorEntry(ulong value, int exponent)
    {
        public readonly ulong Value = value;
        public readonly int Exponent = exponent;
    }

    private readonly struct PartialFactorResult(FactorEntry[]? factors, ulong cofactor, bool fullyFactored, int count)
    {
        public readonly FactorEntry[]? Factors = factors;
        public readonly ulong Cofactor = cofactor;
        public readonly bool FullyFactored = fullyFactored;
        public readonly int Count = count;
        public readonly static PartialFactorResult Empty = new(null, 1UL, true, 0);

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
