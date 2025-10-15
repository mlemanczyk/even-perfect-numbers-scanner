using System;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core.Cpu;

public sealed class MersenneNumberDivisorByDivisorCpuTester : IMersenneNumberDivisorByDivisorTester
{
    private readonly object _sync = new();
    [ThreadStatic]
    private static DivisorScanSession? _threadLocalSession; // The nested session type stays private, so pool it here.

    private ulong _divisorLimit;
    private ulong _lastStatusDivisor;
    private int _batchSize = 1_024;


    public int BatchSize
    {
        get => _batchSize;
        set => _batchSize = Math.Max(1, value);
    }

    public void ConfigureFromMaxPrime(ulong maxPrime)
    {
        _divisorLimit = ComputeDivisorLimitFromMaxPrime(maxPrime);
        _lastStatusDivisor = 0UL;
    }

    public ulong DivisorLimit
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            return _divisorLimit;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ulong GetAllowedMaxDivisor(ulong prime) => ComputeAllowedMaxDivisor(prime, _divisorLimit);

    public bool IsPrime(ulong prime, out bool divisorsExhausted)
    {
        ulong allowedMax = ComputeAllowedMaxDivisor(prime, _divisorLimit);

        if (allowedMax < 3UL)
        {
            divisorsExhausted = true;
            return true;
        }

        ulong processedCount;
        bool processedAll;

        bool composite = CheckDivisors(
            prime,
            allowedMax,
            out processedAll,
            out processedCount);

        if (processedCount > 0UL)
        {
            lock (_sync)
            {
                UpdateStatusUnsafe(processedCount);
            }
        }

        if (composite)
        {
            divisorsExhausted = true;
            return false;
        }

        divisorsExhausted = processedAll || composite;
        return true;
    }

    public void PrepareCandidates(in ReadOnlySpan<ulong> primes, Span<ulong> allowedMaxValues)
    {
        ulong divisorLimit = _divisorLimit;
        int length = primes.Length;
        for (int index = 0; index < length; index++)
        {
            allowedMaxValues[index] = ComputeAllowedMaxDivisor(primes[index], divisorLimit);
        }
    }

    public IMersenneNumberDivisorByDivisorTester.IDivisorScanSession CreateDivisorSession()
    {
        DivisorScanSession? session = _threadLocalSession;
        if (session is not null)
        {
            _threadLocalSession = null;
            session.Reset();
            return session;
        }

        return new DivisorScanSession(this);
    }

    private void ReturnSession(DivisorScanSession session)
    {
        _threadLocalSession = session;
    }

    private struct FactorCacheLease
    {
        private Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>? _cache;

        public void EnsureInitialized(ref Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>? cache)
        {
            Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>? current = cache;
            if (current is null)
            {
                current = ThreadStaticPools.RentMersenneFactorCacheDictionary();
                cache = current;
            }

            _cache = current;
        }

        public void Dispose()
        {
            Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>? cache = _cache;
            if (cache is null)
            {
                return;
            }

            ThreadStaticPools.ReturnMersenneFactorCacheDictionary(cache);
            _cache = null;
        }
    }

    private bool CheckDivisors(
        ulong prime,
        ulong allowedMax,
        out bool processedAll,
        out ulong processedCount)
    {
        processedCount = 0UL;
        processedAll = false;

        // The EvenPerfectBitScanner feeds primes >= 138,000,000 here, so allowedMax >= 3 in production runs.
        // Keeping the guard commented out documents the reasoning for benchmarks and tests.
        // if (allowedMax < 3UL)
        // {
        //     return false;
        // }

        UInt128 step = (UInt128)prime << 1;
        // The EvenPerfectBitScanner advances primes far below the overflow boundary, so a zero step here would indicate
        // a corrupted input. Keeping the guard commented out documents the intent without putting a branch on the hot path.
        // if (step == UInt128.Zero)
        // {
        //     processedAll = true;
        //     return false;
        // }

        UInt128 limit = allowedMax;
        UInt128 divisor = step + UInt128.One;
        if (divisor > limit)
        {
            processedAll = true;
            return false;
        }

        FactorCacheLease factorCacheLease = new FactorCacheLease();
        Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>? factorCache = null;
        bool foundDivisor = false;
        bool requireLargeCycleCache = allowedMax > PerfectNumberConstants.MaxQForDivisorCycles;
        if (requireLargeCycleCache)
        {
            factorCacheLease.EnsureInitialized(ref factorCache);
        }

        byte step10 = (byte)(step % 10UL);
        byte step8 = (byte)(step % 8UL);
        byte step5 = (byte)(step % 5UL);
        byte step3 = (byte)(step % 3UL);
        byte step7 = (byte)(step % 7UL);
        byte step11 = (byte)(step % 11UL);

        byte remainder10 = (byte)(divisor % 10UL);
        byte remainder8 = (byte)(divisor % 8UL);
        byte remainder5 = (byte)(divisor % 5UL);
        byte remainder3 = (byte)(divisor % 3UL);
        byte remainder7 = (byte)(divisor % 7UL);
        byte remainder11 = (byte)(divisor % 11UL);

        // DivisorCycleCache cycleCache = DivisorCycleCache.Shared; // Preserve the reference for easy restoration if the fallback cache is re-enabled.

        // Keep the divisibility filters aligned with the divisor-cycle generator so the
        // CPU path never requests cycles that were skipped during cache creation.
        bool lastIsSeven = (prime & 3UL) == 3UL;

        while (divisor <= limit)
        {
            ulong candidate = (ulong)divisor;
            processedCount++;

            bool admissible = lastIsSeven
                ? (remainder10 == 3 || remainder10 == 7 || remainder10 == 9)
                : (remainder10 == 1 || remainder10 == 3 || remainder10 == 9);

            if (admissible && (remainder8 == 1 || remainder8 == 7) && remainder3 != 0 && remainder5 != 0 && remainder7 != 0 && remainder11 != 0)
            {
                MontgomeryDivisorData divisorData = MontgomeryDivisorDataCache.Get(candidate);
                Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>? cacheForCycle = null;
                if (requireLargeCycleCache && candidate > PerfectNumberConstants.MaxQForDivisorCycles)
                {
                    cacheForCycle = factorCache;
                }

                if (!MersenneDivisorCycles.TryCalculateCycleLengthForExponent(candidate, prime, divisorData, cacheForCycle, cacheForCycle is not null, out ulong divisorCycle))
                {
                    divisorCycle = 0UL;
                }

                if (divisorCycle == prime)
                {
                    // A cycle equal to the tested exponent (which is prime in this path) guarantees that the candidate divides
                    // the corresponding Mersenne number because the order of 2 modulo the divisor is exactly p.
                    foundDivisor = true;
                    break;
                }

                if (divisorCycle == 0UL)
                {
                    Console.WriteLine($"Divisor cycle was not calculated for {prime}");
                }
            }

            divisor += step;
            remainder10 = AddMod(remainder10, step10, (byte)10);
            remainder8 = AddMod(remainder8, step8, (byte)8);
            remainder5 = AddMod(remainder5, step5, (byte)5);
            remainder3 = AddMod(remainder3, step3, (byte)3);
            remainder7 = AddMod(remainder7, step7, (byte)7);
            remainder11 = AddMod(remainder11, step11, (byte)11);
        }

        factorCacheLease.Dispose();
        factorCache = null; // The pooled dictionary was returned to the thread-static slot.

        if (foundDivisor)
        {
            processedAll = true;
            return true;
        }

        processedAll = divisor > limit;
        return false;
    }

    private static byte CheckDivisor(ulong prime, ulong divisorCycle, in MontgomeryDivisorData divisorData)
    {
        ulong residue = prime.Pow2MontgomeryModWithCycleCpu(divisorCycle, divisorData);
        return residue == 1UL ? (byte)1 : (byte)0;
    }

    private static byte AddMod(byte value, byte delta, byte modulus)
    {
        int sum = value + delta;
        if (sum >= modulus)
        {
            sum -= modulus;
        }

        return (byte)sum;
    }

    private void UpdateStatusUnsafe(ulong processedCount)
    {
        if (processedCount == 0UL)
        {
            return;
        }

        ulong interval = PerfectNumberConstants.ConsoleInterval;
        if (interval == 0UL)
        {
            _lastStatusDivisor = 0UL;
            return;
        }

        ulong total = _lastStatusDivisor + processedCount;
        // TODO: Replace this modulo with the ring-buffer style counter (subtract loop) used in the fast CLI
        // status benchmarks so we avoid `%` in this hot loop while still wrapping progress correctly.
        _lastStatusDivisor = total % interval;
    }

    private static ulong ComputeDivisorLimitFromMaxPrime(ulong maxPrime)
    {
        if (maxPrime <= 1UL)
        {
            return 0UL;
        }

        if (maxPrime - 1UL >= 64UL)
        {
            return ulong.MaxValue;
        }

        return (1UL << (int)(maxPrime - 1UL)) - 1UL;
    }

    private static ulong ComputeAllowedMaxDivisor(ulong prime, ulong divisorLimit)
    {
        if (prime <= 1UL)
        {
            return 0UL;
        }

        if (prime - 1UL >= 64UL)
        {
            return divisorLimit;
        }

        return Math.Min((1UL << (int)(prime - 1UL)) - 1UL, divisorLimit);
    }

    private sealed class DivisorScanSession : IMersenneNumberDivisorByDivisorTester.IDivisorScanSession
    {
        private readonly MersenneNumberDivisorByDivisorCpuTester _owner;
        private bool _disposed;

        internal DivisorScanSession(MersenneNumberDivisorByDivisorCpuTester owner)
        {
            _owner = owner;
        }

        internal void Reset()
        {
            _disposed = false;
        }

        public void CheckDivisor(ulong divisor, in MontgomeryDivisorData divisorData, ulong divisorCycle, in ReadOnlySpan<ulong> primes, Span<byte> hits)
        {
            int length = primes.Length;
            if (length == 0)
            {
                return;
            }

            MontgomeryDivisorData cachedData = divisorData;
            if (cachedData.Modulus != divisor)
            {
                cachedData = MontgomeryDivisorDataCache.Get(divisor);
            }

            if (divisorCycle == 0UL)
            {
                divisorCycle = DivisorCycleCache.Shared.GetCycleLength(divisor);
                if (divisorCycle == 0UL)
                {
                    hits.Clear();
                    return;
                }
            }

            // Keep these remainder steppers in place so future updates continue reusing the previously computed residues.
            // They are critical for avoiding repeated full Montgomery exponentiation work when scanning divisors.
            var exponentStepper = new ExponentRemainderStepper(cachedData);
            if (!exponentStepper.IsValidModulus)
            {
                hits.Clear();
                return;
            }

            var cycleStepper = new CycleRemainderStepper(divisorCycle);

            ulong remainder = cycleStepper.Initialize(primes[0]);
            hits[0] = remainder == 0UL
                ? (exponentStepper.ComputeNextIsUnity(primes[0]) ? (byte)1 : (byte)0)
                : (byte)0;

            for (int i = 1; i < length; i++)
            {
                remainder = cycleStepper.ComputeNext(primes[i]);
                if (remainder != 0UL)
                {
                    hits[i] = 0;
                    continue;
                }

                hits[i] = exponentStepper.ComputeNextIsUnity(primes[i]) ? (byte)1 : (byte)0;
            }
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            _owner.ReturnSession(this);
        }
    }
}


