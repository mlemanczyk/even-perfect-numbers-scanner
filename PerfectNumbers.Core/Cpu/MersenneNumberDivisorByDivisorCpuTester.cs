using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Numerics;
using PerfectNumbers.Core;

namespace PerfectNumbers.Core.Cpu;

public sealed class MersenneNumberDivisorByDivisorCpuTester : IMersenneNumberDivisorByDivisorTester
{
    private readonly object _sync = new();
    private readonly ConcurrentBag<DivisorScanSession> _sessionPool = new();
    private ulong _divisorLimit;
    private ulong _lastStatusDivisor;
    private bool _isConfigured;
    private int _batchSize = 1_024;


    public int BatchSize
    {
        get => _batchSize;
        set => _batchSize = Math.Max(1, value);
    }

    public void ConfigureFromMaxPrime(ulong maxPrime)
    {
        lock (_sync)
        {
            _divisorLimit = ComputeDivisorLimitFromMaxPrime(maxPrime);
            _lastStatusDivisor = 0UL;
            _isConfigured = true;
        }
    }

    public ulong DivisorLimit
    {
        get
        {
            lock (_sync)
            {
                if (!_isConfigured)
                {
                    throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
                }

                return _divisorLimit;
            }
        }
    }

    public ulong GetAllowedMaxDivisor(ulong prime)
    {
        lock (_sync)
        {
            if (!_isConfigured)
            {
                throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
            }

            return ComputeAllowedMaxDivisor(prime, _divisorLimit);
        }
    }

    public bool IsPrime(ulong prime, out bool divisorsExhausted, TimeSpan? timeLimit = null)
    {
        ulong allowedMax;
        lock (_sync)
        {
            if (!_isConfigured)
            {
                throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
            }

            allowedMax = ComputeAllowedMaxDivisor(prime, _divisorLimit);
        }

        if (allowedMax < 3UL)
        {
            divisorsExhausted = true;
            return true;
        }

        ulong processedCount;
        ulong lastProcessed;
        bool processedAll;

        bool composite = CheckDivisors(
            prime,
            allowedMax,
            out lastProcessed,
            out processedAll,
            out processedCount,
            timeLimit);

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

    public void PrepareCandidates(ReadOnlySpan<ulong> primes, Span<ulong> allowedMaxValues)
    {
        if (allowedMaxValues.Length < primes.Length)
        {
            throw new ArgumentException("allowedMaxValues span must be at least as long as primes span.", nameof(allowedMaxValues));
        }

        ulong divisorLimit;

        lock (_sync)
        {
            if (!_isConfigured)
            {
                throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
            }

            divisorLimit = _divisorLimit;
        }

        for (int index = 0; index < primes.Length; index++)
        {
            allowedMaxValues[index] = ComputeAllowedMaxDivisor(primes[index], divisorLimit);
        }
    }

    public IMersenneNumberDivisorByDivisorTester.IDivisorScanSession CreateDivisorSession()
    {
        lock (_sync)
        {
            if (!_isConfigured)
            {
                throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
            }

            if (_sessionPool.TryTake(out DivisorScanSession? session))
            {
                session.Reset();
                return session;
            }

            return new DivisorScanSession(this);
        }
    }

    private void ReturnSession(DivisorScanSession session)
    {
        _sessionPool.Add(session);
    }

    private bool CheckDivisors(
        ulong prime,
        ulong allowedMax,
        out ulong lastProcessed,
        out bool processedAll,
        out ulong processedCount,
        TimeSpan? timeLimit)
    {
        lastProcessed = 0UL;
        processedCount = 0UL;
        processedAll = false;

        PrimeTestTimeLimit limitGuard;
        if (!PrimeTestTimeLimit.TryCreate(timeLimit, out limitGuard))
        {
            return false;
        }

        bool enforceLimit = limitGuard.IsActive;
        ulong iterationCounter = 0UL;

        if (allowedMax < 3UL)
        {
            return false;
        }

        UInt128 step = (UInt128)prime << 1;
        if (step == UInt128.Zero)
        {
            processedAll = true;
            return false;
        }

        UInt128 limit = allowedMax;
        UInt128 divisor = step + UInt128.One;
        if (divisor > limit)
        {
            processedAll = true;
            return false;
        }

        Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>? factorCache = null;
        DivisorCycleCache cycleCache = DivisorCycleCache.Shared;

        ulong cachedModulus = 0UL;
        MontgomeryDivisorData cachedDivisorData = default;
        bool hasCachedDivisorData = false;

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

        // Keep the divisibility filters aligned with the divisor-cycle generator so the
        // CPU path never requests cycles that were skipped during cache creation.
        bool lastIsSeven = (prime & 3UL) == 3UL;

        while (divisor <= limit)
        {
            if (enforceLimit && (iterationCounter == 0UL || (iterationCounter & 63UL) == 0UL) && limitGuard.HasExpired())
            {
                processedAll = false;
                return false;
            }

            iterationCounter++;
            ulong candidate = (ulong)divisor;
            processedCount++;
            lastProcessed = candidate;

            bool admissible = lastIsSeven
                ? (remainder10 == 3 || remainder10 == 7 || remainder10 == 9)
                : (remainder10 == 1 || remainder10 == 3 || remainder10 == 9);

            if (admissible && (remainder8 == 1 || remainder8 == 7) && remainder3 != 0 && remainder5 != 0 && remainder7 != 0 && remainder11 != 0)
            {
                if (!hasCachedDivisorData || cachedModulus != candidate)
                {
                    cachedDivisorData = MontgomeryDivisorData.FromModulus(candidate);
                    cachedModulus = candidate;
                    hasCachedDivisorData = true;
                }

                MontgomeryDivisorData divisorData = cachedDivisorData;
                ulong divisorCycle;

                if (candidate <= PerfectNumberConstants.MaxQForDivisorCycles)
                {
                    divisorCycle = cycleCache.GetCycleLength(candidate);
                }
                else
                {
                    factorCache ??= new Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>(8);
                    // EvenPerfectBitScanner reaches this path only for divisors above the precomputed cycle snapshot.
                    // Those candidates follow q = 2 * prime * k + 1 with prime >= 5, so the downstream order calculator
                    // can rely on an odd divisor and a prime exponent when deriving the cycle length.
                    if (!MersenneDivisorCycles.TryCalculateCycleLengthForExponent(candidate, prime, divisorData, factorCache, out divisorCycle) || divisorCycle == 0UL)
                    {
                        divisorCycle = cycleCache.GetCycleLength(candidate);
                    }
                }

                if (divisorCycle == prime)
                {
                    // A cycle equal to the tested exponent (which is prime in this path)
                    // guarantees that the candidate divides the corresponding Mersenne
                    // number because the order of 2 modulo the divisor is exactly p.
                    processedAll = true;
                    return true;
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
        private bool _hasCachedDivisorData;
        private ulong _cachedDivisor;
        private MontgomeryDivisorData _cachedDivisorData;

        internal DivisorScanSession(MersenneNumberDivisorByDivisorCpuTester owner)
        {
            _owner = owner;
        }

        internal void Reset()
        {
            _disposed = false;
            _hasCachedDivisorData = false;
            _cachedDivisor = 0UL;
            _cachedDivisorData = default;
        }

        public void CheckDivisor(ulong divisor, in MontgomeryDivisorData divisorData, ulong divisorCycle, ReadOnlySpan<ulong> primes, Span<byte> hits)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(DivisorScanSession));
            }

            int length = primes.Length;
            if (length == 0)
            {
                return;
            }

            MontgomeryDivisorData cachedData = divisorData;
            if (cachedData.Modulus != divisor)
            {
                if (!_hasCachedDivisorData || _cachedDivisor != divisor)
                {
                    _cachedDivisorData = MontgomeryDivisorData.FromModulus(divisor);
                    _cachedDivisor = divisor;
                    _hasCachedDivisorData = true;
                }

                cachedData = _cachedDivisorData;
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


