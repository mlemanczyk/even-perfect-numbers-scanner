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

    public bool IsPrime(ulong prime, out bool divisorsExhausted)
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

        ulong processedCount = 0UL;
        ulong lastProcessed = 0UL;
        bool processedAll = false;

        bool composite = CheckDivisors(
            prime,
            allowedMax,
            out lastProcessed,
            out processedAll,
            out processedCount);

        if (processedCount > 0UL)
        {
            lock (_sync)
            {
                UpdateStatusUnsafe(lastProcessed, processedCount);
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
        out ulong processedCount)
    {
        lastProcessed = 0UL;
        processedCount = 0UL;
        processedAll = false;

        if (allowedMax < 3UL)
        {
            return false;
        }

        Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>? factorCache = null;
        ulong divisor = 3UL;
        while (divisor <= allowedMax)
        {
            processedCount++;
            lastProcessed = divisor;

            MontgomeryDivisorData divisorData = MontgomeryDivisorDataCache.Get(divisor);
            // TODO: Hoist MontgomeryDivisorData acquisition into the divisor-cycle cache so we reuse the
            // staged ProcessEightBitWindows-ready metadata instead of reloading the slower Montgomery
            // structs for every divisor.
            ulong divisorCycle;
            if (divisor <= PerfectNumberConstants.MaxQForDivisorCycles)
            {
                divisorCycle = DivisorCycleCache.Shared.GetCycleLength(divisor);
            }
            else
            {
                factorCache ??= new Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>(8);
                if (!MersenneDivisorCycles.TryCalculateCycleLengthForExponent(divisor, prime, factorCache, out divisorCycle) || divisorCycle == 0UL)
                {
                    divisorCycle = DivisorCycleCache.Shared.GetCycleLength(divisor);
                }
            }
            if (divisorCycle != 0UL)
            {
                byte hit = CheckDivisor(prime, divisorCycle, divisorData);
                if (hit != 0)
                {
                    processedAll = true;
                    return true;
                }
            }

            if (divisor >= allowedMax - 1UL)
            {
                processedAll = true;
                break;
            }

            if (divisor > ulong.MaxValue - 2UL)
            {
                break;
            }

            divisor += 2UL;
            // TODO: Replace this linear increment with the batched divisor-cycle walker validated in the
            // CPU by-divisor benchmarks so we advance directly to the next cached divisor candidate
            // instead of testing every odd integer.
        }

        if (!processedAll)
        {
            processedAll = divisor > allowedMax;
        }
        return false;
    }

    private static byte CheckDivisor(ulong prime, ulong divisorCycle, in MontgomeryDivisorData divisorData)
    {
        ulong residue = prime.Pow2MontgomeryModWithCycle(divisorCycle, divisorData);
        return residue == 1UL ? (byte)1 : (byte)0;
    }

    private void UpdateStatusUnsafe(ulong lastProcessed, ulong processedCount)
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

        public void CheckDivisor(ulong divisor, ulong divisorCycle, ReadOnlySpan<ulong> primes, Span<byte> hits)
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

            if (divisorCycle == 0UL)
            {
                divisorCycle = DivisorCycleCache.Shared.GetCycleLength(divisor);
                if (divisorCycle == 0UL)
                {
                    hits.Clear();
                    return;
                }
            }

            MontgomeryDivisorData divisorData = MontgomeryDivisorDataCache.Get(divisor);

            var exponentStepper = new ExponentRemainderStepper(divisorData);
            if (!exponentStepper.IsValidModulus)
            {
                hits.Clear();
                return;
            }

            var stepper = new CycleRemainderStepper(divisorCycle);

            ulong remainder = stepper.Initialize(primes[0]);
            hits[0] = remainder == 0UL
                ? (exponentStepper.ComputeNextIsUnity(primes[0]) ? (byte)1 : (byte)0)
                : (byte)0;

            for (int i = 1; i < length; i++)
            {
                remainder = stepper.ComputeNext(primes[i]);
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


