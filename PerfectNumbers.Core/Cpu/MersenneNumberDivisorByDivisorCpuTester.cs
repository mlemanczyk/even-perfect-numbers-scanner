using System;
using System.Collections.Concurrent;
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
    private bool _useDivisorCycles;
    private int _batchSize = 1_024;

    public bool UseDivisorCycles
    {
        get => _useDivisorCycles;
        set => _useDivisorCycles = value; // TODO: Delete the toggle once divisor cycle data is always consulted so CPU scans never fall back to the slower pure-Montgomery path.
    }

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
        bool useCycles;

        lock (_sync)
        {
            if (!_isConfigured)
            {
                throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
            }

            allowedMax = ComputeAllowedMaxDivisor(prime, _divisorLimit);
            useCycles = _useDivisorCycles;
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
                        useCycles,
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
                    bool useCycles,
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

        ulong divisor = 3UL;
        while (divisor <= allowedMax)
        {
            processedCount++;
            lastProcessed = divisor;

            MontgomeryDivisorData divisorData = MontgomeryDivisorDataCache.Get(divisor);
            // TODO: Hoist MontgomeryDivisorData acquisition into the divisor-cycle cache so we reuse the
            // staged ProcessEightBitWindows-ready metadata instead of reloading the slower Montgomery
            // structs for every divisor.
            byte hit;
            ulong divisorCycle = 0UL;
            DivisorCycleCache.CycleBlock? cycleBlock = null;
            if (useCycles)
            {
                cycleBlock = DivisorCycleCache.Shared.Acquire(divisor);
                // TODO: Swap this block lease for the direct single-cycle lookup once the cache exposes it
                // so the CPU scanner mirrors the benchmarked single-block policy without retaining mutable
                // blocks beyond the startup snapshot.
                divisorCycle = cycleBlock.GetCycle(divisor);
                // TODO: When divisorCycle is zero (cache miss), compute only that single cycle on the device
                // selected by the current settings, skip inserting the result into the shared cache, and keep
                // operating with the single on-disk block without requesting additional blocks.
                hit = CheckDivisor(prime, divisorCycle != 0UL, divisorCycle, divisorData);
            }
            else
            {
                // TODO: Remove this no-cycle branch and require on-demand cycle computation (without
                // mutating the cache) so CPU scans never fall back to the slower Montgomery-only path
                // identified in the benchmarks.
                hit = CheckDivisor(prime, false, 0UL, divisorData);
            }

            cycleBlock?.Dispose();

            if (hit != 0)
            {
                processedAll = true;
                return true;
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

        private static byte CheckDivisor(ulong prime, bool useCycles, ulong divisorCycle, in MontgomeryDivisorData divisorData)
        {
            ulong modulus = divisorData.Modulus;
            if (modulus <= 1UL || (modulus & 1UL) == 0UL)
            {
                return 0;
            }

            ulong residue = useCycles && divisorCycle != 0UL
                ? prime.Pow2MontgomeryModWithCycle(divisorCycle, divisorData)
                : prime.Pow2MontgomeryMod(divisorData); // TODO: Switch to the upcoming ProcessEightBitWindows helper once the
                                                        // scalar Pow2Minus1Mod adopts it so CPU scans match the GPU speedups.

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

            ulong modulus = divisor;
            if (modulus <= 1UL || (modulus & 1UL) == 0UL)
            {
                hits.Clear();
                return;
            }

            bool useDivisorCycles = _owner._useDivisorCycles;
            if (useDivisorCycles && divisorCycle == 0UL)
            {
                // TODO: Replace this guard with an on-demand CPU/GPU cycle computation path that skips
                // cache insertion and continues operating with the single cached block instead of
                // requesting additional cycle batches when a lookup misses.
                throw new InvalidOperationException($"Missing divisor cycle for divisor {divisor}.");
            }

            bool cycleEnabled = useDivisorCycles;
            MontgomeryDivisorData divisorData = MontgomeryDivisorDataCache.Get(divisor);

            var exponentStepper = new ExponentRemainderStepper(divisorData);
            if (!exponentStepper.IsValidModulus)
            {
                hits.Clear();
                return;
            }

            if (cycleEnabled)
            {
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

                return;
            }

            // TODO: Remove this no-cycle fallback once divisor cycles are mandatory; the per-prime pow2 ladder here
            // keeps the slower Montgomery stepping alive even though the benchmarks showed the cached cycle path is far
            // faster for large divisor sets.
            for (int i = 0; i < length; i++)
            {
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

