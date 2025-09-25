using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;
using MontgomeryDivisorData = PerfectNumbers.Core.Gpu.MersenneNumberDivisorByDivisorGpuTester.MontgomeryDivisorData;

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
        set => _useDivisorCycles = value;
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

            MontgomeryDivisorData divisorData = CreateMontgomeryDivisorData(divisor);
            byte hit;
            if (useCycles)
            {
                DivisorCycleCache.CycleBlock cycleBlock = DivisorCycleCache.Shared.Acquire(divisor);
                ulong divisorCycle = cycleBlock.GetCycle(divisor);
                hit = CheckDivisor(prime, divisorCycle != 0UL, divisorCycle, divisorData);
            }
            else
            {
                hit = CheckDivisor(prime, false, 0UL, divisorData);
            }

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
                : prime.Pow2MontgomeryMod(divisorData);

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

    private static MontgomeryDivisorData CreateMontgomeryDivisorData(ulong modulus)
    {
        if (modulus <= 1UL || (modulus & 1UL) == 0UL)
        {
            return new MontgomeryDivisorData(modulus, 0UL, 0UL, 0UL);
        }

        return new MontgomeryDivisorData(
                modulus,
                ComputeMontgomeryNPrime(modulus),
                ComputeMontgomeryResidue(1UL, modulus),
                ComputeMontgomeryResidue(2UL, modulus));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong ComputeMontgomeryResidue(ulong value, ulong modulus) => (ulong)((UInt128)value * (UInt128.One << 64) % modulus);

    private static ulong ComputeMontgomeryNPrime(ulong modulus)
    {
        ulong inv = modulus;
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        return unchecked(0UL - inv);
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

            bool cycleEnabled = _owner._useDivisorCycles && divisorCycle != 0UL;
            MontgomeryDivisorData divisorData = CreateMontgomeryDivisorData(divisor);

            for (int i = 0; i < length; i++)
            {
                ulong prime = primes[i];

                if (cycleEnabled)
                {
                    ulong remainder = prime % divisorCycle;
                    if (remainder != 0UL)
                    {
                        hits[i] = 0;
                        continue;
                    }
                }

                ulong residue = cycleEnabled
                    ? prime.Pow2MontgomeryModWithCycle(divisorCycle, divisorData)
                    : prime.Pow2MontgomeryMod(divisorData);

                hits[i] = residue == 1UL ? (byte)1 : (byte)0;
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

