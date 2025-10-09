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
        out ulong processedCount)
    {
        lastProcessed = 0UL;
        processedCount = 0UL;
        processedAll = false;

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
            ulong candidate = (ulong)divisor;
            processedCount++;
            lastProcessed = candidate;

            bool admissible = lastIsSeven
                ? (remainder10 == 3 || remainder10 == 7 || remainder10 == 9)
                : (remainder10 == 1 || remainder10 == 3 || remainder10 == 9);

            if (admissible && (remainder8 == 1 || remainder8 == 7) && remainder3 != 0 && remainder5 != 0 && remainder7 != 0 && remainder11 != 0)
            {
                MontgomeryDivisorData divisorData = MontgomeryDivisorDataCache.Get(candidate);
                ulong divisorCycle;

                if (candidate <= PerfectNumberConstants.MaxQForDivisorCycles)
                {
                    divisorCycle = cycleCache.GetCycleLength(candidate);
                }
                else
                {
                    factorCache ??= new Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>(8);
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
        ulong residue = prime.Pow2MontgomeryModWithCycle(divisorCycle, divisorData);
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
        if (total >= interval)
        {
            do
            {
                total -= interval;
            }
            while (total >= interval);
        }

        _lastStatusDivisor = total;
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
        private readonly Dictionary<ulong, PrimeDivisorState> _primeStates = new();
        private bool _disposed;

        internal DivisorScanSession(MersenneNumberDivisorByDivisorCpuTester owner)
        {
            _owner = owner;
        }

        internal void Reset()
        {
            _disposed = false;
            _primeStates.Clear();
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

            var cycleStepper = new CycleRemainderStepper(divisorCycle);
            ExponentRemainderStepper exponentStepper = default;
            bool exponentStepperInitialized = false;
            byte hitValue = 0; // Reused for every prime processed in this batch.

            bool EnsureExponentStepperInitialized()
            {
                if (exponentStepperInitialized)
                {
                    return true;
                }

                exponentStepper = new ExponentRemainderStepper(cachedData);
                if (!exponentStepper.IsValidModulus)
                {
                    return false;
                }

                exponentStepperInitialized = true;
                return true;
            }

            ulong remainder = cycleStepper.Initialize(primes[0]);
            if (remainder == 0UL)
            {
                if (TryComputeAnalyticHit(primes[0], divisor, cachedData, out hitValue))
                {
                    hits[0] = hitValue;
                }
                else
                {
                    if (!EnsureExponentStepperInitialized())
                    {
                        hits.Clear();
                        return;
                    }

                    hits[0] = exponentStepper.ComputeNextIsUnity(primes[0]) ? (byte)1 : (byte)0;
                }
            }
            else
            {
                hits[0] = 0;
            }

            for (int i = 1; i < length; i++)
            {
                remainder = cycleStepper.ComputeNext(primes[i]);
                if (remainder != 0UL)
                {
                    hits[i] = 0;
                    continue;
                }

                if (TryComputeAnalyticHit(primes[i], divisor, cachedData, out hitValue))
                {
                    hits[i] = hitValue;
                    continue;
                }

                if (!EnsureExponentStepperInitialized())
                {
                    hits.Clear();
                    return;
                }

                hits[i] = exponentStepper.ComputeNextIsUnity(primes[i]) ? (byte)1 : (byte)0;
            }
        }

        private bool TryComputeAnalyticHit(ulong prime, ulong divisor, in MontgomeryDivisorData divisorData, out byte hit)
        {
            hit = 0;

            if (!TryGetMersenneMinusOne(prime, out ulong mersenne))
            {
                return false;
            }

            if (divisor <= 1UL)
            {
                return false;
            }

            ulong step = prime << 1;
            if (step == 0UL)
            {
                return false;
            }

            if (!_primeStates.TryGetValue(prime, out PrimeDivisorState state) || !state.HasState)
            {
                state = CreatePrimeState(step, divisor, mersenne, divisorData);
                _primeStates[prime] = state;
                hit = state.Remainder == 0UL ? (byte)1 : (byte)0;
                return true;
            }

            if (state.Step != step || divisor <= state.LastDivisor)
            {
                state = CreatePrimeState(step, divisor, mersenne, divisorData);
                _primeStates[prime] = state;
                hit = state.Remainder == 0UL ? (byte)1 : (byte)0;
                return true;
            }

            ulong difference = divisor - state.LastDivisor;
            if (difference == 0UL)
            {
                hit = state.Remainder == 0UL ? (byte)1 : (byte)0;
                return true;
            }

            if (difference % step != 0UL)
            {
                state = CreatePrimeState(step, divisor, mersenne, divisorData);
                _primeStates[prime] = state;
                hit = state.Remainder == 0UL ? (byte)1 : (byte)0;
                return true;
            }

            ulong increments = difference / step;
            ulong quotient = state.Quotient;
            ulong remainder = state.Remainder;
            ulong currentDivisor = state.LastDivisor;

            for (ulong index = 0UL; index < increments; index++)
            {
                ulong nextDivisor = currentDivisor + step;
                Int128 stepTimesQuotient = (Int128)step * (Int128)quotient;
                Int128 numerator = stepTimesQuotient - (Int128)remainder;
                Int128 divisorInt = (Int128)nextDivisor;
                Int128 deltaInt = numerator >= 0
                    ? (numerator + divisorInt - Int128.One) / divisorInt
                    : numerator / divisorInt;

                if (deltaInt < Int128.Zero)
                {
                    deltaInt = Int128.Zero;
                }

                ulong delta = (ulong)deltaInt;
                if (delta > quotient)
                {
                    state = CreatePrimeState(step, divisor, mersenne, divisorData);
                    _primeStates[prime] = state;
                    hit = state.Remainder == 0UL ? (byte)1 : (byte)0;
                    return true;
                }

                Int128 t = (Int128)remainder - stepTimesQuotient;
                Int128 nextRemainder = t + deltaInt * divisorInt;
                if (nextRemainder < Int128.Zero || nextRemainder >= divisorInt)
                {
                    state = CreatePrimeState(step, divisor, mersenne, divisorData);
                    _primeStates[prime] = state;
                    hit = state.Remainder == 0UL ? (byte)1 : (byte)0;
                    return true;
                }

                quotient -= delta;
                remainder = (ulong)nextRemainder;
                currentDivisor = nextDivisor;
            }

            ulong residue = ComputeResidueFromRemainder(remainder, divisor);
            state.LastDivisor = divisor;
            state.Quotient = quotient;
            state.Remainder = remainder;
            state.MontgomeryResidue = ToMontgomeryResidue(residue, divisorData);
            // Reuse state to persist the updated quotient and remainder tuple for the next divisor.
            _primeStates[prime] = state;

            hit = remainder == 0UL ? (byte)1 : (byte)0;
            return true;
        }

        private static PrimeDivisorState CreatePrimeState(ulong step, ulong divisor, ulong mersenne, in MontgomeryDivisorData divisorData)
        {
            PrimeDivisorState state = new()
            {
                Step = step,
                LastDivisor = divisor,
                HasState = true,
            };

            ulong quotient = mersenne / divisor;
            ulong remainder = mersenne % divisor;
            state.Quotient = quotient;
            state.Remainder = remainder;
            ulong residue = ComputeResidueFromRemainder(remainder, divisor);
            state.MontgomeryResidue = ToMontgomeryResidue(residue, divisorData);
            return state;
        }

        private static bool TryGetMersenneMinusOne(ulong exponent, out ulong mersenne)
        {
            if (exponent <= 63UL)
            {
                mersenne = (1UL << (int)exponent) - 1UL;
                return true;
            }

            if (exponent == 64UL)
            {
                mersenne = ulong.MaxValue;
                return true;
            }

            mersenne = 0UL;
            return false;
        }

        private static ulong ComputeResidueFromRemainder(ulong remainder, ulong divisor)
        {
            ulong residue = remainder + 1UL;
            if (residue >= divisor)
            {
                residue -= divisor;
            }

            return residue;
        }

        private static ulong ToMontgomeryResidue(ulong value, in MontgomeryDivisorData divisorData)
        {
            ulong modulus = divisorData.Modulus;
            if (modulus <= 1UL)
            {
                return 0UL;
            }

            value %= modulus;
            UInt128 shifted = (UInt128)value << 64;
            return (ulong)(shifted % modulus);
        }

        private struct PrimeDivisorState
        {
            public ulong Step;
            public ulong LastDivisor;
            public ulong Quotient;
            public ulong Remainder;
            public ulong MontgomeryResidue;
            public bool HasState;
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


