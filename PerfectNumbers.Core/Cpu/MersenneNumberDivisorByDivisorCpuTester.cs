using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core.Cpu;

public sealed class MersenneNumberDivisorByDivisorCpuTester : IMersenneNumberDivisorByDivisorTester
{
    private readonly object _sync = new();

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
            // EvenPerfectBitScanner routes primes below the small-divisor cutoff to the GPU path, so the CPU path still sees
            // trivial candidates during targeted tests. Short-circuit here to keep those runs aligned with the production flow.
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
        MersenneCpuDivisorScanSession? session = ThreadStaticPools.RentMersenneCpuDivisorSession();
        if (session is not null)
        {
            return session;
        }

        return new MersenneCpuDivisorScanSession();
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
                    Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>? cacheForCycle = requireLargeCycleCache ? factorCache : null;
                    if (!MersenneDivisorCycles.TryCalculateCycleLengthForExponent(
                            candidate,
                            prime,
                            divisorData,
                            cacheForCycle,
                            cacheForCycle is not null,
                            out ulong computedCycle) || computedCycle == 0UL)
                    {
                        divisorCycle = cycleCache.GetCycleLength(candidate);
                    }
                    else
                    {
                        divisorCycle = computedCycle;
                    }
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


}
