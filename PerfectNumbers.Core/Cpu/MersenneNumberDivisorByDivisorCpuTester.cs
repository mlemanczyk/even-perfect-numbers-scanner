using System.Collections.Generic;
using System.Runtime.CompilerServices;
using PerfectNumbers.Core.Gpu;

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

        // The CPU by-divisor run always hands us primes with enormous divisor limits, so the fallback below never executes.
        // if (allowedMax < 3UL)
        // {
        //     // EvenPerfectBitScanner routes primes below the small-divisor cutoff to the GPU path, so the CPU path still sees
        //     // trivial candidates during targeted tests. Short-circuit here to keep those runs aligned with the production flow.
        //     divisorsExhausted = true;
        //     return true;
        // }

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
        processedAll = false;
        processedCount = 0UL;

        // var enumerator = HeuristicPrimeTester.CreateMersenneDivisorEnumerator(prime, allowedMax);
        var enumerator = PrimeTester.CreateMersenneDivisorEnumerator(prime, allowedMax);

        // while (enumerator.TryGetNext(out HeuristicPrimeTester.HeuristicDivisorCandidate candidate))
        while (enumerator.TryGetNext(out PrimeTester.HeuristicDivisorCandidate candidate))
        {
            processedCount = enumerator.ProcessedCount;

            // HeuristicPrimeTester.HeuristicDivisorPreparation preparation = HeuristicPrimeTester.PrepareHeuristicDivisor(in candidate);
            PrimeTester.HeuristicDivisorPreparation preparation = PrimeTester.PrepareHeuristicDivisor(in candidate);
            // ulong divisorCycle = HeuristicPrimeTester.ResolveHeuristicCycleLength(
            ulong divisorCycle = PrimeTester.ResolveHeuristicCycleLength(
                prime,
                in preparation,
                out _,
                out _,
                out _);

            if (divisorCycle == prime)
            {
                processedAll = true;
                return true;
            }

            if (divisorCycle == 0UL)
            {
                Console.WriteLine($"Divisor cycle was not calculated for {prime}");
            }
        }

        processedCount = enumerator.ProcessedCount;
        processedAll = enumerator.Exhausted;
        return false;
    }

    private static byte CheckDivisor(ulong prime, ulong divisorCycle, in MontgomeryDivisorData divisorData)
    {
        ulong residue = prime.Pow2MontgomeryModWithCycleCpu(divisorCycle, divisorData);
        return residue == 1UL ? (byte)1 : (byte)0;
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
        // The by-divisor CPU configuration only supplies primes greater than 1, so the guard below never trips.
        // if (maxPrime <= 1UL)
        // {
        //     return 0UL;
        // }
        if (maxPrime - 1UL >= 64UL)
        {
            return ulong.MaxValue;
        }

        return (1UL << (int)(maxPrime - 1UL)) - 1UL;
    }

    private static ulong ComputeAllowedMaxDivisor(ulong prime, ulong divisorLimit)
    {
        // Production by-divisor scans only handle primes, so inputs never fall below 2.
        // if (prime <= 1UL)
        // {
        //     return 0UL;
        // }
        if (prime - 1UL >= 64UL)
        {
            return divisorLimit;
        }

        return Math.Min((1UL << (int)(prime - 1UL)) - 1UL, divisorLimit);
    }


}
