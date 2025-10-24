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

    [ThreadStatic]
    private static GpuUInt128WorkSet _divisorScanGpuWorkSet;

    private struct GpuUInt128WorkSet
    {
        public GpuUInt128 Step;
        public GpuUInt128 Divisor;
        public GpuUInt128 Limit;
    }

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
        processedCount = 0UL;
        processedAll = false;

        // The EvenPerfectBitScanner feeds primes >= 138,000,000 here, so allowedMax >= 3 in production runs.
        // Keeping the guard commented out documents the reasoning for benchmarks and tests.
        // if (allowedMax < 3UL)
        // {
        //     return false;
        // }

        ref GpuUInt128WorkSet workSet = ref _divisorScanGpuWorkSet;

        ref GpuUInt128 step = ref workSet.Step;
        step.High = 0UL;
        step.Low = prime;
        step.ShiftLeft(1);

        ref GpuUInt128 limit = ref workSet.Limit;
        limit.High = 0UL;
        limit.Low = allowedMax;

        ref GpuUInt128 divisor = ref workSet.Divisor;
        divisor.High = step.High;
        divisor.Low = step.Low;
        divisor.Add(1UL);
        if (divisor.CompareTo(limit) > 0)
        {
            processedAll = true;
            divisor.High = 0UL;
            divisor.Low = 0UL;
            step.High = 0UL;
            step.Low = 0UL;
            limit.High = 0UL;
            limit.Low = 0UL;
            return false;
        }

        // Intentionally recomputes factorizations without a per-thread cache.
        // The previous factor cache recorded virtually no hits and only slowed down the scan.
        DivisorCycleCache cycleCache = DivisorCycleCache.Shared;

        ulong stepHigh = step.High;
        ulong stepLow = step.Low;
        byte step10 = (byte)((((stepHigh % 10UL) * 6UL) + (stepLow % 10UL)) % 10UL);
        byte step8 = (byte)(stepLow % 8UL);
        byte step5 = (byte)(((stepHigh % 5UL) + (stepLow % 5UL)) % 5UL);
        byte step3 = (byte)(((stepHigh % 3UL) + (stepLow % 3UL)) % 3UL);
        byte step7 = (byte)((((stepHigh % 7UL) * 2UL) + (stepLow % 7UL)) % 7UL);
        byte step11 = (byte)((((stepHigh % 11UL) * 5UL) + (stepLow % 11UL)) % 11UL);

        ulong divisorHigh = divisor.High;
        ulong divisorLow = divisor.Low;
        byte remainder10 = (byte)((((divisorHigh % 10UL) * 6UL) + (divisorLow % 10UL)) % 10UL);
        byte remainder8 = (byte)(divisorLow % 8UL);
        byte remainder5 = (byte)(((divisorHigh % 5UL) + (divisorLow % 5UL)) % 5UL);
        byte remainder3 = (byte)(((divisorHigh % 3UL) + (divisorLow % 3UL)) % 3UL);
        byte remainder7 = (byte)((((divisorHigh % 7UL) * 2UL) + (divisorLow % 7UL)) % 7UL);
        byte remainder11 = (byte)((((divisorHigh % 11UL) * 5UL) + (divisorLow % 11UL)) % 11UL);

        // Keep the divisibility filters aligned with the divisor-cycle generator so the
        // CPU path never requests cycles that were skipped during cache creation.
        bool lastIsSeven = (prime & 3UL) == 3UL;
        const ushort DecimalMaskWhenLastIsSeven = (1 << 3) | (1 << 7) | (1 << 9);
        const ushort DecimalMaskOtherwise = (1 << 1) | (1 << 3) | (1 << 9);
        const byte AcceptableRemainder8Mask = 0b10000010;
        const int NonZeroMod3Mask = 0b00000110;
        const int NonZeroMod5Mask = 0b00011110;
        const int NonZeroMod7Mask = 0b01111110;
        const int NonZeroMod11Mask = 0b11111111110;
        const int RequiredSmallPrimeMask = 0b1111;

        while (divisor.CompareTo(limit) <= 0)
        {
            ulong candidate = divisor.Low;
            processedCount++;

            ushort decimalMask = lastIsSeven ? DecimalMaskWhenLastIsSeven : DecimalMaskOtherwise;
            bool admissible = ((decimalMask >> remainder10) & 1) != 0;

            bool passesOctalMask = ((AcceptableRemainder8Mask >> remainder8) & 1) != 0;
            int smallPrimeBits = ((NonZeroMod3Mask >> remainder3) & 1)
                | (((NonZeroMod5Mask >> remainder5) & 1) << 1)
                | (((NonZeroMod7Mask >> remainder7) & 1) << 2)
                | (((NonZeroMod11Mask >> remainder11) & 1) << 3);
            bool passesSmallPrimeMasks = (smallPrimeBits & RequiredSmallPrimeMask) == RequiredSmallPrimeMask;

            if (admissible && passesOctalMask && passesSmallPrimeMasks)
            {
                MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(candidate);
                ulong divisorCycle;
                // Divisors generated from 2 * k * p + 1 exceed the small-cycle snapshot when p >= 138,000,000, so the short path below never runs.
                // if (candidate <= PerfectNumberConstants.MaxQForDivisorCycles)
                // {
                //     divisorCycle = cycleCache.GetCycleLength(candidate);
                // }
                // else
                {
                    if (!MersenneDivisorCycles.TryCalculateCycleLengthForExponentCpu(
                            candidate,
                            prime,
                            divisorData,
                            out ulong computedCycle,
                            out bool primeOrderFailed) || computedCycle == 0UL)
                    {
                        // Divisors produced by 2 * k * p + 1 always exceed PerfectNumberConstants.MaxQForDivisorCycles
                        // for the exponents scanned here, so skip the unused cache fallback and compute directly.
                        divisorCycle = MersenneDivisorCycles.CalculateCycleLength(
                            candidate,
                            divisorData,
                            skipPrimeOrderHeuristic: primeOrderFailed);
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
                    divisor.High = 0UL;
                    divisor.Low = 0UL;
                    step.High = 0UL;
                    step.Low = 0UL;
                    limit.High = 0UL;
                    limit.Low = 0UL;
                    processedAll = true;
                    return true;
                }

                if (divisorCycle == 0UL)
                {
                    Console.WriteLine($"Divisor cycle was not calculated for {prime}");
                }
            }

            divisor.Add(step);
            remainder10 = AddMod(remainder10, step10, (byte)10);
            remainder8 = AddMod(remainder8, step8, (byte)8);
            remainder5 = AddMod(remainder5, step5, (byte)5);
            remainder3 = AddMod(remainder3, step3, (byte)3);
            remainder7 = AddMod(remainder7, step7, (byte)7);
            remainder11 = AddMod(remainder11, step11, (byte)11);
        }

        processedAll = divisor.CompareTo(limit) > 0;
        divisor.High = 0UL;
        divisor.Low = 0UL;
        step.High = 0UL;
        step.Low = 0UL;
        limit.High = 0UL;
        limit.Low = 0UL;
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
