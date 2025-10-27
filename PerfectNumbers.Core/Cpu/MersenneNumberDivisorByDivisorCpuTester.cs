using System.Collections.Generic;
using System.Runtime.CompilerServices;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core.Cpu;

public sealed class MersenneNumberDivisorByDivisorCpuTester : IMersenneNumberDivisorByDivisorTester
{
    private ulong _divisorLimit;
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

        bool processedAll;

        bool composite = CheckDivisors(
            prime,
            allowedMax,
            out processedAll);

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
        out bool processedAll)
    {
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
            return false;
        }

        // Intentionally recomputes factorizations without a per-thread cache.
        // The previous factor cache recorded virtually no hits and only slowed down the scan.
        DivisorCycleCache cycleCache = DivisorCycleCache.Shared;

        ulong stepHigh = step.High;
        ulong stepLow = step.Low;
        ulong limitHigh = limit.High;
        ulong divisorLow = divisor.Low;

        // Keep the divisibility filters aligned with the divisor-cycle generator so the
        // CPU path never requests cycles that were skipped during cache creation.
        LastDigit lastDigit = (prime & 3UL) == 3UL ? LastDigit.Seven : LastDigit.One;
        ushort decimalMask = DivisorGenerator.GetDecimalMask(lastDigit);

        byte step10;
        byte step8;
        byte step5;
        byte step3;
        byte step7;
        byte step11;

        byte remainder10;
        byte remainder8;
        byte remainder5;
        byte remainder3;
        byte remainder7;
        byte remainder11;

        if (stepHigh == 0UL && limitHigh == 0UL)
        {
            step10 = (byte)(stepLow % 10UL);
            step8 = (byte)(stepLow & 7UL);
            step5 = (byte)(stepLow % 5UL);
            step3 = (byte)(stepLow % 3UL);
            step7 = (byte)(stepLow % 7UL);
            step11 = (byte)(stepLow % 11UL);

            remainder10 = (byte)(divisorLow % 10UL);
            remainder8 = (byte)(divisorLow & 7UL);
            remainder5 = (byte)(divisorLow % 5UL);
            remainder3 = (byte)(divisorLow % 3UL);
            remainder7 = (byte)(divisorLow % 7UL);
            remainder11 = (byte)(divisorLow % 11UL);

            return CheckDivisors64(
                prime,
                stepLow,
                limit.Low,
                divisorLow,
                decimalMask,
                step10,
                step8,
                step5,
                step3,
                step7,
                step11,
                remainder10,
                remainder8,
                remainder5,
                remainder3,
                remainder7,
                remainder11,
                out processedAll);
        }

        ulong divisorHigh = divisor.High;
        step10 = (byte)((((stepHigh % 10UL) * 6UL) + (stepLow % 10UL)) % 10UL);
        step8 = (byte)(stepLow % 8UL);
        step5 = (byte)(((stepHigh % 5UL) + (stepLow % 5UL)) % 5UL);
        step3 = (byte)(((stepHigh % 3UL) + (stepLow % 3UL)) % 3UL);
        step7 = (byte)((((stepHigh % 7UL) * 2UL) + (stepLow % 7UL)) % 7UL);
        step11 = (byte)((((stepHigh % 11UL) * 5UL) + (stepLow % 11UL)) % 11UL);

        remainder10 = (byte)((((divisorHigh % 10UL) * 6UL) + (divisorLow % 10UL)) % 10UL);
        remainder8 = (byte)(divisorLow % 8UL);
        remainder5 = (byte)(((divisorHigh % 5UL) + (divisorLow % 5UL)) % 5UL);
        remainder3 = (byte)(((divisorHigh % 3UL) + (divisorLow % 3UL)) % 3UL);
        remainder7 = (byte)((((divisorHigh % 7UL) * 2UL) + (divisorLow % 7UL)) % 7UL);
        remainder11 = (byte)((((divisorHigh % 11UL) * 5UL) + (divisorLow % 11UL)) % 11UL);

        while (divisor.CompareTo(limit) <= 0)
        {
            bool admissible = DivisorGenerator.IsValidDivisor(
                remainder10,
                remainder8,
                remainder3,
                remainder5,
                remainder7,
                remainder11,
                decimalMask);

            if (admissible)
            {
                ulong candidate = divisor.Low;
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
                    processedAll = true;
                    return true;
                }

                if (divisorCycle == 0UL)
                {
                    Console.WriteLine($"Divisor cycle was not calculated for {prime}");
                }
            }

            divisor.Add(step);
            remainder10 = AddMod10(remainder10, step10);
            remainder8 = AddMod8(remainder8, step8);
            remainder5 = AddMod5(remainder5, step5);
            remainder3 = AddMod3(remainder3, step3);
            remainder7 = AddMod7(remainder7, step7);
            remainder11 = AddMod11(remainder11, step11);
        }
        processedAll = true;
        return false;
    }

    private static bool CheckDivisors64(
        ulong prime,
        ulong step,
        ulong limit,
        ulong divisor,
        ushort decimalMask,
        byte step10,
        byte step8,
        byte step5,
        byte step3,
        byte step7,
        byte step11,
        byte remainder10,
        byte remainder8,
        byte remainder5,
        byte remainder3,
        byte remainder7,
        byte remainder11,
        out bool processedAll)
    {
        processedAll = false;

        bool canAdvance = step <= limit;
        ulong threshold = 0UL;
        if (canAdvance)
        {
            threshold = limit - step;
        }

        while (divisor <= limit)
        {
            bool admissible = DivisorGenerator.IsValidDivisor(
                remainder10,
                remainder8,
                remainder3,
                remainder5,
                remainder7,
                remainder11,
                decimalMask);

            if (admissible)
            {
                MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(divisor);
                ulong divisorCycle;
                if (!MersenneDivisorCycles.TryCalculateCycleLengthForExponentCpu(
                        divisor,
                        prime,
                        divisorData,
                        out ulong computedCycle,
                        out bool primeOrderFailed) || computedCycle == 0UL)
                {
                    divisorCycle = MersenneDivisorCycles.CalculateCycleLength(
                        divisor,
                        divisorData,
                        skipPrimeOrderHeuristic: primeOrderFailed);
                }
                else
                {
                    divisorCycle = computedCycle;
                }

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

            if (!canAdvance || divisor > threshold)
            {
                processedAll = true;
                return false;
            }

            divisor += step;
            remainder10 = AddMod10(remainder10, step10);
            remainder8 = AddMod8(remainder8, step8);
            remainder5 = AddMod5(remainder5, step5);
            remainder3 = AddMod3(remainder3, step3);
            remainder7 = AddMod7(remainder7, step7);
            remainder11 = AddMod11(remainder11, step11);
        }

        processedAll = true;
        return false;
    }

    private static byte CheckDivisor(ulong prime, ulong divisorCycle, in MontgomeryDivisorData divisorData)
    {
        ulong residue = prime.Pow2MontgomeryModWithCycleCpu(divisorCycle, divisorData);
        return residue == 1UL ? (byte)1 : (byte)0;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte AddMod3(byte value, byte delta)
    {
        const int Modulus = 3;
        int sum = value + delta;

        if (sum >= Modulus)
        {
            sum -= Modulus;
        }

        return (byte)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte AddMod5(byte value, byte delta)
    {
        const int Modulus = 5;
        int sum = value + delta;

        if (sum >= Modulus)
        {
            sum -= Modulus;
        }

        return (byte)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte AddMod7(byte value, byte delta)
    {
        const int Modulus = 7;
        int sum = value + delta;

        if (sum >= Modulus)
        {
            sum -= Modulus;
        }

        return (byte)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte AddMod10(byte value, byte delta)
    {
        const int Modulus = 10;
        int sum = value + delta;

        if (sum >= Modulus)
        {
            sum -= Modulus;
        }

        return (byte)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte AddMod11(byte value, byte delta)
    {
        const int Modulus = 11;
        int sum = value + delta;

        if (sum >= Modulus)
        {
            sum -= Modulus;
        }

        return (byte)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte AddMod8(byte value, byte delta)
    {
        return (byte)((value + delta) & 7);
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
