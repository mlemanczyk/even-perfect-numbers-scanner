using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Numerics;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core.Cpu;

public sealed class MersenneNumberDivisorByDivisorCpuTester : IMersenneNumberDivisorByDivisorTester
{
    private readonly object _sync = new();
    private readonly ConcurrentBag<DivisorScanSession> _sessionPool = new();
    private ulong _divisorLimit;
    private ulong _lastStatusDivisor;
    private bool _isConfigured;
    private int _batchSize = 1_024;

    private ulong _configuredMaxPrime;
    private ByDivisorDeltasDevice _deltasDevice = ByDivisorDeltasDevice.Cpu;
    private ByDivisorMontgomeryDevice _montgomeryDevice = ByDivisorMontgomeryDevice.Cpu;

    public int BatchSize
    {
        get => _batchSize;
        set
        {
            int sanitized = Math.Max(1, value);
            _batchSize = sanitized;
        }
    }

    public ByDivisorDeltasDevice DeltasDevice
    {
        get => _deltasDevice;
        set
        {
            lock (_sync)
            {
                _deltasDevice = value;
            }
        }
    }

    public ByDivisorMontgomeryDevice MontgomeryDevice
    {
        get => _montgomeryDevice;
        set
        {
            lock (_sync)
            {
                _montgomeryDevice = value;
            }
        }
    }

    public void ConfigureFromMaxPrime(ulong maxPrime)
    {
        lock (_sync)
        {
            _configuredMaxPrime = maxPrime;
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
        if (_deltasDevice == ByDivisorDeltasDevice.Cpu)
        {
            return CheckDivisorsCpu(prime, allowedMax, out lastProcessed, out processedAll, out processedCount, timeLimit);
        }

        return CheckDivisorsGpu(prime, allowedMax, out lastProcessed, out processedAll, out processedCount, timeLimit);
    }

    private bool CheckDivisorsCpu(
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
                // Each divisor appears only once in this monotonically increasing sequence, so
                // rebuilding the Montgomery data per candidate avoids caching entries that we
                // would never hit again.
                MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(candidate);
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


    private bool CheckDivisorsGpu(
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

        GpuUInt128 step128 = new GpuUInt128(prime);
        step128.ShiftLeft(1);
        if (step128.IsZero)
        {
            processedAll = true;
            return false;
        }

        GpuUInt128 limit128 = new GpuUInt128(allowedMax);
        GpuUInt128 divisor128 = step128;
        divisor128.Add(1UL);
        if (divisor128.CompareTo(limit128) > 0)
        {
            processedAll = true;
            return false;
        }

        ulong step = step128.Low;
        ulong limit64 = allowedMax;

        int batchCapacity = Math.Max(1, _batchSize);
        DivisorCycleCache cycleCache = DivisorCycleCache.Shared;
        int preferredCycleBatch = cycleCache.PreferredBatchSize;
        if (preferredCycleBatch > 0 && preferredCycleBatch < batchCapacity)
        {
            batchCapacity = preferredCycleBatch;
        }

        var candidateEvaluator = new MersenneNumberDivisorCandidateGpuEvaluator(batchCapacity);

        ulong[] candidateBuffer = ArrayPool<ulong>.Shared.Rent(batchCapacity);
        byte[] maskBuffer = ArrayPool<byte>.Shared.Rent(batchCapacity);

        Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>? factorCache = null;

        const int RemainderTableSize = 6;
        byte[] remainderBuffer = ArrayPool<byte>.Shared.Rent(RemainderTableSize);
        byte[] stepBuffer = ArrayPool<byte>.Shared.Rent(RemainderTableSize);
        Span<byte> remainderSpan = remainderBuffer.AsSpan(0, RemainderTableSize);
        Span<byte> stepSpan = stepBuffer.AsSpan(0, RemainderTableSize);
        // The remainder and step tables follow the fixed modulus order: 10, 8, 5, 3, 7, 11.

        ulong startDivisor64 = divisor128.Low;
        remainderSpan[0] = (byte)(startDivisor64 % 10UL);
        remainderSpan[1] = (byte)(startDivisor64 % 8UL);
        remainderSpan[2] = (byte)(startDivisor64 % 5UL);
        remainderSpan[3] = (byte)(startDivisor64 % 3UL);
        remainderSpan[4] = (byte)(startDivisor64 % 7UL);
        remainderSpan[5] = (byte)(startDivisor64 % 11UL);

        stepSpan[0] = (byte)(step % 10UL);
        stepSpan[1] = (byte)(step % 8UL);
        stepSpan[2] = (byte)(step % 5UL);
        stepSpan[3] = (byte)(step % 3UL);
        stepSpan[4] = (byte)(step % 7UL);
        stepSpan[5] = (byte)(step % 11UL);

        bool lastIsSeven = (prime & 3UL) == 3UL;

        MersenneNumberDivisorRemainderGpuStepper remainderStepper = new(stepSpan);
        MersenneNumberDivisorMontgomeryGpuBuilder? montgomeryBuilder = _montgomeryDevice == ByDivisorMontgomeryDevice.Gpu
            ? new MersenneNumberDivisorMontgomeryGpuBuilder(batchCapacity)
            : null;
        MontgomeryDivisorData[]? montgomeryBuffer = montgomeryBuilder != null
            ? ArrayPool<MontgomeryDivisorData>.Shared.Rent(batchCapacity)
            : null;

        ulong remainingCandidates = ComputeRemainingCandidateCount(startDivisor64, limit64, step);

        try
        {
            while (remainingCandidates > 0UL && divisor128.CompareTo(limit128) <= 0)
            {
                if (divisor128.High != 0UL)
                {
                    break;
                }

                ulong currentDivisor = divisor128.Low;
                int chunkCount = remainingCandidates > (ulong)batchCapacity
                    ? batchCapacity
                    : (int)remainingCandidates;
                if (chunkCount <= 0)
                {
                    break;
                }

                Span<ulong> candidateSpan = candidateBuffer.AsSpan(0, chunkCount);
                Span<byte> maskSpan = maskBuffer.AsSpan(0, chunkCount);

                candidateEvaluator.EvaluateCandidates(
                    currentDivisor,
                    step,
                    limit64,
                    remainderSpan,
                    stepSpan,
                    lastIsSeven,
                    candidateSpan,
                    maskSpan);

                Span<MontgomeryDivisorData> montgomerySpan = default;
                if (montgomeryBuilder != null && montgomeryBuffer != null)
                {
                    montgomerySpan = montgomeryBuffer.AsSpan(0, chunkCount);
                    // Every candidate divisor is unique across the scan, so build Montgomery data
                    // for the current batch and drop it immediately instead of maintaining a cache.
                    montgomeryBuilder.Build(candidateSpan, montgomerySpan);
                }

                for (int i = 0; i < chunkCount; i++)
                {
                    if (enforceLimit && (iterationCounter == 0UL || (iterationCounter & 63UL) == 0UL) && limitGuard.HasExpired())
                    {
                        processedAll = false;
                        return false;
                    }

                    iterationCounter++;

                    ulong candidate = candidateSpan[i];
                    processedCount++;
                    lastProcessed = candidate;

                    if (maskSpan[i] == 0)
                    {
                        continue;
                    }

                    MontgomeryDivisorData divisorData;
                    if (!montgomerySpan.IsEmpty)
                    {
                        divisorData = montgomerySpan[i];
                        if (divisorData.Modulus != candidate)
                        {
                            // The GPU builder might skip entries when masks zero them out; recompute on the CPU
                            // without caching because divisors never repeat within a session.
                            divisorData = MontgomeryDivisorData.FromModulus(candidate);
                        }
                    }
                    else
                    {
                        // Candidates remain unique even when GPU Montgomery support is disabled, so rebuild
                        // the divisor data inline instead of persisting unused cache entries.
                        divisorData = MontgomeryDivisorData.FromModulus(candidate);
                    }

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
                        processedAll = true;
                        return true;
                    }

                    if (divisorCycle == 0UL)
                    {
                        Console.WriteLine($"Divisor cycle was not calculated for {prime}");
                    }
                }

                GpuUInt128 increment = step128;
                increment.Mul((ulong)chunkCount);
                divisor128.Add(increment);

                remainderStepper.Advance(chunkCount, remainderSpan);
                remainingCandidates -= (ulong)chunkCount;
            }
        }
        finally
        {
            montgomeryBuilder?.Dispose();
            if (montgomeryBuffer != null)
            {
                ArrayPool<MontgomeryDivisorData>.Shared.Return(montgomeryBuffer, clearArray: false);
            }

            remainderStepper.Dispose();
            ArrayPool<byte>.Shared.Return(remainderBuffer, clearArray: false);
            ArrayPool<byte>.Shared.Return(stepBuffer, clearArray: false);
            ArrayPool<ulong>.Shared.Return(candidateBuffer, clearArray: false);
            ArrayPool<byte>.Shared.Return(maskBuffer, clearArray: false);
            candidateEvaluator.Dispose();
        }

        processedAll = remainingCandidates == 0UL && divisor128.CompareTo(limit128) > 0;
        return false;
    }

    private static ulong ComputeRemainingCandidateCount(ulong currentDivisor, ulong limit, ulong step)
    {
        if (step == 0UL || currentDivisor > limit)
        {
            return 0UL;
        }

        ulong distance = limit - currentDivisor;
        ulong stepsAvailable = distance / step;
        return stepsAvailable + 1UL;
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
        private ulong[]? _primeDeltas;
        private ulong[]? _cycleRemainders;
        private ulong[]? _residues;
        private int _bufferCapacity;
        private MersenneNumberDivisorResidueGpuEvaluator? _gpuResidueEvaluator;

        internal DivisorScanSession(MersenneNumberDivisorByDivisorCpuTester owner)
        {
            _owner = owner;
        }

        internal void Reset()
        {
            _disposed = false;
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

            MontgomeryDivisorData effectiveDivisorData = divisorData;
            if (effectiveDivisorData.Modulus != divisor)
            {
                // Each divisor surfaces only once across the monotonic scan, so rebuild the Montgomery data
                // instead of retaining a cache entry that would never be reused within future checks.
                effectiveDivisorData = MontgomeryDivisorData.FromModulus(divisor);
            }

            ulong modulus = effectiveDivisorData.Modulus;
            if (modulus <= 1UL || (modulus & 1UL) == 0UL)
            {
                hits.Clear();
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

            if (_owner._deltasDevice == ByDivisorDeltasDevice.Cpu)
            {
                CheckDivisorCpu(effectiveDivisorData, divisorCycle, primes, hits);
                return;
            }

            CheckDivisorGpu(effectiveDivisorData, divisorCycle, primes, hits, length);
        }

        private static void CheckDivisorCpu(
            in MontgomeryDivisorData divisorData,
            ulong divisorCycle,
            ReadOnlySpan<ulong> primes,
            Span<byte> hits)
        {
            // Keep these remainder steppers in place so future updates continue reusing the previously computed residues.
            // They are critical for avoiding repeated full Montgomery exponentiation work when scanning divisors.
            var exponentStepper = new ExponentRemainderStepper(divisorData);
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

            int length = primes.Length;
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

        private void CheckDivisorGpu(
            in MontgomeryDivisorData divisorData,
            ulong divisorCycle,
            ReadOnlySpan<ulong> primes,
            Span<byte> hits,
            int length)
        {
            EnsureCapacity(length);

            Span<ulong> deltaSpan = _primeDeltas!.AsSpan(0, length);
            ComputePrimeDeltas(primes, deltaSpan);

            Span<ulong> remainderSpan = _cycleRemainders!.AsSpan(0, length);
            ComputeCycleRemainders(divisorCycle, primes, deltaSpan, remainderSpan);

            EnsureGpuEvaluator();
            Span<ulong> residueSpan = _residues!.AsSpan(0, length);
            _gpuResidueEvaluator!.ComputeResidues(primes, divisorData, residueSpan);

            for (int i = 0; i < length; i++)
            {
                hits[i] = remainderSpan[i] == 0UL && residueSpan[i] == 1UL ? (byte)1 : (byte)0;
            }
        }

        private void EnsureGpuEvaluator()
        {
            if (_gpuResidueEvaluator is null)
            {
                _gpuResidueEvaluator = new MersenneNumberDivisorResidueGpuEvaluator(_owner._batchSize);
            }
        }

        private void ReleaseGpuEvaluator()
        {
            if (_gpuResidueEvaluator is null)
            {
                return;
            }

            _gpuResidueEvaluator.Dispose();
            _gpuResidueEvaluator = null;
        }

        private void EnsureCapacity(int requiredLength)
        {
            if (requiredLength <= 0)
            {
                return;
            }

            if (requiredLength <= _bufferCapacity && _primeDeltas is not null && _cycleRemainders is not null && _residues is not null)
            {
                return;
            }

            int newCapacity = _bufferCapacity;
            if (newCapacity == 0)
            {
                newCapacity = 1;
            }

            while (newCapacity < requiredLength)
            {
                newCapacity <<= 1;
            }

            ReturnBuffers();

            _primeDeltas = ArrayPool<ulong>.Shared.Rent(newCapacity);
            _cycleRemainders = ArrayPool<ulong>.Shared.Rent(newCapacity);
            _residues = ArrayPool<ulong>.Shared.Rent(newCapacity);
            _bufferCapacity = newCapacity;
        }

        private void ReturnBuffers()
        {
            if (_primeDeltas is not null)
            {
                ArrayPool<ulong>.Shared.Return(_primeDeltas, clearArray: false);
                _primeDeltas = null;
            }

            if (_cycleRemainders is not null)
            {
                ArrayPool<ulong>.Shared.Return(_cycleRemainders, clearArray: false);
                _cycleRemainders = null;
            }

            if (_residues is not null)
            {
                ArrayPool<ulong>.Shared.Return(_residues, clearArray: false);
                _residues = null;
            }

            _bufferCapacity = 0;
        }

        private static void ComputePrimeDeltas(ReadOnlySpan<ulong> primes, Span<ulong> deltas)
        {
            if (primes.Length == 0)
            {
                return;
            }

            ulong previous = primes[0];
            deltas[0] = previous;

            for (int i = 1; i < primes.Length; i++)
            {
                ulong current = primes[i];
                if (current <= previous)
                {
                    throw new ArgumentOutOfRangeException(nameof(primes), "Primes must be strictly increasing.");
                }

                deltas[i] = current - previous;
                previous = current;
            }
        }

        private static void ComputeCycleRemainders(ulong cycleLength, ReadOnlySpan<ulong> primes, ReadOnlySpan<ulong> deltas, Span<ulong> remainders)
        {
            if (cycleLength == 0UL || primes.Length == 0)
            {
                remainders.Clear();
                return;
            }

            ulong remainder = primes[0] % cycleLength;
            remainders[0] = remainder;

            for (int i = 1; i < primes.Length; i++)
            {
                remainder = AddMod(remainder, deltas[i], cycleLength);
                remainders[i] = remainder;
            }
        }

        private static ulong AddMod(ulong value, ulong delta, ulong modulus)
        {
            if (modulus == 0UL)
            {
                return 0UL;
            }

            if (ulong.MaxValue - value < delta)
            {
                UInt128 extended = (UInt128)value + delta;
                return (ulong)(extended % modulus);
            }

            ulong sum = value + delta;
            if (sum >= modulus)
            {
                sum -= modulus;
                if (sum >= modulus)
                {
                    sum %= modulus;
                }
            }

            return sum;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            ReleaseGpuEvaluator();
            _owner.ReturnSession(this);
        }
    }
}


