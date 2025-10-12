using System;
using System.Buffers;
using System.Collections.Concurrent;
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
    private MersenneNumberDivisorByDivisorGpuTester? _gpuDeltasTester;
    private bool _gpuTesterConfigured;


    public int BatchSize
    {
        get => _batchSize;
        set
        {
            int sanitized = Math.Max(1, value);
            _batchSize = sanitized;

            lock (_sync)
            {
                if (_gpuDeltasTester is not null)
                {
                    _gpuDeltasTester.GpuBatchSize = sanitized;
                }
            }
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

    public void ConfigureFromMaxPrime(ulong maxPrime)
    {
        lock (_sync)
        {
            _configuredMaxPrime = maxPrime;
            _divisorLimit = ComputeDivisorLimitFromMaxPrime(maxPrime);
            _lastStatusDivisor = 0UL;
            _isConfigured = true;
            _gpuTesterConfigured = false;
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

    internal IMersenneNumberDivisorByDivisorTester.IDivisorScanSession CreateGpuDeltasSession()
    {
        MersenneNumberDivisorByDivisorGpuTester tester;

        lock (_sync)
        {
            if (!_isConfigured)
            {
                throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
            }

            tester = _gpuDeltasTester ??= new MersenneNumberDivisorByDivisorGpuTester();

            if (!_gpuTesterConfigured)
            {
                tester.ConfigureFromMaxPrime(_configuredMaxPrime);
                _gpuTesterConfigured = true;
            }

            tester.GpuBatchSize = _batchSize;
        }

        return tester.CreateDivisorSession();
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

        int batchCapacity = Math.Max(1, _batchSize);
        DivisorCycleCache cycleCache = DivisorCycleCache.Shared;
        int preferredCycleBatch = cycleCache.PreferredBatchSize;
        if (preferredCycleBatch > 0 && preferredCycleBatch < batchCapacity)
        {
            batchCapacity = preferredCycleBatch;
        }

        ulong[] candidateBuffer = ArrayPool<ulong>.Shared.Rent(batchCapacity);
        ulong[] cycleBuffer = ArrayPool<ulong>.Shared.Rent(batchCapacity);
        MontgomeryDivisorData[] divisorDataBuffer = ArrayPool<MontgomeryDivisorData>.Shared.Rent(batchCapacity);
        byte[] remainderBuffer = ArrayPool<byte>.Shared.Rent(batchCapacity * 6);

        Span<byte> remainder10Span = remainderBuffer.AsSpan(0, batchCapacity);
        Span<byte> remainder8Span = remainderBuffer.AsSpan(batchCapacity, batchCapacity);
        Span<byte> remainder5Span = remainderBuffer.AsSpan(batchCapacity * 2, batchCapacity);
        Span<byte> remainder3Span = remainderBuffer.AsSpan(batchCapacity * 3, batchCapacity);
        Span<byte> remainder7Span = remainderBuffer.AsSpan(batchCapacity * 4, batchCapacity);
        Span<byte> remainder11Span = remainderBuffer.AsSpan(batchCapacity * 5, batchCapacity);

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

        bool lastIsSeven = (prime & 3UL) == 3UL;

        try
        {
            while (divisor <= limit)
            {
                Span<ulong> candidateSpan = candidateBuffer.AsSpan(0, batchCapacity);
                int chunkCount = 0;
                UInt128 localDivisor = divisor;

                byte localRemainder10 = remainder10;
                byte localRemainder8 = remainder8;
                byte localRemainder5 = remainder5;
                byte localRemainder3 = remainder3;
                byte localRemainder7 = remainder7;
                byte localRemainder11 = remainder11;

                while (chunkCount < batchCapacity && localDivisor <= limit)
                {
                    if (enforceLimit && (iterationCounter == 0UL || (iterationCounter & 63UL) == 0UL) && limitGuard.HasExpired())
                    {
                        processedAll = false;
                        return false;
                    }

                    ulong candidate = (ulong)localDivisor;
                    candidateSpan[chunkCount] = candidate;
                    remainder10Span[chunkCount] = localRemainder10;
                    remainder8Span[chunkCount] = localRemainder8;
                    remainder5Span[chunkCount] = localRemainder5;
                    remainder3Span[chunkCount] = localRemainder3;
                    remainder7Span[chunkCount] = localRemainder7;
                    remainder11Span[chunkCount] = localRemainder11;

                    processedCount++;
                    lastProcessed = candidate;
                    iterationCounter++;
                    chunkCount++;

                    localDivisor += step;
                    localRemainder10 = AddMod(localRemainder10, step10, (byte)10);
                    localRemainder8 = AddMod(localRemainder8, step8, (byte)8);
                    localRemainder5 = AddMod(localRemainder5, step5, (byte)5);
                    localRemainder3 = AddMod(localRemainder3, step3, (byte)3);
                    localRemainder7 = AddMod(localRemainder7, step7, (byte)7);
                    localRemainder11 = AddMod(localRemainder11, step11, (byte)11);
                }

                if (chunkCount == 0)
                {
                    break;
                }

                divisor = localDivisor;
                remainder10 = localRemainder10;
                remainder8 = localRemainder8;
                remainder5 = localRemainder5;
                remainder3 = localRemainder3;
                remainder7 = localRemainder7;
                remainder11 = localRemainder11;

                // Reuse the candidate buffer in-place so the filtered results stay contiguous without additional arrays.
                Span<ulong> chunkCandidates = candidateSpan[..chunkCount];
                int filteredCount = 0;

                for (int i = 0; i < chunkCount; i++)
                {
                    bool admissible = lastIsSeven
                        ? (remainder10Span[i] == 3 || remainder10Span[i] == 7 || remainder10Span[i] == 9)
                        : (remainder10Span[i] == 1 || remainder10Span[i] == 3 || remainder10Span[i] == 9);

                    if (!admissible)
                    {
                        continue;
                    }

                    if ((remainder8Span[i] != 1 && remainder8Span[i] != 7)
                        || remainder3Span[i] == 0
                        || remainder5Span[i] == 0
                        || remainder7Span[i] == 0
                        || remainder11Span[i] == 0)
                    {
                        continue;
                    }

                    // Store the accepted candidate at the next free slot so the front of the span becomes the filtered set.
                    chunkCandidates[filteredCount++] = chunkCandidates[i];
                }

                if (filteredCount == 0)
                {
                    continue;
                }

                Span<ulong> filteredSpan = chunkCandidates[..filteredCount];
                Span<ulong> cycleSpan = cycleBuffer.AsSpan(0, filteredCount);
                cycleCache.GetCycleLengths(filteredSpan, cycleSpan);

                int admissibleCount = 0;
                Span<MontgomeryDivisorData> divisorDataSpan = divisorDataBuffer.AsSpan(0, filteredCount);

                for (int i = 0; i < filteredCount; i++)
                {
                    if (cycleSpan[i] != prime)
                    {
                        continue;
                    }

                    ulong candidate = filteredSpan[i];
                    // Reuse the filtered span's leading segment to accumulate the admissible divisors for Montgomery checks.
                    filteredSpan[admissibleCount] = candidate;
                    divisorDataSpan[admissibleCount] = MontgomeryDivisorData.FromModulus(candidate);
                    admissibleCount++;
                }

                if (admissibleCount == 0)
                {
                    continue;
                }

                Span<ulong> admissibleSpan = filteredSpan[..admissibleCount];
                Span<MontgomeryDivisorData> admissibleDataSpan = divisorDataSpan[..admissibleCount];

                for (int i = 0; i < admissibleCount; i++)
                {
                    if (CheckDivisor(prime, prime, in admissibleDataSpan[i]) != 0)
                    {
                        lastProcessed = admissibleSpan[i];
                        processedAll = true;
                        return true;
                    }
                }
            }
        }
        finally
        {
            ArrayPool<ulong>.Shared.Return(candidateBuffer, clearArray: false);
            ArrayPool<ulong>.Shared.Return(cycleBuffer, clearArray: false);
            ArrayPool<MontgomeryDivisorData>.Shared.Return(divisorDataBuffer, clearArray: false);
            ArrayPool<byte>.Shared.Return(remainderBuffer, clearArray: false);
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
        private ulong[]? _primeDeltas;
        private ulong[]? _cycleRemainders;
        private ulong[]? _montgomeryDeltas;
        private int _bufferCapacity;
        private IMersenneNumberDivisorByDivisorTester.IDivisorScanSession? _gpuSession;

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

            ulong modulus = cachedData.Modulus;
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

            if (_owner._deltasDevice == ByDivisorDeltasDevice.Gpu)
            {
                EnsureGpuSession();
                _gpuSession!.CheckDivisor(divisor, cachedData, divisorCycle, primes, hits);
                return;
            }

            EnsureCapacity(length);

            Span<ulong> deltaSpan = _primeDeltas!.AsSpan(0, length);
            ComputePrimeDeltas(primes, deltaSpan);

            Span<ulong> remainderSpan = _cycleRemainders!.AsSpan(0, length);
            ComputeCycleRemainders(divisorCycle, primes, deltaSpan, remainderSpan);

            Span<ulong> montgomerySpan = _montgomeryDeltas!.AsSpan(0, length);
            ComputeMontgomeryDeltas(cachedData, primes, deltaSpan, montgomerySpan);

            ulong nPrime = cachedData.NPrime;
            ulong montgomeryOne = cachedData.MontgomeryOne;
            ulong currentMontgomery = montgomerySpan[0];

            hits[0] = remainderSpan[0] == 0UL && currentMontgomery == montgomeryOne ? (byte)1 : (byte)0;

            for (int i = 1; i < length; i++)
            {
                currentMontgomery = currentMontgomery.MontgomeryMultiply(montgomerySpan[i], modulus, nPrime);
                hits[i] = remainderSpan[i] == 0UL && currentMontgomery == montgomeryOne ? (byte)1 : (byte)0;
            }
        }

        private void EnsureGpuSession()
        {
            if (_gpuSession is null)
            {
                _gpuSession = _owner.CreateGpuDeltasSession();
            }
        }

        private void ReleaseGpuSession()
        {
            if (_gpuSession is null)
            {
                return;
            }

            _gpuSession.Dispose();
            _gpuSession = null;
        }

        private void EnsureCapacity(int requiredLength)
        {
            if (requiredLength <= 0)
            {
                return;
            }

            if (requiredLength <= _bufferCapacity && _primeDeltas is not null && _cycleRemainders is not null && _montgomeryDeltas is not null)
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
            _montgomeryDeltas = ArrayPool<ulong>.Shared.Rent(newCapacity);
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

            if (_montgomeryDeltas is not null)
            {
                ArrayPool<ulong>.Shared.Return(_montgomeryDeltas, clearArray: false);
                _montgomeryDeltas = null;
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

        private static void ComputeMontgomeryDeltas(in MontgomeryDivisorData divisorData, ReadOnlySpan<ulong> primes, ReadOnlySpan<ulong> deltas, Span<ulong> montgomeryValues)
        {
            if (primes.Length == 0)
            {
                montgomeryValues.Clear();
                return;
            }

            montgomeryValues[0] = primes[0].Pow2MontgomeryModWindowedCpu(divisorData, keepMontgomery: true);

            for (int i = 1; i < primes.Length; i++)
            {
                montgomeryValues[i] = deltas[i].Pow2MontgomeryModWindowedCpu(divisorData, keepMontgomery: true);
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
            ReleaseGpuSession();
            _owner.ReturnSession(this);
        }
    }
}


