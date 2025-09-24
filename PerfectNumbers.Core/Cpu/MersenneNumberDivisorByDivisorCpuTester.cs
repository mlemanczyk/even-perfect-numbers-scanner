using System;
using System.Buffers;
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
                int batchCapacity;

                lock (_sync)
                {
                        if (!_isConfigured)
                        {
                                throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
                        }

                        allowedMax = ComputeAllowedMaxDivisor(prime, _divisorLimit);
                        useCycles = _useDivisorCycles;
                        batchCapacity = _batchSize;
                }

                if (allowedMax < 3UL)
                {
                        divisorsExhausted = true;
                        return true;
                }

                ulong[] divisors = ArrayPool<ulong>.Shared.Rent(batchCapacity);
                byte[] hits = ArrayPool<byte>.Shared.Rent(batchCapacity);
                MontgomeryDivisorData[] divisorData = ArrayPool<MontgomeryDivisorData>.Shared.Rent(batchCapacity);

                ulong processedCount = 0UL;
                ulong lastProcessed = 0UL;
                bool composite = false;
                bool processedAll = false;

                try
                {
                        composite = CheckDivisors(
                                prime,
                                allowedMax,
                                useCycles,
                                divisors,
                                hits,
                                divisorData,
                                out lastProcessed,
                                out processedAll,
                                out processedCount);
                }
                finally
                {
                        ArrayPool<ulong>.Shared.Return(divisors, clearArray: false);
                        ArrayPool<byte>.Shared.Return(hits, clearArray: false);
                        ArrayPool<MontgomeryDivisorData>.Shared.Return(divisorData, clearArray: false);
                }

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
                ulong[] divisors,
                byte[] hits,
                MontgomeryDivisorData[] divisorData,
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

                int batchCapacity = _batchSize;
                ulong divisor = 3UL;
                bool composite = false;

                while (divisor <= allowedMax)
                {
                        int batchSize = 0;
                        bool reachedEndInBatch = false;

                        while (batchSize < batchCapacity && divisor <= allowedMax)
                        {
                                ulong currentDivisor = divisor;
                                ulong nextDivisor = currentDivisor + 2UL;

                                divisors[batchSize++] = currentDivisor;
                                processedCount++;
                                lastProcessed = currentDivisor;

                                if (nextDivisor <= currentDivisor)
                                {
                                        reachedEndInBatch = true;
                                        break;
                                }

                                if (nextDivisor > allowedMax)
                                {
                                        divisor = nextDivisor;
                                        reachedEndInBatch = true;
                                        break;
                                }

                                divisor = nextDivisor;
                        }

                        if (batchSize == 0)
                        {
                                if (reachedEndInBatch)
                                {
                                        processedAll = true;
                                }

                                break;
                        }

                        Span<MontgomeryDivisorData> divisorDataSpan = divisorData.AsSpan(0, batchSize);
                        Span<ulong> divisorSpan = divisors.AsSpan(0, batchSize);
                        Span<byte> hitsSpan = hits.AsSpan(0, batchSize);

                        for (int i = 0; i < batchSize; i++)
                        {
                                divisorDataSpan[i] = CreateMontgomeryDivisorData(divisorSpan[i]);
                        }

                        for (int i = 0; i < batchSize; i++)
                        {
                                hitsSpan[i] = CheckDivisor(prime, useCycles, divisorSpan[i], divisorDataSpan[i]);
                                if (hitsSpan[i] != 0)
                                {
                                        composite = true;
                                        lastProcessed = divisorSpan[i];
                                        break;
                                }
                        }

                        if (!composite)
                        {
                                lastProcessed = divisorSpan[batchSize - 1];
                        }

                        if (composite)
                        {
                                break;
                        }

                        if (reachedEndInBatch)
                        {
                                processedAll = true;
                                break;
                        }
                }

                processedAll = processedAll || divisor > allowedMax;
                return composite;
        }

        private static byte CheckDivisor(ulong prime, bool useCycles, ulong divisor, in MontgomeryDivisorData divisorData)
        {
                ulong modulus = divisorData.Modulus;
                if (modulus <= 1UL || (modulus & 1UL) == 0UL)
                {
                        return 0;
                }

                ulong exponent = prime;

                if (useCycles)
                {
                        ulong cycle = MersenneDivisorCycles.CalculateCycleLength(divisor);
                        if (cycle == 0UL)
                        {
                                return 0;
                        }

                        ulong remainder = prime % cycle;
                        if (remainder != 0UL)
                        {
                                return 0;
                        }

                        exponent = remainder;
                }

                return exponent.Pow2MontgomeryMod(divisorData) == 1UL ? (byte)1 : (byte)0;
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
                private ulong[] _primeBuffer;
                private int[] _positionBuffer;
                private ulong[] _exponentBuffer;
                private int _capacity;
                private bool _disposed;

                internal DivisorScanSession(MersenneNumberDivisorByDivisorCpuTester owner)
                {
                        _owner = owner;
                        _capacity = Math.Max(1, owner._batchSize);
                        _primeBuffer = ArrayPool<ulong>.Shared.Rent(_capacity);
                        _positionBuffer = ArrayPool<int>.Shared.Rent(_capacity);
                        _exponentBuffer = ArrayPool<ulong>.Shared.Rent(_capacity);
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

                        int primesLength = primes.Length;
                        if (primesLength == 0)
                        {
                                return;
                        }

                        MersenneNumberDivisorByDivisorCpuTester owner = _owner;
                        int batchSize = owner._batchSize;
                        EnsureCapacity(batchSize);

                        ulong modulus = divisor;
                        if (modulus <= 1UL || (modulus & 1UL) == 0UL)
                        {
                                return;
                        }

                        bool cycleEnabled = owner._useDivisorCycles && divisorCycle != 0UL;
                        ulong cycle = divisorCycle;
                        MontgomeryDivisorData divisorData = CreateMontgomeryDivisorData(divisor);

                        int offset = 0;
                        Span<ulong> primeSpan = _primeBuffer.AsSpan();
                        Span<int> positionSpan = _positionBuffer.AsSpan();
                        Span<ulong> exponentSpan = _exponentBuffer.AsSpan();
                        ulong previousResidue = 0UL;
                        ulong previousPrime = 0UL;
                        bool hasPreviousResidue = false;

                        while (offset < primesLength)
                        {
                                int sliceLength = Math.Min(batchSize, primesLength - offset);
                                ReadOnlySpan<ulong> primesSlice = primes.Slice(offset, sliceLength);
                                Span<byte> hitsSlice = hits.Slice(offset, sliceLength);
                                hitsSlice.Clear();

                                int computeCount = 0;

                                if (cycleEnabled)
                                {
                                        for (int i = 0; i < sliceLength; i++)
                                        {
                                                ulong prime = primesSlice[i];
                                                if (prime % cycle != 0UL)
                                                {
                                                        continue;
                                                }

                                                primeSpan[computeCount] = prime;
                                                positionSpan[computeCount] = i;
                                                computeCount++;
                                        }
                                }
                                else
                                {
                                        for (int i = 0; i < sliceLength; i++)
                                        {
                                                primeSpan[computeCount] = primesSlice[i];
                                                positionSpan[computeCount] = i;
                                                computeCount++;
                                        }
                                }

                                if (computeCount > 0)
                                {
                                        Span<ulong> exponentSlice = exponentSpan[..computeCount];
                                        ulong currentPrime = hasPreviousResidue ? previousPrime : 0UL;
                                        bool hasDeltaSource = hasPreviousResidue;

                                        for (int i = 0; i < computeCount; i++)
                                        {
                                                ulong prime = primeSpan[i];
                                                ulong exponentValue;
                                                if (hasDeltaSource)
                                                {
                                                        exponentValue = prime - currentPrime;
                                                }
                                                else
                                                {
                                                        exponentValue = prime;
                                                        hasDeltaSource = true;
                                                }

                                                if (cycleEnabled)
                                                {
                                                        exponentValue %= cycle;
                                                }

                                                exponentSlice[i] = exponentValue;
                                                currentPrime = prime;
                                        }

                                        for (int i = 0; i < computeCount; i++)
                                        {
                                                exponentSlice[i] = exponentSlice[i].Pow2MontgomeryMod(divisorData);
                                        }

                                        for (int i = 0; i < computeCount; i++)
                                        {
                                                int position = positionSpan[i];
                                                ulong stepResidue = exponentSlice[i];
                                                ulong prime = primeSpan[i];
                                                ulong residue = hasPreviousResidue ? MultiplyMod(previousResidue, stepResidue, modulus) : stepResidue;
                                                hitsSlice[position] = residue == 1UL ? (byte)1 : (byte)0;
                                                previousResidue = residue;
                                                previousPrime = prime;
                                                hasPreviousResidue = true;
                                        }
                                }

                                offset += sliceLength;
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

                private void EnsureCapacity(int requiredCapacity)
                {
                        if (requiredCapacity <= _capacity)
                        {
                                return;
                        }

                        ArrayPool<ulong>.Shared.Return(_primeBuffer, clearArray: false);
                        ArrayPool<int>.Shared.Return(_positionBuffer, clearArray: false);
                        ArrayPool<ulong>.Shared.Return(_exponentBuffer, clearArray: false);

                        _capacity = requiredCapacity;
                        _primeBuffer = ArrayPool<ulong>.Shared.Rent(_capacity);
                        _positionBuffer = ArrayPool<int>.Shared.Rent(_capacity);
                        _exponentBuffer = ArrayPool<ulong>.Shared.Rent(_capacity);
                }

                private static ulong MultiplyMod(ulong left, ulong right, ulong modulus)
                {
                        if (modulus == 0UL)
                        {
                                return 0UL;
                        }

                        UInt128 product = (UInt128)left * right;
                        return (ulong)(product % modulus);
                }
        }
}

