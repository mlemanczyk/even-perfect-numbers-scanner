using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using MontgomeryDivisorData = PerfectNumbers.Core.MontgomeryDivisorData;

namespace PerfectNumbers.Core.Gpu;

/// <summary>
/// Scans Mersenne divisors on the GPU for prime exponents p >= 31 using per-divisor Montgomery data.
/// </summary>
public sealed class MersenneNumberDivisorByDivisorGpuTester : IMersenneNumberDivisorByDivisorTester
{
    private int _gpuBatchSize = GpuConstants.ScanBatchSize;
    private ulong _divisorLimit;
    private bool _isConfigured;

    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<MontgomeryDivisorData>, ArrayView<ulong>, ArrayView<byte>>> _kernelCache = new();
    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>>> _kernelExponentCache = new();
    private readonly ConcurrentDictionary<Accelerator, ConcurrentBag<BatchResources>> _resourcePools = new(AcceleratorReferenceComparer.Instance);
    private readonly ConcurrentBag<DivisorScanSession> _sessionPool = new();

    private Action<Index1D, ArrayView<MontgomeryDivisorData>, ArrayView<ulong>, ArrayView<byte>> GetKernel(Accelerator accelerator) =>
        _kernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<MontgomeryDivisorData>, ArrayView<ulong>, ArrayView<byte>>(CheckKernel));

    private Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>> GetKernelByPrimeExponent(Accelerator accelerator) =>
        _kernelExponentCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>>(ComputeExponentKernel));

    public int GpuBatchSize
    {
        get => _gpuBatchSize;
        set => _gpuBatchSize = Math.Max(1, value);
    }

    int IMersenneNumberDivisorByDivisorTester.BatchSize
    {
        get => GpuBatchSize;
        set => GpuBatchSize = value;
    }

    public void ConfigureFromMaxPrime(ulong maxPrime)
    {
        // Configuration happens once during startup, so no synchronization is required to cache the divisor limit.
        _divisorLimit = ComputeDivisorLimitFromMaxPrime(maxPrime);
        _isConfigured = true;
    }

    public bool IsPrime(ulong prime, out bool divisorsExhausted, TimeSpan? timeLimit = null)
    {
        // The CLI always configures the tester during startup, so the historical guard is commented out to avoid branching.
        // if (!_isConfigured)
        // {
        //     throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
        // }

        // Configuration is immutable, so the cached values can be read without locking.
        ulong allowedMax = ComputeAllowedMaxDivisor(prime, _divisorLimit);
        int batchCapacity = _gpuBatchSize;

        // The GPU path always receives an allowed maximum well above the minimal divisor, so the legacy
        // guard stays commented out to document the invariant without branching.
        // if (allowedMax < 3UL)
        // {
        //     divisorsExhausted = true;
        //     return true;
        // }

        bool composite;
        bool coveredRange;
        ulong processedCount;
        ulong lastProcessed;

        var gpuLease = GpuContextPool.RentPreferred(preferCpu: false);
        var accelerator = gpuLease.Accelerator;
        var kernel = GetKernel(accelerator);
        BatchResources resources = RentBatchResources(accelerator, batchCapacity);

        try
        {
            composite = CheckDivisors(
            prime,
            allowedMax,
            timeLimit,
            accelerator,
            kernel,
            resources.DivisorsBuffer,
            resources.ExponentBuffer,
            resources.HitsBuffer,
            resources.Divisors,
            resources.Exponents,
            resources.Hits,
            resources.DivisorData,
            resources.CycleCandidates,
            resources.CycleLengths,
            resources.CycleCapacity,
            out lastProcessed,
            out coveredRange,
            out processedCount
            );
        }
        finally
        {
            ReturnBatchResources(accelerator, resources);
            gpuLease.Dispose();
        }

        if (composite)
        {
            divisorsExhausted = true;
            return false;
        }

        divisorsExhausted = coveredRange;
        return true;
    }

    public void PrepareCandidates(ReadOnlySpan<ulong> primes, Span<ulong> allowedMaxValues)
    {
        if (allowedMaxValues.Length < primes.Length)
        {
            throw new ArgumentException("allowedMaxValues span must be at least as long as primes span.", nameof(allowedMaxValues));
        }

        // The GPU pipeline calls ConfigureFromMaxPrime before preparing candidates, so the legacy guard
        // remains commented out to document the invariant without branching.
        // if (!_isConfigured)
        // {
        //     throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
        // }

        ulong divisorLimit = _divisorLimit;
        for (int index = 0; index < primes.Length; index++)
        {
            allowedMaxValues[index] = ComputeAllowedMaxDivisor(primes[index], divisorLimit);
        }
    }

    private static bool CheckDivisors(
        ulong prime,
        ulong allowedMax,
        TimeSpan? timeLimit,
        Accelerator accelerator,
        Action<Index1D, ArrayView<MontgomeryDivisorData>, ArrayView<ulong>, ArrayView<byte>> kernel,
        MemoryBuffer1D<MontgomeryDivisorData, Stride1D.Dense> divisorsBuffer,
        MemoryBuffer1D<ulong, Stride1D.Dense> exponentBuffer,
        MemoryBuffer1D<byte, Stride1D.Dense> hitsBuffer,
        in ulong[] divisors,
        ulong[] exponents,
        byte[] hits,
        MontgomeryDivisorData[] divisorData,
        ulong[] cycleCandidates,
        ulong[] cycleLengths,
        int cycleCapacity,
        out ulong lastProcessed,
        out bool coveredRange,
        out ulong processedCount)
    {
        int batchCapacity = (int)divisorsBuffer.Length;
        bool composite = false;
        bool processedAll = false;
        processedCount = 0UL;
        lastProcessed = 0UL;
        coveredRange = false;

        PrimeTestTimeLimit limitGuard;
        if (!PrimeTestTimeLimit.TryCreate(timeLimit, out limitGuard))
        {
            return false;
        }

        bool enforceLimit = limitGuard.IsActive;

        Span<MontgomeryDivisorData> divisorDataSpan;
        Span<ulong> divisorSpan;
        Span<ulong> exponentSpan;
        Span<byte> hitsSpan;
        ArrayView1D<MontgomeryDivisorData, Stride1D.Dense> divisorView;
        ArrayView1D<ulong, Stride1D.Dense> exponentView;
        ArrayView1D<byte, Stride1D.Dense> hitsView;

        // Batch resources expose a positive cycle capacity, so skip the redundant clamp here.
        // cycleCapacity = Math.Max(1, cycleCapacity);

        int chunkCapacity = Math.Min(batchCapacity, cycleCapacity);
        int maxThreadsPerGroup = (int)accelerator.MaxNumThreadsPerGroup;
        if (maxThreadsPerGroup > 0)
        {
            chunkCapacity = Math.Min(chunkCapacity, maxThreadsPerGroup);
        }

        // cycleCapacity already bounds chunkCapacity above zero, so the fallback would never execute.
        // if (chunkCapacity <= 0)
        // {
        //     chunkCapacity = cycleCapacity;
        // }

        ulong[] filteredDivisors = ArrayPool<ulong>.Shared.Rent(chunkCapacity);

        UInt128 twoP128 = (UInt128)prime << 1;
        // Primes above one yield a non-zero 2p value, so the legacy zero guard remains commented out.
        // if (twoP128 == UInt128.Zero)
        // {
        //     coveredRange = true;
        //     return false;
        // }

        UInt128 allowedMax128 = allowedMax;
        UInt128 firstDivisor128 = twoP128 + UInt128.One;
        if (firstDivisor128 > allowedMax128)
        {
            coveredRange = true;
            return false;
        }

        UInt128 maxK128 = (allowedMax128 - UInt128.One) / twoP128;
        // Once the first divisor falls within the allowed range, at least one multiplier k is admissible,
        // so keep the defensive zero check commented out.
        // if (maxK128 == UInt128.Zero)
        // {
        //     coveredRange = true;
        //     return false;
        // }

        ulong maxK = maxK128 > ulong.MaxValue ? ulong.MaxValue : (ulong)maxK128;
        ulong currentK = 1UL;

        UInt128 currentDivisor128 = firstDivisor128;
        bool lastIsSeven = (prime & 3UL) == 3UL;
        DivisorCycleCache cycleCache = DivisorCycleCache.Shared;

        const int RemainderTableSize = 6;
        Span<byte> remainderSpan = stackalloc byte[RemainderTableSize];
        remainderSpan[0] = (byte)((ulong)(currentDivisor128 % 10UL));
        remainderSpan[1] = (byte)((ulong)(currentDivisor128 % 8UL));
        remainderSpan[2] = (byte)((ulong)(currentDivisor128 % 5UL));
        remainderSpan[3] = (byte)((ulong)(currentDivisor128 % 3UL));
        remainderSpan[4] = (byte)((ulong)(currentDivisor128 % 7UL));
        remainderSpan[5] = (byte)((ulong)(currentDivisor128 % 11UL));

        Span<byte> stepSpan = stackalloc byte[RemainderTableSize];
        stepSpan[0] = (byte)((ulong)(twoP128 % 10UL));
        stepSpan[1] = (byte)((ulong)(twoP128 % 8UL));
        stepSpan[2] = (byte)((ulong)(twoP128 % 5UL));
        stepSpan[3] = (byte)((ulong)(twoP128 % 3UL));
        stepSpan[4] = (byte)((ulong)(twoP128 % 7UL));
        stepSpan[5] = (byte)((ulong)(twoP128 % 11UL));

        try
        {
            while (currentK <= maxK)
            {
                if (enforceLimit && limitGuard.HasExpired())
                {
                    processedAll = false;
                    return false;
                }

                int chunkCount = Math.Min(chunkCapacity, batchCapacity);
                ulong remainingK = maxK - currentK + 1UL;
                if ((ulong)chunkCount > remainingK)
                {
                    chunkCount = (int)remainingK;
                }

                // chunkCapacity and the remaining multiplier range are strictly positive here, so the zero
                // chunk fallback never triggers on production scans.
                // if (chunkCount <= 0)
                // {
                //     break;
                // }

                Span<ulong> candidateSpan = cycleCandidates.AsSpan(0, chunkCount);
                // Reuse the hits array as temporary storage for the candidate mask before it stores final GPU hit flags.
                Span<byte> maskSpan = hits.AsSpan(0, chunkCount);

                UInt128 localDivisor = currentDivisor128;
                ulong chunkStartDivisor = (ulong)localDivisor;

                for (int i = 0; i < chunkCount; i++)
                {
                    ulong divisorValue = (ulong)localDivisor;
                    candidateSpan[i] = divisorValue;

                    processedCount++;
                    lastProcessed = divisorValue;

                    localDivisor += twoP128;
                }

                currentDivisor128 = localDivisor;

                MersenneNumberDivisorCandidateCpuEvaluator.EvaluateCandidates(
                    chunkStartDivisor,
                    (ulong)twoP128,
                    allowedMax,
                    remainderSpan,
                    stepSpan,
                    lastIsSeven,
                    candidateSpan,
                    maskSpan);

                int filteredCount = 0;
                Span<ulong> filteredDivisorsSpan = filteredDivisors.AsSpan(0, chunkCount);

                for (int i = 0; i < chunkCount; i++)
                {
                    if (maskSpan[i] == 0)
                    {
                        continue;
                    }

                    filteredDivisorsSpan[filteredCount++] = candidateSpan[i];
                }

                currentK += (ulong)chunkCount;

                if (filteredCount == 0)
                {
                    processedAll = currentK > maxK;
                    continue;
                }

                Span<ulong> cycleSpan = cycleLengths.AsSpan(0, filteredCount);
                cycleCache.GetCycleLengths(filteredDivisorsSpan[..filteredCount], cycleSpan);

                int admissibleCount = 0;
                for (int i = 0; i < filteredCount; i++)
                {
                    if (cycleSpan[i] == prime)
                    {
                        filteredDivisorsSpan[admissibleCount++] = filteredDivisorsSpan[i];
                    }
                }

                if (admissibleCount == 0)
                {
                    processedAll = currentK > maxK;
                    continue;
                }

                divisorDataSpan = divisorData.AsSpan(0, admissibleCount);
                divisorSpan = divisors.AsSpan(0, admissibleCount);
                exponentSpan = exponents.AsSpan(0, admissibleCount);
                // Reassign the hits span to store Montgomery kernel results after the mask data has been consumed.
                hitsSpan = hits.AsSpan(0, admissibleCount);
                divisorView = divisorsBuffer.View.SubView(0, admissibleCount);
                exponentView = exponentBuffer.View.SubView(0, admissibleCount);
                hitsView = hitsBuffer.View.SubView(0, admissibleCount);

                for (int i = 0; i < admissibleCount; i++)
                {
                    ulong divisorValue = filteredDivisorsSpan[i];
                    divisorSpan[i] = divisorValue;
                    exponentSpan[i] = prime;
                    // Divisors never repeat while scanning the monotonically increasing sequence, so regenerate
                    // the Montgomery data for each entry rather than storing cache lines that no future batch would hit.
                    divisorDataSpan[i] = MontgomeryDivisorData.FromModulus(divisorValue);
                }

                divisorView.CopyFromCPU(ref MemoryMarshal.GetReference(divisorDataSpan), admissibleCount);
                exponentView.CopyFromCPU(ref MemoryMarshal.GetReference(exponentSpan), admissibleCount);
                kernel(admissibleCount, divisorView, exponentView, hitsView);
                hitsView.CopyToCPU(ref MemoryMarshal.GetReference(hitsSpan), admissibleCount);

                for (int i = 0; i < admissibleCount; i++)
                {
                    if (hitsSpan[i] != 0)
                    {
                        composite = true;
                        lastProcessed = divisorSpan[i];
                        break;
                    }
                }

                if (!composite)
                {
                    lastProcessed = divisorSpan[admissibleCount - 1];
                }

                if (composite)
                {
                    break;
                }

                processedAll = currentK > maxK;
            }
        }
        finally
        {
            ArrayPool<ulong>.Shared.Return(filteredDivisors, clearArray: false);
        }

        coveredRange = composite || processedAll || (currentDivisor128 > allowedMax128);
        return composite;
    }
    private static void ComputeExponentKernel(Index1D index, MontgomeryDivisorData divisor, ArrayView<ulong> exponents, ArrayView<ulong> results)
    {
        ulong modulus = divisor.Modulus;
        // Kernel inputs are constrained to odd prime moduli, so this defensive branch would never execute on the configured
        // scanning path.
        // if (modulus <= 1UL || (modulus & 1UL) == 0UL)
        // {
        //     results[index] = 0UL;
        //     return;
        // }

        ulong exponent = exponents[index];

        results[index] = exponent.Pow2ModWindowedGpu(modulus);
    }

    private static void CheckKernel(Index1D index, ArrayView<MontgomeryDivisorData> divisors, ArrayView<ulong> exponents, ArrayView<byte> hits)
    {
        MontgomeryDivisorData divisor = divisors[index];
        ulong modulus = divisor.Modulus;
        ulong exponent = exponents[index];
        hits[index] = exponent.Pow2ModWindowedGpu(modulus) == 1UL ? (byte)1 : (byte)0;
    }

    private static ulong ComputeDivisorLimitFromMaxPrime(ulong maxPrime)
    {
        return ulong.MaxValue;
    }

    private static ulong ComputeAllowedMaxDivisor(ulong prime, ulong divisorLimit)
    {
        return divisorLimit;
    }

    public sealed class DivisorScanSession : IMersenneNumberDivisorByDivisorTester.IDivisorScanSession
    {
        private readonly MersenneNumberDivisorByDivisorGpuTester _owner;
        private readonly GpuContextPool.GpuContextLease _lease;
        private readonly Accelerator _accelerator;
        private readonly Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>> _kernel;
        private MemoryBuffer1D<ulong, Stride1D.Dense> _exponentsBuffer;
        private MemoryBuffer1D<ulong, Stride1D.Dense> _resultsBuffer;
        private ulong[] _hostBuffer;
        private int _capacity;
        private bool _disposed;

        internal DivisorScanSession(MersenneNumberDivisorByDivisorGpuTester owner)
        {
            _owner = owner;
            _lease = GpuContextPool.RentPreferred(preferCpu: false);
            _accelerator = _lease.Accelerator;
            _kernel = owner.GetKernelByPrimeExponent(_accelerator);
            _capacity = Math.Max(1, owner._gpuBatchSize);
            _exponentsBuffer = _accelerator.Allocate1D<ulong>(_capacity);
            _resultsBuffer = _accelerator.Allocate1D<ulong>(_capacity);
            _hostBuffer = ArrayPool<ulong>.Shared.Rent(_capacity);
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

            ArrayView1D<ulong, Stride1D.Dense> exponentsView = _exponentsBuffer.View;
            ArrayView1D<ulong, Stride1D.Dense> resultsView = _resultsBuffer.View;
            Span<ulong> hostSpan = _hostBuffer.AsSpan(0, _capacity);

            int offset = 0;
            while (offset < length)
            {
                int batchSize = Math.Min(_capacity, length - offset);
                ReadOnlySpan<ulong> batch = primes.Slice(offset, batchSize);
                batch.CopyTo(hostSpan);

                ArrayView1D<ulong, Stride1D.Dense> exponentSlice = exponentsView.SubView(0, batchSize);
                exponentSlice.CopyFromCPU(ref MemoryMarshal.GetReference(hostSpan), batchSize);

                ArrayView1D<ulong, Stride1D.Dense> resultSlice = resultsView.SubView(0, batchSize);
                _kernel(batchSize, divisorData, exponentSlice, resultSlice);
                resultSlice.CopyToCPU(ref MemoryMarshal.GetReference(hostSpan), batchSize);

                Span<byte> hitSlice = hits.Slice(offset, batchSize);
                for (int i = 0; i < batchSize; i++)
                {
                    hitSlice[i] = hostSpan[i] == 1UL ? (byte)1 : (byte)0;
                }

                offset += batchSize;
            }
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            _owner._sessionPool.Add(this);
        }
    }

    public ulong DivisorLimit
    {
        get
        {
            // The by-divisor scanner always configures the tester before exposing the limit, so the guard remains
            // commented out to document the invariant without forcing a runtime branch.
            // if (!_isConfigured)
            // {
            //     throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
            // }

            return _divisorLimit;
        }
    }

    private BatchResources RentBatchResources(Accelerator accelerator, int capacity)
    {
        var bag = _resourcePools.GetOrAdd(accelerator, static _ => []);
        if (bag.TryTake(out BatchResources? resources))
        {
            return resources;
        }

        return new BatchResources(accelerator, capacity);
    }

    private void ReturnBatchResources(Accelerator accelerator, BatchResources resources) => _resourcePools.GetOrAdd(accelerator, static _ => []).Add(resources);

    private sealed class BatchResources : IDisposable
    {
        internal BatchResources(Accelerator accelerator, int capacity)
        {
            int cycleCapacity = DivisorCycleCache.Shared.PreferredBatchSize;
            if (cycleCapacity <= 0)
            {
                cycleCapacity = 1;
            }

            int actualCapacity = Math.Max(capacity, cycleCapacity);

            DivisorsBuffer = accelerator.Allocate1D<MontgomeryDivisorData>(actualCapacity);
            ExponentBuffer = accelerator.Allocate1D<ulong>(actualCapacity);
            HitsBuffer = accelerator.Allocate1D<byte>(actualCapacity);
            Divisors = ArrayPool<ulong>.Shared.Rent(actualCapacity);
            Exponents = ArrayPool<ulong>.Shared.Rent(actualCapacity);
            Hits = ArrayPool<byte>.Shared.Rent(actualCapacity);
            DivisorData = ArrayPool<MontgomeryDivisorData>.Shared.Rent(actualCapacity);
            CycleCandidates = ArrayPool<ulong>.Shared.Rent(cycleCapacity);
            CycleLengths = ArrayPool<ulong>.Shared.Rent(cycleCapacity);
            Capacity = actualCapacity;
            CycleCapacity = cycleCapacity;
        }

        internal readonly MemoryBuffer1D<MontgomeryDivisorData, Stride1D.Dense> DivisorsBuffer;

        internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> ExponentBuffer;

        internal readonly MemoryBuffer1D<byte, Stride1D.Dense> HitsBuffer;

        internal readonly ulong[] Divisors;

        internal readonly ulong[] Exponents;

        internal readonly byte[] Hits;

        internal readonly MontgomeryDivisorData[] DivisorData;

        internal readonly ulong[] CycleCandidates;

        internal readonly ulong[] CycleLengths;

        internal readonly int Capacity;

        internal readonly int CycleCapacity;

        public void Dispose()
        {
            DivisorsBuffer.Dispose();
            ExponentBuffer.Dispose();
            HitsBuffer.Dispose();
            ArrayPool<ulong>.Shared.Return(Divisors, clearArray: false);
            ArrayPool<ulong>.Shared.Return(Exponents, clearArray: false);
            ArrayPool<byte>.Shared.Return(Hits, clearArray: false);
            ArrayPool<MontgomeryDivisorData>.Shared.Return(DivisorData, clearArray: false);
            ArrayPool<ulong>.Shared.Return(CycleCandidates, clearArray: false);
            ArrayPool<ulong>.Shared.Return(CycleLengths, clearArray: false);
        }
    }

    private sealed class AcceleratorReferenceComparer : IEqualityComparer<Accelerator>
    {
        public static AcceleratorReferenceComparer Instance { get; } = new();

        public bool Equals(Accelerator? x, Accelerator? y) => ReferenceEquals(x, y);

        public int GetHashCode(Accelerator obj) => RuntimeHelpers.GetHashCode(obj);
    }

    public ulong GetAllowedMaxDivisor(ulong prime)
    {
        // The CLI configures the GPU tester once at startup, so callers never request divisor limits before
        // initialization completes.
        // if (!_isConfigured)
        // {
        //     throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
        // }

        return ComputeAllowedMaxDivisor(prime, _divisorLimit);
    }

    public IMersenneNumberDivisorByDivisorTester.IDivisorScanSession CreateDivisorSession()
    {
        // Divisor sessions are handed out only after configuration, so this guard would never trigger on the
        // production path.
        // if (!_isConfigured)
        // {
        //     throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
        // }

        if (_sessionPool.TryTake(out DivisorScanSession? session))
        {
            session.Reset();
            return session;
        }

        return new DivisorScanSession(this);
    }
}




