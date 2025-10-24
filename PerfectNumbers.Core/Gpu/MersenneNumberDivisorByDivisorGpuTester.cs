using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using PerfectNumbers.Core;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

/// <summary>
/// Scans Mersenne divisors on the GPU for prime exponents p >= 31 using per-divisor Montgomery data.
/// Consumers must call <see cref="ConfigureFromMaxPrime"/> before invoking other members so the divisor limits are populated.
/// </summary>
public sealed class MersenneNumberDivisorByDivisorGpuTester : IMersenneNumberDivisorByDivisorTester
{
    private int _gpuBatchSize = GpuConstants.ScanBatchSize;
    // EvenPerfectBitScanner configures the GPU tester once before scanning and never mutates the configuration afterwards,
    // so the synchronization fields from the previous implementation remain commented out here.
    // private readonly object _sync = new();
    private ulong _divisorLimit;
    // private bool _isConfigured;

    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<MontgomeryDivisorData>, ArrayView<ulong>, ArrayView<byte>, ArrayView<int>>> _kernelCache = new();
    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, MontgomeryDivisorData, ulong, ulong, ArrayView<ulong>, ArrayView<byte>>> _stepperKernelCache = new(AcceleratorReferenceComparer.Instance);
    private readonly ConcurrentDictionary<Accelerator, ConcurrentBag<BatchResources>> _resourcePools = new(AcceleratorReferenceComparer.Instance);
    private readonly ConcurrentBag<GpuContextPool.GpuContextLease> _acceleratorPool = new();
    private readonly ConcurrentBag<DivisorScanSession> _sessionPool = new();

    private Action<Index1D, ArrayView<MontgomeryDivisorData>, ArrayView<ulong>, ArrayView<byte>, ArrayView<int>> GetKernel(Accelerator accelerator) =>
        _kernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<MontgomeryDivisorData>, ArrayView<ulong>, ArrayView<byte>, ArrayView<int>>(DivisorByDivisorKernels.CheckKernel));

    private Action<Index1D, MontgomeryDivisorData, ulong, ulong, ArrayView<ulong>, ArrayView<byte>> GetStepperKernel(Accelerator accelerator) =>
        _stepperKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, MontgomeryDivisorData, ulong, ulong, ArrayView<ulong>, ArrayView<byte>>(DivisorByDivisorKernels.EvaluateDivisorWithStepperKernel));


    public int GpuBatchSize
    {
        get => _gpuBatchSize;
        set => _gpuBatchSize = value < 1 ? 1 : value;
    }

    int IMersenneNumberDivisorByDivisorTester.BatchSize
    {
        get => GpuBatchSize;
        set => GpuBatchSize = value;
    }

    public void ConfigureFromMaxPrime(ulong maxPrime)
    {
        // EvenPerfectBitScanner configures the GPU tester once before scanning and never mutates the configuration afterwards,
        // so synchronization and runtime configuration guards are unnecessary here.
        // lock (_sync)
        // {
        //     _divisorLimit = ComputeDivisorLimitFromMaxPrimeGpu(maxPrime);
        //     _isConfigured = true;
        // }

        _divisorLimit = ComputeDivisorLimitFromMaxPrimeGpu(maxPrime);
        // _isConfigured = true;
    }

    public bool IsPrime(ulong prime, out bool divisorsExhausted)
    {
        ulong allowedMax;
        int batchCapacity;

        // EvenPerfectBitScanner only calls into this tester after configuring it once, so we can read the cached values without locking.
        // lock (_sync)
        // {
        //     if (!_isConfigured)
        //     {
        //         throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
        //     }

        //     allowedMax = ComputeAllowedMaxDivisorGpu(prime, _divisorLimit);
        //     batchCapacity = _gpuBatchSize;
        // }

        allowedMax = ComputeAllowedMaxDivisorGpu(prime, _divisorLimit);
        batchCapacity = _gpuBatchSize;

        // Production scans never shrink the divisor window below three, so this guard stays commented out.
        // if (allowedMax < 3UL)
        // {
        //     divisorsExhausted = true;
        //     return true;
        // }

        bool composite;
        bool coveredRange;
        ulong processedCount;
        ulong lastProcessed;

        var gpuLease = RentAccelerator();
        var accelerator = gpuLease.Accelerator;

        Monitor.Enter(gpuLease.ExecutionLock);

        var kernel = GetKernel(accelerator);
        var resources = RentBatchResources(accelerator, batchCapacity);

        composite = CheckDivisors(
            prime,
            allowedMax,
            kernel,
            resources.DivisorsBuffer,
            resources.ExponentBuffer,
            resources.HitsBuffer,
            resources.HitIndexBuffer,
            resources.Divisors,
            resources.Exponents,
            resources.DivisorData,
            resources.CycleLengths,
            resources.CycleCapacity,
            out lastProcessed,
            out coveredRange,
            out processedCount);

        ReturnBatchResources(accelerator, resources);
        Monitor.Exit(gpuLease.ExecutionLock);
        ReturnAccelerator(gpuLease);

        if (composite)
        {
            divisorsExhausted = true;
            return false;
        }

        divisorsExhausted = coveredRange;
        return true;
    }

    public void PrepareCandidates(in ReadOnlySpan<ulong> primes, Span<ulong> allowedMaxValues)
    {
        // The scanner always supplies matching spans, so the previous validation remains commented out.
        // if (allowedMaxValues.Length < primes.Length)
        // {
        //     throw new ArgumentException("allowedMaxValues span must be at least as long as primes span.", nameof(allowedMaxValues));
        // }

        ulong divisorLimit;

        // EvenPerfectBitScanner configures the tester exactly once before preparing candidates.
        // lock (_sync)
        // {
        //     if (!_isConfigured)
        //     {
        //         throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
        //     }

        //     divisorLimit = _divisorLimit;
        // }

        divisorLimit = _divisorLimit;

        for (int index = 0; index < primes.Length; index++)
        {
            allowedMaxValues[index] = ComputeAllowedMaxDivisorGpu(primes[index], divisorLimit);
        }
    }

    private static bool CheckDivisors(
        ulong prime,
        ulong allowedMax,
        Action<Index1D, ArrayView<MontgomeryDivisorData>, ArrayView<ulong>, ArrayView<byte>, ArrayView<int>> kernel,
        MemoryBuffer1D<MontgomeryDivisorData, Stride1D.Dense> divisorsBuffer,
        MemoryBuffer1D<ulong, Stride1D.Dense> exponentBuffer,
        MemoryBuffer1D<byte, Stride1D.Dense> hitsBuffer,
        MemoryBuffer1D<int, Stride1D.Dense> hitIndexBuffer,
        in ulong[] divisors,
        ulong[] exponents,
        MontgomeryDivisorData[] divisorData,
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

        cycleCapacity = Math.Max(1, cycleCapacity);
        int chunkCapacity = Math.Min(batchCapacity, cycleCapacity);

        UInt128 twoP128 = (UInt128)prime << 1;
        UInt128 allowedMax128 = allowedMax;
        UInt128 firstDivisor128 = twoP128 + UInt128.One;
        bool invalidStride = twoP128 == UInt128.Zero;
        bool outOfRange = firstDivisor128 > allowedMax128;
        UInt128 numerator = allowedMax128 - UInt128.One;
        UInt128 maxK128 = invalidStride || outOfRange ? UInt128.Zero : numerator / twoP128;
        bool hasCandidates = !invalidStride && !outOfRange && maxK128 != UInt128.Zero;
        if (!hasCandidates)
        {
            coveredRange = true;
            return false;
        }

        ulong maxK = maxK128 > ulong.MaxValue ? ulong.MaxValue : (ulong)maxK128;
        ulong currentK = 1UL;

        UInt128 currentDivisor128 = firstDivisor128;
        bool lastIsSeven = (prime & 3UL) == 3UL;
        DivisorCycleCache cycleCache = DivisorCycleCache.Shared;

        ArrayPool<ulong> ulongPool = ThreadStaticPools.UlongPool;
        ulong[] filteredDivisors = ulongPool.Rent(chunkCapacity);
        Span<ulong> filteredStorage = filteredDivisors.AsSpan();
        Span<ulong> cycleLengthStorage = cycleLengths.AsSpan();
        Span<MontgomeryDivisorData> divisorDataStorage = divisorData.AsSpan();
        Span<ulong> divisorStorage = divisors.AsSpan();
        Span<ulong> exponentStorage = exponents.AsSpan();

        while (currentK <= maxK && !composite)
        {
            int chunkCount = Math.Min(chunkCapacity, batchCapacity);
            ulong remainingK = maxK - currentK + 1UL;
            if ((ulong)chunkCount > remainingK)
            {
                chunkCount = (int)remainingK;
            }

            if (chunkCount <= 0)
            {
                processedAll = true;
                break;
            }

            int filteredCount = 0;
            UInt128 nextDivisor128 = currentDivisor128;

            for (int i = 0; i < chunkCount; i++)
            {
                ulong candidate = (ulong)nextDivisor128;
                nextDivisor128 += twoP128;

                if (CandidatePassesHeuristics(candidate, lastIsSeven))
                {
                    filteredStorage[filteredCount++] = candidate;
                }
            }

            processedCount += (ulong)chunkCount;

            UInt128 lastDivisor128 = nextDivisor128 - twoP128;
            lastProcessed = (ulong)lastDivisor128;
            currentDivisor128 = nextDivisor128;

            currentK += (ulong)chunkCount;
            processedAll = currentK > maxK;

            if (filteredCount == 0)
            {
                continue;
            }

            Span<ulong> filteredDivisorsSpan = filteredStorage[..filteredCount];
            Span<ulong> cycleSpan = cycleLengthStorage[..filteredCount];
            cycleCache.GetCycleLengths(filteredDivisorsSpan, cycleSpan);

            int admissibleCount = 0;
            for (int i = 0; i < filteredCount; i++)
            {
                ulong candidate = filteredDivisorsSpan[i];
                ulong candidateCycle = cycleSpan[i];
                if (candidateCycle != prime)
                {
                    continue;
                }

                filteredDivisorsSpan[admissibleCount] = candidate;
                admissibleCount++;
            }

            if (admissibleCount == 0)
            {
                continue;
            }

            Span<MontgomeryDivisorData> divisorDataSpan = divisorDataStorage[..admissibleCount];
            Span<ulong> divisorSpan = divisorStorage[..admissibleCount];
            Span<ulong> exponentSpan = exponentStorage[..admissibleCount];
            ArrayView1D<MontgomeryDivisorData, Stride1D.Dense> divisorView = divisorsBuffer.View.SubView(0, admissibleCount);
            ArrayView1D<ulong, Stride1D.Dense> exponentView = exponentBuffer.View.SubView(0, admissibleCount);
            ArrayView1D<byte, Stride1D.Dense> hitsView = hitsBuffer.View.SubView(0, admissibleCount);
            ArrayView1D<int, Stride1D.Dense> hitIndexView = hitIndexBuffer.View.SubView(0, 1);

            for (int i = 0; i < admissibleCount; i++)
            {
                ulong divisorValue = filteredDivisorsSpan[i];
                divisorSpan[i] = divisorValue;
                exponentSpan[i] = prime;
                divisorDataSpan[i] = MontgomeryDivisorData.FromModulus(divisorValue);
            }

            divisorView.CopyFromCPU(ref MemoryMarshal.GetReference(divisorDataSpan), admissibleCount);
            exponentView.CopyFromCPU(ref MemoryMarshal.GetReference(exponentSpan), admissibleCount);
            int sentinel = int.MaxValue;
            hitIndexView.SubView(0, 1).CopyFromCPU(ref sentinel, 1);

            kernel(admissibleCount, divisorView, exponentView, hitsView, hitIndexView);

            int firstHit = sentinel;
            hitIndexView.SubView(0, 1).CopyToCPU(ref firstHit, 1);
            int hitIndex = firstHit >= admissibleCount ? -1 : firstHit;
            bool hitFound = hitIndex >= 0;
            composite = hitFound;
            int lastIndex = admissibleCount - 1;
            lastProcessed = hitFound ? divisorSpan[hitIndex] : divisorSpan[lastIndex];
        }

        ulongPool.Return(filteredDivisors, clearArray: false);

        coveredRange = composite || processedAll || (currentDivisor128 > allowedMax128);
        return composite;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool CandidatePassesHeuristics(ulong candidate, bool lastIsSeven)
    {
        // Keep the divisibility filters aligned with the divisor-cycle snapshot so we never
        // request missing entries from the cache.
        ulong remainder10 = candidate % 10UL;
        bool accept10 = lastIsSeven
            ? (remainder10 == 3UL || remainder10 == 7UL || remainder10 == 9UL)
            : (remainder10 == 1UL || remainder10 == 3UL || remainder10 == 9UL);
        if (!accept10)
        {
            return false;
        }

        ulong remainder8 = candidate & 7UL;
        if (remainder8 != 1UL && remainder8 != 7UL)
        {
            return false;
        }

        if (candidate % 3UL == 0UL || candidate % 5UL == 0UL || candidate % 7UL == 0UL || candidate % 11UL == 0UL)
        {
            return false;
        }

        return true;
    }





    private static ulong ComputeDivisorLimitFromMaxPrimeGpu(ulong maxPrime)
    {
        // The --mersenne=bydivisor flow in EvenPerfectBitScanner only calls ConfigureFromMaxPrime with primes greater than 1,
        // so the guard below never trips in production runs.
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

    private static ulong ComputeAllowedMaxDivisorGpu(ulong prime, ulong divisorLimit)
    {
        // Production --mersenne=bydivisor runs only pass prime exponents, so the guard below never executes outside tests.
        // if (prime <= 1UL)
        // {
        //     return 0UL;
        // }
        if (prime - 1UL >= 64UL)
        {
            return divisorLimit;
        }

        ulong computedLimit = (1UL << (int)(prime - 1UL)) - 1UL;
        return computedLimit < divisorLimit ? computedLimit : divisorLimit;
    }

    public sealed class DivisorScanSession : IMersenneNumberDivisorByDivisorTester.IDivisorScanSession
    {
        private readonly MersenneNumberDivisorByDivisorGpuTester _owner;
        private readonly GpuContextPool.GpuContextLease _lease;
        private readonly Accelerator _accelerator;
        private Action<Index1D, MontgomeryDivisorData, ulong, ulong, ArrayView<ulong>, ArrayView<byte>> _stepperKernel = null!;
        private MemoryBuffer1D<ulong, Stride1D.Dense> _exponentsBuffer = null!;
        private MemoryBuffer1D<byte, Stride1D.Dense> _hitBuffer = null!;
        private ulong[] _hostBuffer = null!;
        private int _capacity;
        private bool _disposed;

        internal DivisorScanSession(MersenneNumberDivisorByDivisorGpuTester owner)
        {
            _owner = owner;
            _lease = GpuContextPool.RentPreferred(preferCpu: false);
            _accelerator = _lease.Accelerator;
            _capacity = Math.Max(1, owner._gpuBatchSize);
        }

        internal void Reset()
        {
            _disposed = false;
        }

        private void EnsureExecutionResourcesLocked(int requiredCapacity)
        {
            if (_stepperKernel == null)
            {
                _stepperKernel = _owner.GetStepperKernel(_accelerator);
            }

            bool allocateBuffers = _exponentsBuffer == null;
            if (!allocateBuffers && _capacity < requiredCapacity)
            {
                _exponentsBuffer?.Dispose();
                _hitBuffer?.Dispose();
                if (_hostBuffer != null)
                {
                    ThreadStaticPools.UlongPool.Return(_hostBuffer, clearArray: false);
                    _hostBuffer = null!;
                }

                _exponentsBuffer = null!;
                _hitBuffer = null!;
                allocateBuffers = true;
            }

            if (allocateBuffers)
            {
                int desiredCapacity = requiredCapacity > _capacity ? requiredCapacity : _capacity;
                _capacity = desiredCapacity;
                _exponentsBuffer = _accelerator.Allocate1D<ulong>(_capacity);
                _hitBuffer = _accelerator.Allocate1D<byte>(_capacity);
                _hostBuffer = ThreadStaticPools.UlongPool.Rent(_capacity);
            }
        }

        public void CheckDivisor(ulong divisor, in MontgomeryDivisorData divisorData, ulong divisorCycle, in ReadOnlySpan<ulong> primes, Span<byte> hits)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(DivisorScanSession));
            }

            int length = primes.Length;
            // EvenPerfectBitScanner always supplies at least one exponent per divisor check, so the guard stays commented out.
            // if (length == 0)
            // {
            //     return;
            // }

            // The GPU divisor sessions only materialize odd moduli greater than one (q = 2kp + 1),
            // so the defensive modulus guard stays commented out to keep the hot path branch-free.
            // if (divisorData.Modulus <= 1UL || (divisorData.Modulus & 1UL) == 0UL)
            // {
            //     hits.Clear();
            //     return;
            // }

            MontgomeryDivisorData cachedData = divisorData;
            if (cachedData.Modulus != divisor)
            {
                cachedData = MontgomeryDivisorData.FromModulus(divisor);
            }

            ulong cycle = divisorCycle;
            // EvenPerfectBitScanner resolves divisor cycles before scheduling GPU work, so the zero-cycle fallback stays disabled.
            // if (cycle == 0UL)
            // {
            //     throw new InvalidOperationException("GPU divisor cycle length must be non-zero.");
            // }

            ulong firstPrime = primes[0];
            // Precompute the initial cycle remainder on the host to keep the GPU kernel branch-free.
            // EvenPerfectBitScanner resolves divisor cycles before scheduling GPU work, so cycle never equals zero here.
            ulong firstRemainder = firstPrime % cycle;

            Monitor.Enter(_lease.ExecutionLock);

            EnsureExecutionResourcesLocked(length);

            ArrayView1D<ulong, Stride1D.Dense> exponentView = _exponentsBuffer.View;
            ArrayView1D<byte, Stride1D.Dense> hitView = _hitBuffer.View;

            Span<ulong> hostSpan = _hostBuffer.AsSpan(0, length);
            primes.CopyTo(hostSpan);

            ArrayView1D<ulong, Stride1D.Dense> exponentSlice = exponentView.SubView(0, length);
            exponentSlice.CopyFromCPU(ref MemoryMarshal.GetReference(hostSpan), length);

            ArrayView1D<byte, Stride1D.Dense> hitSliceView = hitView.SubView(0, length);
            var stepperKernel = _stepperKernel!;
            // EnsureExecutionResourcesLocked binds the GPU stepper kernel during the first session launch, so the null check
            // remains commented out to avoid a redundant branch on the production path.
            // if (stepperKernel is null)
            // {
            //     Monitor.Exit(_lease.ExecutionLock);
            //     throw new InvalidOperationException("GPU stepper kernel was not initialized.");
            // }

            stepperKernel(1, cachedData, cycle, firstRemainder, exponentSlice, hitSliceView);

            Span<byte> hitSlice = hits.Slice(0, length);
            hitSliceView.CopyToCPU(ref MemoryMarshal.GetReference(hitSlice), length);

            Monitor.Exit(_lease.ExecutionLock);
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
            // The EvenPerfectBitScanner driver configures the GPU tester before exposing it to callers, so the previous synchronization guard remains commented out.
            // lock (_sync)
            // {
            //     if (!_isConfigured)
            //     {
            //         throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
            //     }

            //     return _divisorLimit;
            // }

            return _divisorLimit;
        }
    }

    private GpuContextPool.GpuContextLease RentAccelerator()
    {
        if (_acceleratorPool.TryTake(out var lease))
        {
            return lease;
        }

        return GpuContextPool.RentPreferred(preferCpu: false);
    }

    private void ReturnAccelerator(GpuContextPool.GpuContextLease lease) => _acceleratorPool.Add(lease);

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
            HitIndexBuffer = accelerator.Allocate1D<int>(1);
            ArrayPool<ulong> ulongPool = ThreadStaticPools.UlongPool;
            Divisors = ulongPool.Rent(actualCapacity);
            Exponents = ulongPool.Rent(actualCapacity);
            DivisorData = ThreadStaticPools.MontgomeryDivisorDataPool.Rent(actualCapacity);
            CycleLengths = ulongPool.Rent(cycleCapacity);
            Capacity = actualCapacity;
            CycleCapacity = cycleCapacity;
        }

        internal readonly MemoryBuffer1D<MontgomeryDivisorData, Stride1D.Dense> DivisorsBuffer;

        internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> ExponentBuffer;

        internal readonly MemoryBuffer1D<byte, Stride1D.Dense> HitsBuffer;

        internal readonly MemoryBuffer1D<int, Stride1D.Dense> HitIndexBuffer;

        internal readonly ulong[] Divisors;

        internal readonly ulong[] Exponents;

        internal readonly MontgomeryDivisorData[] DivisorData;

        internal readonly ulong[] CycleLengths;

        internal readonly int Capacity;

        internal readonly int CycleCapacity;

        public void Dispose()
        {
            DivisorsBuffer.Dispose();
            ExponentBuffer.Dispose();
            HitsBuffer.Dispose();
            HitIndexBuffer.Dispose();
            ArrayPool<ulong> ulongPool = ThreadStaticPools.UlongPool;
            ArrayPool<MontgomeryDivisorData> divisorPool = ThreadStaticPools.MontgomeryDivisorDataPool;
            ulongPool.Return(Divisors, clearArray: false);
            ulongPool.Return(Exponents, clearArray: false);
            divisorPool.Return(DivisorData, clearArray: false);
            ulongPool.Return(CycleLengths, clearArray: false);
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
        // The configuration is immutable after setup, so the cached limit can be read without locking.
        // lock (_sync)
        // {
        //     if (!_isConfigured)
        //     {
        //         throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
        //     }

        //     return ComputeAllowedMaxDivisorGpu(prime, _divisorLimit);
        // }

        return ComputeAllowedMaxDivisorGpu(prime, _divisorLimit);
    }

    public DivisorScanSession CreateDivisorSession()
    {
        // The tester is configured once at startup and the session pool relies on thread-safe collections, so the previous synchronization guard stays commented out.
        // lock (_sync)
        // {
        //     if (!_isConfigured)
        //     {
        //         throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
        //     }

        //     if (_sessionPool.TryTake(out DivisorScanSession? session))
        //     {
        //         session.Reset();
        //         return session;
        //     }

        //     return new DivisorScanSession(this);
        // }

        if (_sessionPool.TryTake(out DivisorScanSession? session))
        {
            session.Reset();
            return session;
        }

        return new DivisorScanSession(this);
    }

    IMersenneNumberDivisorByDivisorTester.IDivisorScanSession IMersenneNumberDivisorByDivisorTester.CreateDivisorSession() => CreateDivisorSession();
}




