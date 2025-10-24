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
/// Scans Mersenne divisors on the GPU for prime exponents p >= 31 using cached GPU divisor partial data.
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

    private delegate void StepperKernel(Index1D index, in GpuDivisorPartialData divisor, ulong divisorCycle, ulong firstCycleRemainder, ArrayView<ulong> exponents, ArrayView<byte> hits);

    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>, ArrayView<int>>> _kernelCache = new();
    private readonly ConcurrentDictionary<Accelerator, StepperKernel> _stepperKernelCache = new(AcceleratorReferenceComparer.Instance);
    private readonly ConcurrentDictionary<Accelerator, ConcurrentBag<BatchResources>> _resourcePools = new(AcceleratorReferenceComparer.Instance);
    private readonly ConcurrentBag<GpuContextPool.GpuContextLease> _acceleratorPool = new();
    private readonly ConcurrentBag<DivisorScanSession> _sessionPool = new();

    private Action<Index1D, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>, ArrayView<int>> GetKernel(Accelerator accelerator) =>
        _kernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>, ArrayView<int>>(DivisorByDivisorKernels.CheckKernel));

    private StepperKernel GetStepperKernel(Accelerator accelerator) =>
        _stepperKernelCache.GetOrAdd(accelerator, acc =>
        {
            var kernel = acc.LoadAutoGroupedStreamKernel<Index1D, GpuDivisorPartialData, ulong, ulong, ArrayView<ulong>, ArrayView<byte>>(DivisorByDivisorKernels.EvaluateDivisorWithStepperKernel);
            return (Index1D extent, in GpuDivisorPartialData divisor, ulong divisorCycle, ulong firstCycleRemainder, ArrayView<ulong> exponents, ArrayView<byte> hits) =>
            {
                kernel(extent, divisor, divisorCycle, firstCycleRemainder, exponents, hits);
            };
        });


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
        Action<Index1D, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>, ArrayView<int>> kernel,
        MemoryBuffer1D<ulong, Stride1D.Dense> divisorsBuffer,
        MemoryBuffer1D<ulong, Stride1D.Dense> exponentBuffer,
        MemoryBuffer1D<byte, Stride1D.Dense> hitsBuffer,
        MemoryBuffer1D<int, Stride1D.Dense> hitIndexBuffer,
        in ulong[] divisors,
        ulong[] exponents,
        out ulong lastProcessed,
        out bool coveredRange,
        out ulong processedCount)
    {
        int batchCapacity = (int)divisorsBuffer.Length;
        bool composite = false;
        bool processedAll = false;
        processedCount = 0UL;
        lastProcessed = 0UL;

        int chunkCapacity = Math.Max(1, batchCapacity);

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

        byte step10 = (byte)(twoP128 % 10UL);
        byte step8 = (byte)(twoP128 % 8UL);
        byte step5 = (byte)(twoP128 % 5UL);
        byte step3 = (byte)(twoP128 % 3UL);
        byte step7 = (byte)(twoP128 % 7UL);
        byte step11 = (byte)(twoP128 % 11UL);

        UInt128 currentDivisor128 = firstDivisor128;
        byte remainder10 = (byte)((ulong)(currentDivisor128 % 10UL));
        byte remainder8 = (byte)((ulong)(currentDivisor128 & 7UL));
        byte remainder5 = (byte)((ulong)(currentDivisor128 % 5UL));
        byte remainder3 = (byte)((ulong)(currentDivisor128 % 3UL));
        byte remainder7 = (byte)((ulong)(currentDivisor128 % 7UL));
        byte remainder11 = (byte)((ulong)(currentDivisor128 % 11UL));
        bool lastIsSeven = (prime & 3UL) == 3UL;

        ArrayPool<ulong> ulongPool = ThreadStaticPools.UlongPool;
        ulong[] filteredDivisors = ulongPool.Rent(chunkCapacity);
        Span<ulong> filteredStorage = filteredDivisors.AsSpan();
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

            byte localRemainder10 = remainder10;
            byte localRemainder8 = remainder8;
            byte localRemainder5 = remainder5;
            byte localRemainder3 = remainder3;
            byte localRemainder7 = remainder7;
            byte localRemainder11 = remainder11;

            for (int i = 0; i < chunkCount; i++)
            {
                ulong candidate = (ulong)nextDivisor128;

                if (CandidatePassesHeuristics(localRemainder10, localRemainder8, localRemainder3, localRemainder5, localRemainder7, localRemainder11, lastIsSeven))
                {
                    filteredStorage[filteredCount++] = candidate;
                }

                nextDivisor128 += twoP128;
                localRemainder10 = AddMod(localRemainder10, step10, 10);
                localRemainder8 = AddMod(localRemainder8, step8, 8);
                localRemainder5 = AddMod(localRemainder5, step5, 5);
                localRemainder3 = AddMod(localRemainder3, step3, 3);
                localRemainder7 = AddMod(localRemainder7, step7, 7);
                localRemainder11 = AddMod(localRemainder11, step11, 11);
            }

            remainder10 = localRemainder10;
            remainder8 = localRemainder8;
            remainder5 = localRemainder5;
            remainder3 = localRemainder3;
            remainder7 = localRemainder7;
            remainder11 = localRemainder11;

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

            Span<ulong> divisorSpan = divisorStorage;
            Span<ulong> exponentSpan = exponentStorage;

            int admissibleCount = 0;
            for (int i = 0; i < filteredCount; i++)
            {
                ulong divisorValue = filteredDivisorsSpan[i];
                MontgomeryDivisorData montgomeryData = MontgomeryDivisorData.FromModulus(divisorValue);
                ulong divisorCycle = ResolveDivisorCycle(divisorValue, prime, in montgomeryData);
                if (divisorCycle != prime)
                {
                    continue;
                }

                divisorSpan[admissibleCount] = divisorValue;
                exponentSpan[admissibleCount] = prime;
                admissibleCount++;
            }

            if (admissibleCount == 0)
            {
                continue;
            }

            Span<ulong> activeDivisors = divisorSpan[..admissibleCount];
            Span<ulong> activeExponents = exponentSpan[..admissibleCount];
            ArrayView1D<ulong, Stride1D.Dense> divisorView = divisorsBuffer.View.SubView(0, admissibleCount);
            ArrayView1D<ulong, Stride1D.Dense> exponentView = exponentBuffer.View.SubView(0, admissibleCount);
            ArrayView1D<byte, Stride1D.Dense> hitsView = hitsBuffer.View.SubView(0, admissibleCount);
            ArrayView1D<int, Stride1D.Dense> hitIndexView = hitIndexBuffer.View.SubView(0, 1);

            divisorView.CopyFromCPU(ref MemoryMarshal.GetReference(activeDivisors), admissibleCount);
            exponentView.CopyFromCPU(ref MemoryMarshal.GetReference(activeExponents), admissibleCount);
            int sentinel = int.MaxValue;
            hitIndexView.SubView(0, 1).CopyFromCPU(ref sentinel, 1);

            kernel(admissibleCount, divisorView, exponentView, hitsView, hitIndexView);

            int firstHit = sentinel;
            hitIndexView.SubView(0, 1).CopyToCPU(ref firstHit, 1);
            int hitIndex = firstHit >= admissibleCount ? -1 : firstHit;
            bool hitFound = hitIndex >= 0;
            composite = hitFound;
            int lastIndex = admissibleCount - 1;
            lastProcessed = hitFound ? activeDivisors[hitIndex] : activeDivisors[lastIndex];
        }

        ulongPool.Return(filteredDivisors, clearArray: false);

        coveredRange = composite || processedAll || (currentDivisor128 > allowedMax128);
        return composite;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong ResolveDivisorCycle(ulong divisor, ulong prime, in MontgomeryDivisorData divisorData)
    {
        if (!MersenneDivisorCycles.TryCalculateCycleLengthForExponentCpu(divisor, prime, divisorData, out ulong computedCycle, out bool primeOrderFailed) || computedCycle == 0UL)
        {
            return MersenneDivisorCycles.CalculateCycleLength(divisor, divisorData, skipPrimeOrderHeuristic: primeOrderFailed);
        }

        return computedCycle;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool CandidatePassesHeuristics(
        byte remainder10,
        byte remainder8,
        byte remainder3,
        byte remainder5,
        byte remainder7,
        byte remainder11,
        bool lastIsSeven)
    {
        const ushort DecimalMaskWhenLastIsSeven = (1 << 3) | (1 << 7) | (1 << 9);
        const ushort DecimalMaskOtherwise = (1 << 1) | (1 << 3) | (1 << 9);

        ushort decimalMask = lastIsSeven ? DecimalMaskWhenLastIsSeven : DecimalMaskOtherwise;
        if (((decimalMask >> remainder10) & 1) == 0)
        {
            return false;
        }

        if (remainder8 != 1 && remainder8 != 7)
        {
            return false;
        }

        if (remainder3 == 0 || remainder5 == 0 || remainder7 == 0 || remainder11 == 0)
        {
            return false;
        }

        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte AddMod(byte value, byte delta, byte modulus)
    {
        int result = value + delta;
        if (result >= modulus)
        {
            result -= modulus;
            if (result >= modulus)
            {
                result -= modulus;
            }
        }

        return (byte)result;
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
        private StepperKernel _stepperKernel = null!;
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

            GpuDivisorPartialData cachedData = new GpuDivisorPartialData(divisor);

            ulong cycle = divisorCycle;
            ulong firstPrime = primes[0];

            if (cycle == 0UL)
            {
                if (!MersenneDivisorCycles.TryCalculateCycleLengthForExponentCpu(divisor, firstPrime, divisorData, out ulong computedCycle, out bool primeOrderFailed) || computedCycle == 0UL)
                {
                    cycle = MersenneDivisorCycles.CalculateCycleLength(divisor, divisorData, skipPrimeOrderHeuristic: primeOrderFailed);
                }
                else
                {
                    cycle = computedCycle;
                }

                if (cycle == 0UL)
                {
                    throw new InvalidOperationException("GPU divisor cycle length must be non-zero.");
                }
            }

            // Precompute the initial cycle remainder on the host to keep the GPU kernel branch-free.
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

            stepperKernel(1, in cachedData, cycle, firstRemainder, exponentSlice, hitSliceView);

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
            int actualCapacity = Math.Max(1, capacity);

            DivisorsBuffer = accelerator.Allocate1D<ulong>(actualCapacity);
            ExponentBuffer = accelerator.Allocate1D<ulong>(actualCapacity);
            HitsBuffer = accelerator.Allocate1D<byte>(actualCapacity);
            HitIndexBuffer = accelerator.Allocate1D<int>(1);
            ArrayPool<ulong> ulongPool = ThreadStaticPools.UlongPool;
            Divisors = ulongPool.Rent(actualCapacity);
            Exponents = ulongPool.Rent(actualCapacity);
            Capacity = actualCapacity;
        }

        internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> DivisorsBuffer;

        internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> ExponentBuffer;

        internal readonly MemoryBuffer1D<byte, Stride1D.Dense> HitsBuffer;

        internal readonly MemoryBuffer1D<int, Stride1D.Dense> HitIndexBuffer;

        internal readonly ulong[] Divisors;

        internal readonly ulong[] Exponents;

        internal readonly int Capacity;

        public void Dispose()
        {
            DivisorsBuffer.Dispose();
            ExponentBuffer.Dispose();
            HitsBuffer.Dispose();
            HitIndexBuffer.Dispose();
            ArrayPool<ulong> ulongPool = ThreadStaticPools.UlongPool;
            ulongPool.Return(Divisors, clearArray: false);
            ulongPool.Return(Exponents, clearArray: false);
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




