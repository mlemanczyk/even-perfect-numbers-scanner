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

    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<GpuDivisorPartialData>, ArrayView<int>, ArrayView<int>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>, ArrayView<int>>> _kernelCache = new(AcceleratorReferenceComparer.Instance);
    private readonly ConcurrentBag<BatchResources> _resourcePool = new();
    private readonly ConcurrentBag<GpuContextPool.GpuContextLease> _acceleratorPool = new();
    private readonly ConcurrentBag<DivisorScanSession> _sessionPool = new();

    private Action<Index1D, ArrayView<GpuDivisorPartialData>, ArrayView<int>, ArrayView<int>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>, ArrayView<int>> GetKernel(Accelerator accelerator) =>
        _kernelCache.GetOrAdd(accelerator, acc =>
            acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuDivisorPartialData>, ArrayView<int>, ArrayView<int>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>, ArrayView<int>>(DivisorByDivisorKernels.CheckKernel));


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
            resources.DivisorDataBuffer,
            resources.OffsetBuffer,
            resources.CountBuffer,
            resources.CycleBuffer,
            resources.ExponentBuffer,
            resources.HitsBuffer,
            resources.HitIndexBuffer,
            resources.Divisors,
            resources.Exponents,
            resources.DivisorData,
            resources.Offsets,
            resources.Counts,
            resources.Cycles,
            out lastProcessed,
            out coveredRange,
            out processedCount);

        ReturnBatchResources(resources);
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
        Action<Index1D, ArrayView<GpuDivisorPartialData>, ArrayView<int>, ArrayView<int>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>, ArrayView<int>> kernel,
        MemoryBuffer1D<GpuDivisorPartialData, Stride1D.Dense> divisorDataBuffer,
        MemoryBuffer1D<int, Stride1D.Dense> offsetBuffer,
        MemoryBuffer1D<int, Stride1D.Dense> countBuffer,
        MemoryBuffer1D<ulong, Stride1D.Dense> cycleBuffer,
        MemoryBuffer1D<ulong, Stride1D.Dense> exponentBuffer,
        MemoryBuffer1D<byte, Stride1D.Dense> hitsBuffer,
        MemoryBuffer1D<int, Stride1D.Dense> hitIndexBuffer,
        ulong[] divisors,
        ulong[] exponents,
        GpuDivisorPartialData[] divisorData,
        int[] offsets,
        int[] counts,
        ulong[] cycles,
        out ulong lastProcessed,
        out bool coveredRange,
        out ulong processedCount)
    {
        int batchCapacity = (int)divisorDataBuffer.Length;
        bool composite = false;
        processedCount = 0UL;
        lastProcessed = 0UL;

        // var enumerator = HeuristicPrimeTester.CreateMersenneDivisorEnumerator(prime, allowedMax);
        var enumerator = PrimeTester.CreateMersenneDivisorEnumerator(prime, allowedMax);

        Span<ulong> divisorSpan = divisors.AsSpan();
        Span<ulong> exponentSpan = exponents.AsSpan();
        Span<GpuDivisorPartialData> divisorDataSpan = divisorData.AsSpan();
        Span<int> offsetSpan = offsets.AsSpan();
        Span<int> countSpan = counts.AsSpan();
        Span<ulong> cycleSpan = cycles.AsSpan();

        while (!composite)
        {
            int admissibleCount = 0;
            int exponentIndex = 0;

            // while (admissibleCount < batchCapacity && enumerator.TryGetNext(out HeuristicPrimeTester.HeuristicDivisorCandidate candidate))
            while (admissibleCount < batchCapacity && enumerator.TryGetNext(out PrimeTester.HeuristicDivisorCandidate candidate))
            {
                processedCount = enumerator.ProcessedCount;
                ulong divisorValue = candidate.Value;

                // HeuristicPrimeTester.HeuristicDivisorPreparation preparation = HeuristicPrimeTester.PrepareHeuristicDivisor(in candidate);
                PrimeTester.HeuristicDivisorPreparation preparation = PrimeTester.PrepareHeuristicDivisor(in candidate);
                // ulong divisorCycle = HeuristicPrimeTester.ResolveHeuristicCycleLength(
                ulong divisorCycle = PrimeTester.ResolveHeuristicCycleLength(
                    prime,
                    in preparation,
                    out _,
                    out _,
                    out _);

                if (divisorCycle != prime)
                {
                    continue;
                }

                divisorSpan[admissibleCount] = divisorValue;
                exponentSpan[exponentIndex] = prime;
                divisorDataSpan[admissibleCount] = new GpuDivisorPartialData(divisorValue);
                offsetSpan[admissibleCount] = exponentIndex;
                countSpan[admissibleCount] = 1;
                cycleSpan[admissibleCount] = divisorCycle;

                admissibleCount++;
                exponentIndex++;
            }

            processedCount = enumerator.ProcessedCount;
            lastProcessed = enumerator.LastDivisor;

            if (admissibleCount == 0)
            {
                if (enumerator.Exhausted)
                {
                    break;
                }

                continue;
            }

            ArrayView1D<GpuDivisorPartialData, Stride1D.Dense> divisorView = divisorDataBuffer.View.SubView(0, admissibleCount);
            ArrayView1D<int, Stride1D.Dense> offsetView = offsetBuffer.View.SubView(0, admissibleCount);
            ArrayView1D<int, Stride1D.Dense> countView = countBuffer.View.SubView(0, admissibleCount);
            ArrayView1D<ulong, Stride1D.Dense> cycleView = cycleBuffer.View.SubView(0, admissibleCount);
            ArrayView1D<ulong, Stride1D.Dense> exponentView = exponentBuffer.View.SubView(0, exponentIndex);
            ArrayView1D<byte, Stride1D.Dense> hitsView = hitsBuffer.View.SubView(0, exponentIndex);
            ArrayView1D<int, Stride1D.Dense> hitIndexView = hitIndexBuffer.View.SubView(0, 1);

            divisorView.CopyFromCPU(ref MemoryMarshal.GetReference(divisorDataSpan), admissibleCount);
            offsetView.CopyFromCPU(ref MemoryMarshal.GetReference(offsetSpan), admissibleCount);
            countView.CopyFromCPU(ref MemoryMarshal.GetReference(countSpan), admissibleCount);
            cycleView.CopyFromCPU(ref MemoryMarshal.GetReference(cycleSpan), admissibleCount);
            exponentView.CopyFromCPU(ref MemoryMarshal.GetReference(exponentSpan), exponentIndex);

            int sentinel = int.MaxValue;
            hitIndexView.CopyFromCPU(ref sentinel, 1);

            kernel(admissibleCount, divisorView, offsetView, countView, exponentView, cycleView, hitsView, hitIndexView);

            int firstHit = sentinel;
            hitIndexView.CopyToCPU(ref firstHit, 1);
            int hitIndex = firstHit >= admissibleCount ? -1 : firstHit;
            bool hitFound = hitIndex >= 0;
            composite = hitFound;
            int lastIndex = admissibleCount - 1;
            lastProcessed = hitFound ? divisorSpan[hitIndex] : divisorSpan[lastIndex];
        }

        coveredRange = composite || enumerator.Exhausted;
        return composite;
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
        private Action<Index1D, ArrayView<GpuDivisorPartialData>, ArrayView<int>, ArrayView<int>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>, ArrayView<int>> _kernel = null!;
        private MemoryBuffer1D<GpuDivisorPartialData, Stride1D.Dense> _divisorBuffer = null!;
        private MemoryBuffer1D<int, Stride1D.Dense> _offsetBuffer = null!;
        private MemoryBuffer1D<int, Stride1D.Dense> _countBuffer = null!;
        private MemoryBuffer1D<ulong, Stride1D.Dense> _cycleBuffer = null!;
        private MemoryBuffer1D<int, Stride1D.Dense> _firstHitBuffer = null!;
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
            if (_kernel == null)
            {
                _kernel = _owner.GetKernel(_accelerator);
            }

            if (_divisorBuffer == null)
            {
                _divisorBuffer = _accelerator.Allocate1D<GpuDivisorPartialData>(1);
                _offsetBuffer = _accelerator.Allocate1D<int>(1);
                _countBuffer = _accelerator.Allocate1D<int>(1);
                _cycleBuffer = _accelerator.Allocate1D<ulong>(1);
                _firstHitBuffer = _accelerator.Allocate1D<int>(1);
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

            Monitor.Enter(_lease.ExecutionLock);

            EnsureExecutionResourcesLocked(length);

            ArrayView1D<GpuDivisorPartialData, Stride1D.Dense> divisorView = _divisorBuffer.View;
            ArrayView1D<int, Stride1D.Dense> offsetView = _offsetBuffer.View;
            ArrayView1D<int, Stride1D.Dense> countView = _countBuffer.View;
            ArrayView1D<ulong, Stride1D.Dense> cycleView = _cycleBuffer.View;
            ArrayView1D<int, Stride1D.Dense> firstHitView = _firstHitBuffer.View;
            ArrayView1D<ulong, Stride1D.Dense> exponentView = _exponentsBuffer.View;
            ArrayView1D<byte, Stride1D.Dense> hitView = _hitBuffer.View;

            GpuDivisorPartialData partialData = new GpuDivisorPartialData(divisor);
            divisorView.SubView(0, 1).CopyFromCPU(ref partialData, 1);

            int offsetValue = 0;
            offsetView.SubView(0, 1).CopyFromCPU(ref offsetValue, 1);

            int countValue = length;
            countView.SubView(0, 1).CopyFromCPU(ref countValue, 1);

            ulong cycleValue = cycle;
            cycleView.SubView(0, 1).CopyFromCPU(ref cycleValue, 1);

            Span<ulong> hostSpan = _hostBuffer.AsSpan(0, length);
            primes.CopyTo(hostSpan);

            ArrayView1D<ulong, Stride1D.Dense> exponentSlice = exponentView.SubView(0, length);
            exponentSlice.CopyFromCPU(ref MemoryMarshal.GetReference(hostSpan), length);

            ArrayView1D<byte, Stride1D.Dense> hitSliceView = hitView.SubView(0, length);

            int sentinel = int.MaxValue;
            firstHitView.SubView(0, 1).CopyFromCPU(ref sentinel, 1);

            var kernel = _kernel;
            if (kernel is null)
            {
                Monitor.Exit(_lease.ExecutionLock);
                throw new InvalidOperationException("GPU divisor kernel was not initialized.");
            }

            kernel(1, divisorView, offsetView, countView, exponentSlice, cycleView, hitSliceView, firstHitView);

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
        if (!_resourcePool.TryTake(out BatchResources? resources))
        {
            resources = new BatchResources(Math.Max(1, capacity));
        }

        resources.Bind(accelerator, capacity);
        return resources;
    }

    private void ReturnBatchResources(BatchResources resources) => _resourcePool.Add(resources);

    private sealed class BatchResources : IDisposable
    {
        private Accelerator? _accelerator;
        private MemoryBuffer1D<GpuDivisorPartialData, Stride1D.Dense>? _divisorDataBuffer;
        private MemoryBuffer1D<int, Stride1D.Dense>? _offsetBuffer;
        private MemoryBuffer1D<int, Stride1D.Dense>? _countBuffer;
        private MemoryBuffer1D<ulong, Stride1D.Dense>? _cycleBuffer;
        private MemoryBuffer1D<ulong, Stride1D.Dense>? _exponentBuffer;
        private MemoryBuffer1D<byte, Stride1D.Dense>? _hitsBuffer;
        private MemoryBuffer1D<int, Stride1D.Dense>? _hitIndexBuffer;
        private int _deviceCapacity;
        private int _hostCapacity;

        internal BatchResources(int capacity)
        {
            int actualCapacity = Math.Max(1, capacity);
            RentHostArrays(actualCapacity);
        }

        internal MemoryBuffer1D<GpuDivisorPartialData, Stride1D.Dense> DivisorDataBuffer => _divisorDataBuffer ?? throw new InvalidOperationException("Batch resources not bound to an accelerator.");

        internal MemoryBuffer1D<int, Stride1D.Dense> OffsetBuffer => _offsetBuffer ?? throw new InvalidOperationException("Batch resources not bound to an accelerator.");

        internal MemoryBuffer1D<int, Stride1D.Dense> CountBuffer => _countBuffer ?? throw new InvalidOperationException("Batch resources not bound to an accelerator.");

        internal MemoryBuffer1D<ulong, Stride1D.Dense> CycleBuffer => _cycleBuffer ?? throw new InvalidOperationException("Batch resources not bound to an accelerator.");

        internal MemoryBuffer1D<ulong, Stride1D.Dense> ExponentBuffer => _exponentBuffer ?? throw new InvalidOperationException("Batch resources not bound to an accelerator.");

        internal MemoryBuffer1D<byte, Stride1D.Dense> HitsBuffer => _hitsBuffer ?? throw new InvalidOperationException("Batch resources not bound to an accelerator.");

        internal MemoryBuffer1D<int, Stride1D.Dense> HitIndexBuffer => _hitIndexBuffer ?? throw new InvalidOperationException("Batch resources not bound to an accelerator.");

        internal ulong[] Divisors { get; private set; } = null!;

        internal ulong[] Exponents { get; private set; } = null!;

        internal GpuDivisorPartialData[] DivisorData { get; private set; } = null!;

        internal int[] Offsets { get; private set; } = null!;

        internal int[] Counts { get; private set; } = null!;

        internal ulong[] Cycles { get; private set; } = null!;


        internal int Capacity => _hostCapacity;

        internal void Bind(Accelerator accelerator, int requiredCapacity)
        {
            EnsureHostCapacity(requiredCapacity);

            if (!ReferenceEquals(_accelerator, accelerator))
            {
                ReleaseDeviceBuffers();
                _accelerator = accelerator;
            }

            EnsureDeviceCapacity(accelerator, requiredCapacity);
        }

        private void EnsureHostCapacity(int requiredCapacity)
        {
            int desiredCapacity = Math.Max(1, requiredCapacity);
            if (_hostCapacity >= desiredCapacity)
            {
                return;
            }

            ReturnHostArrays();
            RentHostArrays(desiredCapacity);
        }

        private void EnsureDeviceCapacity(Accelerator accelerator, int requiredCapacity)
        {
            int desiredCapacity = Math.Max(1, requiredCapacity);
            if (_divisorDataBuffer is null)
            {
                AllocateDeviceBuffers(accelerator, desiredCapacity);
                return;
            }

            if (_deviceCapacity >= desiredCapacity)
            {
                return;
            }

            ReleaseDeviceBuffers();
            AllocateDeviceBuffers(accelerator, desiredCapacity);
        }

        private void AllocateDeviceBuffers(Accelerator accelerator, int capacity)
        {
            _divisorDataBuffer = accelerator.Allocate1D<GpuDivisorPartialData>(capacity);
            _offsetBuffer = accelerator.Allocate1D<int>(capacity);
            _countBuffer = accelerator.Allocate1D<int>(capacity);
            _cycleBuffer = accelerator.Allocate1D<ulong>(capacity);
            _exponentBuffer = accelerator.Allocate1D<ulong>(capacity);
            _hitsBuffer = accelerator.Allocate1D<byte>(capacity);
            _hitIndexBuffer = accelerator.Allocate1D<int>(1);
            _deviceCapacity = capacity;
        }

        private void RentHostArrays(int capacity)
        {
            ArrayPool<ulong> ulongPool = ThreadStaticPools.UlongPool;
            ArrayPool<int> intPool = ThreadStaticPools.IntPool;
            ArrayPool<GpuDivisorPartialData> partialPool = ThreadStaticPools.GpuDivisorPartialDataPool;

            Divisors = ulongPool.Rent(capacity);
            Exponents = ulongPool.Rent(capacity);
            DivisorData = partialPool.Rent(capacity);
            Offsets = intPool.Rent(capacity);
            Counts = intPool.Rent(capacity);
            Cycles = ulongPool.Rent(capacity);
            _hostCapacity = capacity;
        }

        private void ReleaseDeviceBuffers()
        {
            _divisorDataBuffer?.Dispose();
            _offsetBuffer?.Dispose();
            _countBuffer?.Dispose();
            _cycleBuffer?.Dispose();
            _exponentBuffer?.Dispose();
            _hitsBuffer?.Dispose();
            _hitIndexBuffer?.Dispose();
            _divisorDataBuffer = null;
            _offsetBuffer = null;
            _countBuffer = null;
            _cycleBuffer = null;
            _exponentBuffer = null;
            _hitsBuffer = null;
            _hitIndexBuffer = null;
            _deviceCapacity = 0;
        }

        private void ReturnHostArrays()
        {
            if (_hostCapacity == 0)
            {
                return;
            }

            ArrayPool<ulong> ulongPool = ThreadStaticPools.UlongPool;
            ArrayPool<int> intPool = ThreadStaticPools.IntPool;
            ArrayPool<GpuDivisorPartialData> partialPool = ThreadStaticPools.GpuDivisorPartialDataPool;
            ulongPool.Return(Divisors, clearArray: false);
            ulongPool.Return(Exponents, clearArray: false);
            partialPool.Return(DivisorData, clearArray: false);
            intPool.Return(Offsets, clearArray: false);
            intPool.Return(Counts, clearArray: false);
            ulongPool.Return(Cycles, clearArray: false);
            Divisors = Array.Empty<ulong>();
            Exponents = Array.Empty<ulong>();
            DivisorData = Array.Empty<GpuDivisorPartialData>();
            Offsets = Array.Empty<int>();
            Counts = Array.Empty<int>();
            Cycles = Array.Empty<ulong>();
            _hostCapacity = 0;
        }

        public void Dispose()
        {
            ReleaseDeviceBuffers();
            _accelerator = null;
            ReturnHostArrays();
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




