using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using MontgomeryDivisorData = PerfectNumbers.Core.MontgomeryDivisorData;
using MontgomeryDivisorDataCache = PerfectNumbers.Core.MontgomeryDivisorDataCache;
using CycleRemainderStepper = PerfectNumbers.Core.CycleRemainderStepper;

namespace PerfectNumbers.Core.Gpu;

public sealed class MersenneNumberDivisorByDivisorGpuTester : IMersenneNumberDivisorByDivisorTester
{
    private int _gpuBatchSize = GpuConstants.ScanBatchSize;
    private readonly object _sync = new();
    private ulong _divisorLimit;
    private bool _isConfigured;
    private bool _useDivisorCycles;

    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<MontgomeryDivisorData>, ArrayView<ulong>, ArrayView<byte>>> _kernelCache = new();
    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>>> _kernelExponentCache = new();
    private readonly ConcurrentDictionary<Accelerator, ConcurrentBag<BatchResources>> _resourcePools = new(AcceleratorReferenceComparer.Instance);
    private readonly ConcurrentBag<DivisorScanSession> _sessionPool = new();

    private Action<Index1D, ArrayView<MontgomeryDivisorData>, ArrayView<ulong>, ArrayView<byte>> GetKernel(Accelerator accelerator) =>
        _kernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<MontgomeryDivisorData>, ArrayView<ulong>, ArrayView<byte>>(CheckKernel));

    private Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>> GetKernelByPrimeExponent(Accelerator accelerator) =>
        _kernelExponentCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>>(ComputeMontgomeryExponentKernel));

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

    public bool UseDivisorCycles
    {
        get => _useDivisorCycles;
        set => _useDivisorCycles = value;
    }

    public void ConfigureFromMaxPrime(ulong maxPrime)
    {
        lock (_sync)
        {
            _divisorLimit = ComputeDivisorLimitFromMaxPrime(maxPrime);
            _isConfigured = true;
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
            batchCapacity = _gpuBatchSize;
        }

        if (allowedMax < 3UL)
        {
            divisorsExhausted = true;
            return true;
        }

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
            useCycles,
            accelerator,
            kernel,
            resources.DivisorsBuffer,
            resources.ExponentBuffer,
            resources.HitsBuffer,
            resources.Divisors,
            resources.Exponents,
            resources.Hits,
            resources.DivisorData,
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

    private static bool CheckDivisors(
        ulong prime,
        ulong allowedMax,
        bool useCycles,
        Accelerator accelerator,
        Action<Index1D, ArrayView<MontgomeryDivisorData>, ArrayView<ulong>, ArrayView<byte>> kernel,
        MemoryBuffer1D<MontgomeryDivisorData, Stride1D.Dense> divisorsBuffer,
        MemoryBuffer1D<ulong, Stride1D.Dense> exponentBuffer,
        MemoryBuffer1D<byte, Stride1D.Dense> hitsBuffer,
        in ulong[] divisors,
        ulong[] exponents,
        byte[] hits,
        MontgomeryDivisorData[] divisorData,
        out ulong lastProcessed,
        out bool coveredRange,
        out ulong processedCount)
    {
        lastProcessed = 0UL;
        processedCount = 0UL;

        if (allowedMax < 3UL)
        {
            coveredRange = true;
            return false;
        }

        return useCycles
            ? CheckDivisorsWithCycles(
                prime,
                allowedMax,
                accelerator,
                kernel,
                divisorsBuffer,
                exponentBuffer,
                hitsBuffer,
                divisors,
                exponents,
                hits,
                divisorData,
                ref lastProcessed,
                ref processedCount,
                out coveredRange)
            : CheckDivisorsWithoutCycles(
                prime,
                allowedMax,
                accelerator,
                kernel,
                divisorsBuffer,
                exponentBuffer,
                hitsBuffer,
                divisors,
                exponents,
                hits,
                divisorData,
                ref lastProcessed,
                ref processedCount,
                out coveredRange);
    }

    private static bool CheckDivisorsWithoutCycles(
        ulong prime,
        ulong allowedMax,
        Accelerator accelerator,
        Action<Index1D, ArrayView<MontgomeryDivisorData>, ArrayView<ulong>, ArrayView<byte>> kernel,
        MemoryBuffer1D<MontgomeryDivisorData, Stride1D.Dense> divisorsBuffer,
        MemoryBuffer1D<ulong, Stride1D.Dense> exponentBuffer,
        MemoryBuffer1D<byte, Stride1D.Dense> hitsBuffer,
        in ulong[] divisors,
        ulong[] exponents,
        byte[] hits,
        MontgomeryDivisorData[] divisorData,
        ref ulong lastProcessed,
        ref ulong processedCount,
        out bool coveredRange)
    {
        int batchCapacity = (int)divisorsBuffer.Length;
        bool composite = false;
        bool processedAll = false;
        ulong currentDivisor = 3UL;

        Span<MontgomeryDivisorData> divisorDataSpan;
        Span<ulong> divisorSpan;
        Span<ulong> exponentSpan;
        Span<byte> hitsSpan;
        ArrayView1D<MontgomeryDivisorData, Stride1D.Dense> divisorView;
        ArrayView1D<ulong, Stride1D.Dense> exponentView;
        ArrayView1D<byte, Stride1D.Dense> hitsView;

        while (currentDivisor <= allowedMax)
        {
            int batchSize = 0;
            bool reachedEndInBatch = false;

            // The GPU buffers are sized exactly to the configured batch capacity, so we
            // must stop once the current batch is full to avoid writing past the spans.
            while (batchSize < batchCapacity && currentDivisor <= allowedMax)
            {
                ulong nextDivisor = currentDivisor + 2UL;
                processedCount++;

                divisors[batchSize] = currentDivisor;
                exponents[batchSize] = prime;
                batchSize++;
                lastProcessed = currentDivisor;

                if (nextDivisor <= currentDivisor)
                {
                    reachedEndInBatch = true;
                    break;
                }

                if (nextDivisor > allowedMax)
                {
                    currentDivisor = nextDivisor;
                    reachedEndInBatch = true;
                    break;
                }

                currentDivisor = nextDivisor;
            }

            if (batchSize == 0)
            {
                if (reachedEndInBatch)
                {
                    processedAll = true;
                    break;
                }

                continue;
            }

            divisorDataSpan = divisorData.AsSpan(0, batchSize);
            divisorSpan = divisors.AsSpan(0, batchSize);
            exponentSpan = exponents.AsSpan(0, batchSize);
            hitsSpan = hits.AsSpan(0, batchSize);
            divisorView = divisorsBuffer.View.SubView(0, batchSize);
            exponentView = exponentBuffer.View.SubView(0, batchSize);
            hitsView = hitsBuffer.View.SubView(0, batchSize);

            for (int i = 0; i < batchSize; i++)
            {
                divisorDataSpan[i] = MontgomeryDivisorDataCache.Get(divisorSpan[i]);
            }

            divisorView.CopyFromCPU(ref MemoryMarshal.GetReference(divisorDataSpan), batchSize);
            exponentView.CopyFromCPU(ref MemoryMarshal.GetReference(exponentSpan), batchSize);
            kernel(batchSize, divisorView, exponentView, hitsView);
            hitsView.CopyToCPU(ref MemoryMarshal.GetReference(hitsSpan), batchSize);

            for (int i = 0; i < batchSize; i++)
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

        coveredRange = composite || processedAll || currentDivisor > allowedMax;
        return composite;
    }

    private static bool CheckDivisorsWithCycles(
        ulong prime,
        ulong allowedMax,
        Accelerator accelerator,
        Action<Index1D, ArrayView<MontgomeryDivisorData>, ArrayView<ulong>, ArrayView<byte>> kernel,
        MemoryBuffer1D<MontgomeryDivisorData, Stride1D.Dense> divisorsBuffer,
        MemoryBuffer1D<ulong, Stride1D.Dense> exponentBuffer,
        MemoryBuffer1D<byte, Stride1D.Dense> hitsBuffer,
        in ulong[] divisors,
        ulong[] exponents,
        byte[] hits,
        MontgomeryDivisorData[] divisorData,
        ref ulong lastProcessed,
        ref ulong processedCount,
        out bool coveredRange)
    {
        int batchCapacity = (int)divisorsBuffer.Length;
        bool composite = false;
        bool processedAll = false;
        ulong currentDivisor = 3UL;
        ulong cycleLeaseEnd;

        Span<MontgomeryDivisorData> divisorDataSpan;
        Span<ulong> divisorSpan;
        Span<ulong> exponentSpan;
        Span<byte> hitsSpan;
        ArrayView1D<MontgomeryDivisorData, Stride1D.Dense> divisorView;
        ArrayView1D<ulong, Stride1D.Dense> exponentView;
        ArrayView1D<byte, Stride1D.Dense> hitsView;

        DivisorCycleCache divisorCyclesCache = DivisorCycleCache.Shared;
        DivisorCycleCache.CycleBlock cycleLease = divisorCyclesCache.Acquire(currentDivisor);
        cycleLeaseEnd = cycleLease.End;

        try
        {
            while (currentDivisor <= allowedMax)
            {
                int batchSize = 0;
                bool reachedEndInBatch = false;

                // Batch writes share the same buffers as the kernel launch, so we guard
                // against exceeding the rented capacity on each iteration.
                while (batchSize < batchCapacity && currentDivisor <= allowedMax)
                {
                    ulong nextDivisor = currentDivisor + 2UL;
                    processedCount++;

                    if (currentDivisor > cycleLeaseEnd)
                    {
                        cycleLease.Dispose();
                        cycleLease = divisorCyclesCache.Acquire(currentDivisor);
                        cycleLeaseEnd = cycleLease.End;
                    }

                    bool includeDivisor = true;
                    ulong exponent = prime;
                    ulong cycleLength = cycleLease.GetCycle(currentDivisor);
                    if (cycleLength > 0UL)
                    {
                        ulong remainder = prime;
                        if (remainder >= cycleLength)
                        {
                            remainder %= cycleLength;
                        }

                        if (remainder != 0UL)
                        {
                            includeDivisor = false;
                        }
                        else
                        {
                            exponent = 0UL;
                        }
                    }

                    if (includeDivisor)
                    {
                        divisors[batchSize] = currentDivisor;
                        exponents[batchSize] = exponent;
                        batchSize++;
                    }

                    lastProcessed = currentDivisor;

                    if (nextDivisor <= currentDivisor)
                    {
                        reachedEndInBatch = true;
                        break;
                    }

                    if (nextDivisor > allowedMax)
                    {
                        currentDivisor = nextDivisor;
                        reachedEndInBatch = true;
                        break;
                    }

                    currentDivisor = nextDivisor;
                }

                if (batchSize == 0)
                {
                    if (reachedEndInBatch)
                    {
                        processedAll = true;
                        break;
                    }

                    continue;
                }

                divisorDataSpan = divisorData.AsSpan(0, batchSize);
                divisorSpan = divisors.AsSpan(0, batchSize);
                exponentSpan = exponents.AsSpan(0, batchSize);
                hitsSpan = hits.AsSpan(0, batchSize);
                divisorView = divisorsBuffer.View.SubView(0, batchSize);
                exponentView = exponentBuffer.View.SubView(0, batchSize);
                hitsView = hitsBuffer.View.SubView(0, batchSize);

                for (int i = 0; i < batchSize; i++)
                {
                    divisorDataSpan[i] = MontgomeryDivisorDataCache.Get(divisorSpan[i]);
                }

                divisorView.CopyFromCPU(ref MemoryMarshal.GetReference(divisorDataSpan), batchSize);
                exponentView.CopyFromCPU(ref MemoryMarshal.GetReference(exponentSpan), batchSize);
                kernel(batchSize, divisorView, exponentView, hitsView);
                hitsView.CopyToCPU(ref MemoryMarshal.GetReference(hitsSpan), batchSize);

                for (int i = 0; i < batchSize; i++)
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
        }
        finally
        {
            cycleLease.Dispose();
        }

        coveredRange = composite || processedAll || currentDivisor > allowedMax;
        return composite;
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

    private static void CheckKernel(Index1D index, ArrayView<MontgomeryDivisorData> divisors, ArrayView<ulong> exponents, ArrayView<byte> hits)
    {
        MontgomeryDivisorData divisor = divisors[index];
        ulong modulus = divisor.Modulus;
        if (modulus <= 1UL || (modulus & 1UL) == 0UL)
        {
            hits[index] = 0;
            return;
        }

        ulong exponent = exponents[index];
        hits[index] = exponent.Pow2MontgomeryMod(divisor) == 1UL ? (byte)1 : (byte)0;
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
        private int[]? _positionBuffer;
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
            _positionBuffer = null;
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

            MersenneNumberDivisorByDivisorGpuTester owner = _owner;
            int primesLength = primes.Length;
            if (primesLength == 0)
            {
                return;
            }

            // The production pipeline never adjusts the GPU batch size after a session
            // is initialized, so the buffers allocated in the constructor remain
            // large enough for every launch.

            MontgomeryDivisorData divisorData = MontgomeryDivisorDataCache.Get(divisor);
            var exponentStepper = new ExponentRemainderStepper(divisorData);
            if (!exponentStepper.IsValidModulus)
            {
                hits.Clear();
                return;
            }

            bool useDivisorCycles = owner._useDivisorCycles;
            if (!useDivisorCycles)
            {
                ProcessWithoutCycle(ref exponentStepper, divisorData, primes, hits);
                return;
            }

            if (divisorCycle == 0UL)
            {
                throw new InvalidOperationException($"Missing divisor cycle for divisor {divisor}.");
            }

            var cycleStepper = new CycleRemainderStepper(divisorCycle);
            ulong initialCycleRemainder = cycleStepper.Initialize(primes[0]);
            ProcessWithCycle(ref exponentStepper, ref cycleStepper, initialCycleRemainder, divisorCycle, divisorData, primes, hits);
        }

        private void ProcessWithoutCycle(
        ref ExponentRemainderStepper exponentStepper,
        in MontgomeryDivisorData divisorData,
        ReadOnlySpan<ulong> primes,
        Span<byte> hits)
        {
            Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>> kernel = _kernel;
            ArrayView1D<ulong, Stride1D.Dense> exponentsView = _exponentsBuffer.View;
            ArrayView1D<ulong, Stride1D.Dense> resultsView = _resultsBuffer.View;
            Span<ulong> hostSpan = _hostBuffer.AsSpan(0, _capacity);

            int primesLength = primes.Length;
            int gpuBatchSize = _owner._gpuBatchSize;
            int offset = 0;

            while (offset < primesLength)
            {
                int batchSize = Math.Min(gpuBatchSize, primesLength - offset);
                ReadOnlySpan<ulong> primesSlice = primes.Slice(offset, batchSize);
                Span<byte> hitsSlice = hits.Slice(offset, batchSize);
                Span<ulong> exponentSlice = hostSpan[..batchSize];

                bool hasState = exponentStepper.HasState;
                ulong previousExponent = exponentStepper.PreviousExponent;

                int startIndex = 0;
                if (!hasState)
                {
                    ulong primeValue = primesSlice[0];
                    exponentSlice[0] = primeValue;
                    previousExponent = primeValue;
                    hasState = true;
                    startIndex = 1;
                }

                for (int i = startIndex; i < batchSize; i++)
                {
                    ulong primeValue = primesSlice[i];
                    exponentSlice[i] = primeValue - previousExponent;
                    previousExponent = primeValue;
                }

                ArrayView1D<ulong, Stride1D.Dense> exponentView = exponentsView.SubView(0, batchSize);
                exponentView.CopyFromCPU(ref MemoryMarshal.GetReference(exponentSlice), batchSize);

                ArrayView1D<ulong, Stride1D.Dense> resultView = resultsView.SubView(0, batchSize);
                kernel(batchSize, divisorData, exponentView, resultView);
                resultView.CopyToCPU(ref MemoryMarshal.GetReference(exponentSlice), batchSize);

                bool requiresInitialization = !exponentStepper.HasState;
                for (int i = 0; i < batchSize; i++)
                {
                    ulong primeValue = primesSlice[i];
                    ulong montgomeryResult = exponentSlice[i];
                    bool isUnity;

                    if (requiresInitialization)
                    {
                        exponentStepper.TryInitializeFromMontgomeryResult(primeValue, montgomeryResult, out isUnity);
                        requiresInitialization = false;
                    }
                    else
                    {
                        exponentStepper.TryAdvanceWithMontgomeryDelta(primeValue, montgomeryResult, out isUnity);
                    }

                    hitsSlice[i] = isUnity ? (byte)1 : (byte)0;
                }

                offset += batchSize;
            }
        }

        private void ProcessWithCycle(
        ref ExponentRemainderStepper exponentStepper,
        ref CycleRemainderStepper cycleStepper,
        ulong initialCycleRemainder,
        ulong divisorCycle,
        in MontgomeryDivisorData divisorData,
        ReadOnlySpan<ulong> primes,
        Span<byte> hits)
        {
            Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>> kernel = _kernel;
            ArrayView1D<ulong, Stride1D.Dense> exponentsView = _exponentsBuffer.View;
            ArrayView1D<ulong, Stride1D.Dense> resultsView = _resultsBuffer.View;
            int capacity = _capacity;
            Span<ulong> hostSpan = _hostBuffer.AsSpan(0, capacity);
            int[]? positionBuffer = _positionBuffer;
            if (positionBuffer is null)
            {
                // Session construction already enforces the batch capacity, so we only
                // need to rent the position buffer on the first cycle-enabled invocation.
                positionBuffer = ArrayPool<int>.Shared.Rent(capacity);
                _positionBuffer = positionBuffer;
            }

            Span<int> positionSpan = positionBuffer.AsSpan(0, capacity);

            int gpuBatchSize = _owner._gpuBatchSize;
            int primesLength = primes.Length;
            int offset = 0;
            bool cycleRemainderPending = true;

            while (offset < primesLength)
            {
                int batchSize = Math.Min(gpuBatchSize, primesLength - offset);
                ReadOnlySpan<ulong> primesSlice = primes.Slice(offset, batchSize);
                Span<byte> hitsSlice = hits.Slice(offset, batchSize);

                int computeCount = 0;
                bool hasState = exponentStepper.HasState;
                ulong previousExponent = exponentStepper.PreviousExponent;
                bool useInitialRemainder = cycleRemainderPending;

                for (int i = 0; i < batchSize; i++)
                {
                    hitsSlice[i] = 0;
                    ulong primeValue = primesSlice[i];

                    ulong remainder = useInitialRemainder
                        ? initialCycleRemainder
                        : cycleStepper.ComputeNext(primeValue);
                    useInitialRemainder = false;

                    if (remainder != 0UL)
                    {
                        continue;
                    }

                    ulong workValue;
                    if (!hasState)
                    {
                        workValue = primeValue;
                        hasState = true;
                    }
                    else
                    {
                        workValue = primeValue - previousExponent;
                    }

                    previousExponent = primeValue;

                    if (workValue >= divisorCycle)
                    {
                        workValue %= divisorCycle;
                    }

                    hostSpan[computeCount] = workValue;
                    positionSpan[computeCount] = i;
                    computeCount++;
                }

                cycleRemainderPending = useInitialRemainder;

                if (computeCount > 0)
                {
                    Span<ulong> exponentSlice = hostSpan[..computeCount];
                    ArrayView1D<ulong, Stride1D.Dense> exponentView = exponentsView.SubView(0, computeCount);
                    exponentView.CopyFromCPU(ref MemoryMarshal.GetReference(exponentSlice), computeCount);

                    ArrayView1D<ulong, Stride1D.Dense> resultView = resultsView.SubView(0, computeCount);
                    kernel(computeCount, divisorData, exponentView, resultView);
                    resultView.CopyToCPU(ref MemoryMarshal.GetReference(exponentSlice), computeCount);

                    bool requiresInitialization = !exponentStepper.HasState;
                    for (int i = 0; i < computeCount; i++)
                    {
                        int position = positionSpan[i];
                        ulong primeValue = primesSlice[position];
                        ulong montgomeryResult = exponentSlice[i];
                        bool isUnity;

                        if (requiresInitialization)
                        {
                            exponentStepper.TryInitializeFromMontgomeryResult(primeValue, montgomeryResult, out isUnity);
                            requiresInitialization = false;
                        }
                        else
                        {
                            exponentStepper.TryAdvanceWithMontgomeryDelta(primeValue, montgomeryResult, out isUnity);
                        }

                        hitsSlice[position] = isUnity ? (byte)1 : (byte)0;
                    }
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

    private static void ComputeMontgomeryExponentKernel(Index1D index, MontgomeryDivisorData divisor, ArrayView<ulong> exponents, ArrayView<ulong> results)
    {
        ulong modulus = divisor.Modulus;
        if (modulus <= 1UL || (modulus & 1UL) == 0UL)
        {
            results[index] = 0UL;
            return;
        }

        ulong exponent = exponents[index];

        results[index] = exponent.Pow2MontgomeryModMontgomery(divisor);
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
            DivisorsBuffer = accelerator.Allocate1D<MontgomeryDivisorData>(capacity);
            ExponentBuffer = accelerator.Allocate1D<ulong>(capacity);
            HitsBuffer = accelerator.Allocate1D<byte>(capacity);
            Divisors = ArrayPool<ulong>.Shared.Rent(capacity);
            Exponents = ArrayPool<ulong>.Shared.Rent(capacity);
            Hits = ArrayPool<byte>.Shared.Rent(capacity);
            DivisorData = ArrayPool<MontgomeryDivisorData>.Shared.Rent(capacity);
            Capacity = capacity;
        }

        internal readonly MemoryBuffer1D<MontgomeryDivisorData, Stride1D.Dense> DivisorsBuffer;

        internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> ExponentBuffer;

        internal readonly MemoryBuffer1D<byte, Stride1D.Dense> HitsBuffer;

        internal readonly ulong[] Divisors;

        internal readonly ulong[] Exponents;

        internal readonly byte[] Hits;

        internal readonly MontgomeryDivisorData[] DivisorData;

        internal readonly int Capacity;

        public void Dispose()
        {
            DivisorsBuffer.Dispose();
            ExponentBuffer.Dispose();
            HitsBuffer.Dispose();
            ArrayPool<ulong>.Shared.Return(Divisors, clearArray: false);
            ArrayPool<ulong>.Shared.Return(Exponents, clearArray: false);
            ArrayPool<byte>.Shared.Return(Hits, clearArray: false);
            ArrayPool<MontgomeryDivisorData>.Shared.Return(DivisorData, clearArray: false);
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
        lock (_sync)
        {
            if (!_isConfigured)
            {
                throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
            }

            return ComputeAllowedMaxDivisor(prime, _divisorLimit);
        }
    }

    public DivisorScanSession CreateDivisorSession()
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

    IMersenneNumberDivisorByDivisorTester.IDivisorScanSession IMersenneNumberDivisorByDivisorTester.CreateDivisorSession() => CreateDivisorSession();
}

