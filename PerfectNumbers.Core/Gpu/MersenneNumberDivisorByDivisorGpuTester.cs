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
/// </summary>
public sealed class MersenneNumberDivisorByDivisorGpuTester : IMersenneNumberDivisorByDivisorTester
{
    private const int RemainderSlotCount = 4;

    private int _gpuBatchSize = GpuConstants.ScanBatchSize;
    // EvenPerfectBitScanner configures the GPU tester once before scanning and never mutates the configuration afterwards,
    // so the synchronization fields from the previous implementation remain commented out here.
    // private readonly object _sync = new();
    private ulong _divisorLimit;
    // private bool _isConfigured;

    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<MontgomeryDivisorData>, ArrayView<ulong>, ArrayView<byte>>> _kernelCache = new();
    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>>> _kernelExponentCache = new();
    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<ulong>, ArrayView<byte>, byte>> _remainderDeltaKernelCache = new(AcceleratorReferenceComparer.Instance);
    private readonly ConcurrentDictionary<Accelerator, Action<KernelConfig, Index1D, ArrayView<byte>, ArrayView<byte>, byte, byte>> _remainderScanKernelCache = new(AcceleratorReferenceComparer.Instance);
    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, byte, ArrayView<byte>>> _candidateMaskKernelCache = new(AcceleratorReferenceComparer.Instance);
    private readonly ConcurrentDictionary<Accelerator, ConcurrentBag<BatchResources>> _resourcePools = new(AcceleratorReferenceComparer.Instance);
    private readonly ConcurrentBag<GpuContextPool.GpuContextLease> _acceleratorPool = new();
    private readonly ConcurrentBag<DivisorScanSession> _sessionPool = new();

    private Action<Index1D, ArrayView<MontgomeryDivisorData>, ArrayView<ulong>, ArrayView<byte>> GetKernel(Accelerator accelerator) =>
        _kernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<MontgomeryDivisorData>, ArrayView<ulong>, ArrayView<byte>>(DivisorByDivisorKernels.CheckKernel));

    private Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>> GetKernelByPrimeExponent(Accelerator accelerator) =>
        _kernelExponentCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>>(DivisorByDivisorKernels.ComputeMontgomeryExponentKernel));

    private Action<Index1D, ArrayView<ulong>, ArrayView<byte>, byte> GetRemainderDeltaKernel(Accelerator accelerator) =>
        _remainderDeltaKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<byte>, byte>(DivisorByDivisorKernels.ComputeRemainderDeltasKernel));

    private Action<KernelConfig, Index1D, ArrayView<byte>, ArrayView<byte>, byte, byte> GetRemainderScanKernel(Accelerator accelerator) =>
        _remainderScanKernelCache.GetOrAdd(
            accelerator,
            acc => acc.LoadStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, byte, byte>(DivisorByDivisorKernels.AccumulateRemaindersKernel));

    private Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, byte, ArrayView<byte>> GetCandidateMaskKernel(Accelerator accelerator) =>
        _candidateMaskKernelCache.GetOrAdd(
            accelerator,
            acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, byte, ArrayView<byte>>(DivisorByDivisorKernels.EvaluateCandidateMaskKernel));

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
            accelerator,
            kernel,
            resources.DivisorsBuffer,
            resources.ExponentBuffer,
            resources.HitsBuffer,
            resources.DivisorDeltaBuffer,
            resources.RemainderDeltaBuffer,
            resources.RemainderBuffer,
            resources.RemainderDeltaKernel,
            resources.RemainderScanKernel,
            resources.CandidateMaskKernel,
            resources.Divisors,
            resources.Exponents,
            resources.Hits,
            resources.DivisorData,
            resources.CycleCandidates,
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
        Accelerator accelerator,
        Action<Index1D, ArrayView<MontgomeryDivisorData>, ArrayView<ulong>, ArrayView<byte>> kernel,
        MemoryBuffer1D<MontgomeryDivisorData, Stride1D.Dense> divisorsBuffer,
        MemoryBuffer1D<ulong, Stride1D.Dense> exponentBuffer,
        MemoryBuffer1D<byte, Stride1D.Dense> hitsBuffer,
        MemoryBuffer1D<ulong, Stride1D.Dense> divisorDeltaBuffer,
        MemoryBuffer1D<byte, Stride1D.Dense> remainderDeltaBuffer,
        MemoryBuffer1D<byte, Stride1D.Dense> remainderBuffer,
        Action<Index1D, ArrayView<ulong>, ArrayView<byte>, byte> remainderDeltaKernel,
        Action<KernelConfig, Index1D, ArrayView<byte>, ArrayView<byte>, byte, byte> remainderScanKernel,
        Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, byte, ArrayView<byte>> candidateMaskKernel,
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

        cycleCapacity = Math.Max(1, cycleCapacity);
        int chunkCapacity = Math.Min(batchCapacity, cycleCapacity);
        int maxThreadsPerGroup = (int)accelerator.MaxNumThreadsPerGroup;
        bool clampThreads = maxThreadsPerGroup > 0;
        chunkCapacity = clampThreads && chunkCapacity > maxThreadsPerGroup ? maxThreadsPerGroup : chunkCapacity;
        chunkCapacity = chunkCapacity <= 0 ? cycleCapacity : chunkCapacity;

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

        UInt128 currentDivisor128 = firstDivisor128;
        byte remainder10 = (byte)((ulong)(currentDivisor128 % 10UL));
        byte remainder8 = (byte)((ulong)(currentDivisor128 & 7UL));
        byte remainder5 = (byte)((ulong)(currentDivisor128 % 5UL));
        byte remainder3 = (byte)((ulong)(currentDivisor128 % 3UL));
        bool lastIsSeven = (prime & 3UL) == 3UL;
        byte lastIsSevenFlag = lastIsSeven ? (byte)1 : (byte)0;
        DivisorCycleCache cycleCache = DivisorCycleCache.Shared;
        int remainderStride = (int)divisorDeltaBuffer.Length;
        remainderStride = remainderStride <= 0 ? chunkCapacity : remainderStride;

        ArrayPool<ulong> ulongPool = ThreadStaticPools.UlongPool;
        ulong[] filteredDivisors = ulongPool.Rent(chunkCapacity);
        ulong[] divisorGaps = ulongPool.Rent(chunkCapacity);
        var viewCache = new BufferSliceCache(divisorDeltaBuffer, remainderDeltaBuffer, remainderBuffer, hitsBuffer, remainderStride);
        Span<ulong> candidateStorage = cycleCandidates.AsSpan();
        Span<ulong> gapStorage = divisorGaps.AsSpan();
        Span<byte> hitStorage = hits.AsSpan();
        Span<ulong> filteredStorage = filteredDivisors.AsSpan();
        Span<ulong> cycleLengthStorage = cycleLengths.AsSpan();
        Span<MontgomeryDivisorData> divisorDataStorage = divisorData.AsSpan();
        Span<ulong> divisorStorage = divisors.AsSpan();
        Span<ulong> exponentStorage = exponents.AsSpan();

        while (currentK <= maxK && !composite)
        {
            int chunkCount = Math.Min(chunkCapacity, batchCapacity);
            ulong remainingK = maxK - currentK + 1UL;
            chunkCount = (ulong)chunkCount > remainingK ? (int)remainingK : chunkCount;
            bool hasChunk = chunkCount > 0;
            int effectiveCount = hasChunk ? chunkCount : 0;
            if (!hasChunk)
            {
                processedAll = true;
                break;
            }

            Span<ulong> candidateSpan = candidateStorage[..effectiveCount];
            Span<ulong> gapSpan = gapStorage[..effectiveCount];
            Span<byte> maskSpan = hitStorage[..effectiveCount];

            UInt128 localDivisor = currentDivisor128;

            for (int i = 0; i < effectiveCount; i++)
            {
                ulong divisorValue = (ulong)localDivisor;
                candidateSpan[i] = divisorValue;
                processedCount++;
                lastProcessed = divisorValue;
                localDivisor += twoP128;
            }

            gapSpan[0] = 0UL;
            for (int i = 1; i < effectiveCount; i++)
            {
                gapSpan[i] = candidateSpan[i] - candidateSpan[i - 1];
            }

            currentDivisor128 = localDivisor;

            viewCache.Ensure(effectiveCount);

            ArrayView1D<ulong, Stride1D.Dense> gapView = viewCache.DivisorDelta;
            ArrayView1D<byte, Stride1D.Dense> deltaView = viewCache.RemainderDelta;
            ArrayView1D<byte, Stride1D.Dense> remainder10View = viewCache.Remainder10;
            ArrayView1D<byte, Stride1D.Dense> remainder8View = viewCache.Remainder8;
            ArrayView1D<byte, Stride1D.Dense> remainder5View = viewCache.Remainder5;
            ArrayView1D<byte, Stride1D.Dense> remainder3View = viewCache.Remainder3;
            ArrayView1D<byte, Stride1D.Dense> maskView = viewCache.Hits;

            gapView.CopyFromCPU(ref MemoryMarshal.GetReference(gapSpan), effectiveCount);

            ComputeRemaindersOnGpu(effectiveCount, remainder10, 10, gapView, deltaView, remainder10View, remainderDeltaKernel, remainderScanKernel);
            ComputeRemaindersOnGpu(effectiveCount, remainder8, 8, gapView, deltaView, remainder8View, remainderDeltaKernel, remainderScanKernel);
            ComputeRemaindersOnGpu(effectiveCount, remainder5, 5, gapView, deltaView, remainder5View, remainderDeltaKernel, remainderScanKernel);
            ComputeRemaindersOnGpu(effectiveCount, remainder3, 3, gapView, deltaView, remainder3View, remainderDeltaKernel, remainderScanKernel);

            BuildCandidateMaskOnGpu(effectiveCount, remainder10View, remainder8View, remainder5View, remainder3View, lastIsSevenFlag, maskView, maskSpan, candidateMaskKernel);

            byte lastRemainder10 = FetchLastRemainder(remainder10View, effectiveCount);
            byte lastRemainder8 = FetchLastRemainder(remainder8View, effectiveCount);
            byte lastRemainder5 = FetchLastRemainder(remainder5View, effectiveCount);
            byte lastRemainder3 = FetchLastRemainder(remainder3View, effectiveCount);

            remainder10 = AddModGpu(lastRemainder10, step10, 10);
            remainder8 = AddModGpu(lastRemainder8, step8, 8);
            remainder5 = AddModGpu(lastRemainder5, step5, 5);
            remainder3 = AddModGpu(lastRemainder3, step3, 3);

            int filteredCount = 0;
            Span<ulong> filteredDivisorsSpan = filteredStorage[..effectiveCount];

            for (int i = 0; i < effectiveCount; i++)
            {
                int add = maskSpan[i] == 0 ? 0 : 1;
                filteredDivisorsSpan[filteredCount] = candidateSpan[i];
                filteredCount += add;
            }

            currentK += (ulong)effectiveCount;
            processedAll = currentK > maxK;

            if (filteredCount == 0)
            {
                continue;
            }

            Span<ulong> cycleSpan = cycleLengthStorage[..filteredCount];
            cycleCache.GetCycleLengths(filteredDivisorsSpan[..filteredCount], cycleSpan);

            int admissibleCount = 0;
            for (int i = 0; i < filteredCount; i++)
            {
                int add = cycleSpan[i] == prime ? 1 : 0;
                filteredDivisorsSpan[admissibleCount] = filteredDivisorsSpan[i];
                admissibleCount += add;
            }

            if (admissibleCount == 0)
            {
                continue;
            }

            Span<MontgomeryDivisorData> divisorDataSpan = divisorDataStorage[..admissibleCount];
            Span<ulong> divisorSpan = divisorStorage[..admissibleCount];
            Span<ulong> exponentSpan = exponentStorage[..admissibleCount];
            Span<byte> hitsSpan = hitStorage[..admissibleCount];
            ArrayView1D<MontgomeryDivisorData, Stride1D.Dense> divisorView = divisorsBuffer.View.SubView(0, admissibleCount);
            ArrayView1D<ulong, Stride1D.Dense> exponentView = exponentBuffer.View.SubView(0, admissibleCount);
            ArrayView1D<byte, Stride1D.Dense> hitsView = hitsBuffer.View.SubView(0, admissibleCount);

            for (int i = 0; i < admissibleCount; i++)
            {
                ulong divisorValue = filteredDivisorsSpan[i];
                divisorSpan[i] = divisorValue;
                exponentSpan[i] = prime;
                divisorDataSpan[i] = MontgomeryDivisorData.FromModulus(divisorValue);
            }

            divisorView.CopyFromCPU(ref MemoryMarshal.GetReference(divisorDataSpan), admissibleCount);
            exponentView.CopyFromCPU(ref MemoryMarshal.GetReference(exponentSpan), admissibleCount);
            kernel(admissibleCount, divisorView, exponentView, hitsView);
            hitsView.CopyToCPU(ref MemoryMarshal.GetReference(hitsSpan), admissibleCount);

            int hitIndex = System.MemoryExtensions.IndexOf(hitsSpan, (byte)1);
            bool hitFound = hitIndex >= 0;
            composite = hitFound;
            int lastIndex = admissibleCount - 1;
            lastProcessed = hitFound ? divisorSpan[hitIndex] : divisorSpan[lastIndex];
        }

        ulongPool.Return(filteredDivisors, clearArray: false);
        ulongPool.Return(divisorGaps, clearArray: false);

        coveredRange = composite || processedAll || (currentDivisor128 > allowedMax128);
        return composite;
    }

    private ref struct BufferSliceCache
    {
        private readonly MemoryBuffer1D<ulong, Stride1D.Dense> _divisorDeltaBuffer;
        private readonly MemoryBuffer1D<byte, Stride1D.Dense> _remainderDeltaBuffer;
        private readonly MemoryBuffer1D<byte, Stride1D.Dense> _remainderBuffer;
        private readonly MemoryBuffer1D<byte, Stride1D.Dense> _hitsBuffer;
        private readonly int _remainderStride;
        private int _length;
        private ArrayView1D<ulong, Stride1D.Dense> _divisorDelta;
        private ArrayView1D<byte, Stride1D.Dense> _remainderDelta;
        private ArrayView1D<byte, Stride1D.Dense> _remainder10;
        private ArrayView1D<byte, Stride1D.Dense> _remainder8;
        private ArrayView1D<byte, Stride1D.Dense> _remainder5;
        private ArrayView1D<byte, Stride1D.Dense> _remainder3;
        private ArrayView1D<byte, Stride1D.Dense> _hits;

        internal BufferSliceCache(
            MemoryBuffer1D<ulong, Stride1D.Dense> divisorDeltaBuffer,
            MemoryBuffer1D<byte, Stride1D.Dense> remainderDeltaBuffer,
            MemoryBuffer1D<byte, Stride1D.Dense> remainderBuffer,
            MemoryBuffer1D<byte, Stride1D.Dense> hitsBuffer,
            int remainderStride)
        {
            _divisorDeltaBuffer = divisorDeltaBuffer;
            _remainderDeltaBuffer = remainderDeltaBuffer;
            _remainderBuffer = remainderBuffer;
            _hitsBuffer = hitsBuffer;
            _remainderStride = remainderStride;
            _length = -1;
            _divisorDelta = default;
            _remainderDelta = default;
            _remainder10 = default;
            _remainder8 = default;
            _remainder5 = default;
            _remainder3 = default;
            _hits = default;
        }

        internal void Ensure(int length)
        {
            if (_length == length)
            {
                return;
            }

            _length = length;
            ArrayView1D<byte, Stride1D.Dense> remainderBaseView = _remainderBuffer.View;
            _divisorDelta = _divisorDeltaBuffer.View.SubView(0, length);
            _remainderDelta = _remainderDeltaBuffer.View.SubView(0, length);
            _remainder10 = remainderBaseView.SubView(0, length);
            _remainder8 = remainderBaseView.SubView(_remainderStride, length);
            _remainder5 = remainderBaseView.SubView(_remainderStride * 2, length);
            _remainder3 = remainderBaseView.SubView(_remainderStride * 3, length);
            _hits = _hitsBuffer.View.SubView(0, length);
        }

        internal ArrayView1D<ulong, Stride1D.Dense> DivisorDelta => _divisorDelta;

        internal ArrayView1D<byte, Stride1D.Dense> RemainderDelta => _remainderDelta;

        internal ArrayView1D<byte, Stride1D.Dense> Remainder10 => _remainder10;

        internal ArrayView1D<byte, Stride1D.Dense> Remainder8 => _remainder8;

        internal ArrayView1D<byte, Stride1D.Dense> Remainder5 => _remainder5;

        internal ArrayView1D<byte, Stride1D.Dense> Remainder3 => _remainder3;

        internal ArrayView1D<byte, Stride1D.Dense> Hits => _hits;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ComputeRemaindersOnGpu(
        int count,
        byte baseRemainder,
        byte modulus,
        ArrayView1D<ulong, Stride1D.Dense> gapView,
        ArrayView1D<byte, Stride1D.Dense> deltaView,
        ArrayView1D<byte, Stride1D.Dense> remainderView,
        Action<Index1D, ArrayView<ulong>, ArrayView<byte>, byte> deltaKernel,
        Action<KernelConfig, Index1D, ArrayView<byte>, ArrayView<byte>, byte, byte> scanKernel)
    {
        if (count <= 0)
        {
            return;
        }

        deltaKernel(count, gapView, deltaView, modulus);
        SharedMemoryConfig sharedMemoryConfig = SharedMemoryConfig.RequestDynamic<int>(count);
        Index1D gridDim = new Index1D(1);
        Index1D groupDim = new Index1D(count);
        KernelConfig kernelConfig = new KernelConfig(gridDim, groupDim, sharedMemoryConfig);

        scanKernel(kernelConfig, new Index1D(count), deltaView, remainderView, baseRemainder, modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void BuildCandidateMaskOnGpu(
        int count,
        ArrayView1D<byte, Stride1D.Dense> remainder10View,
        ArrayView1D<byte, Stride1D.Dense> remainder8View,
        ArrayView1D<byte, Stride1D.Dense> remainder5View,
        ArrayView1D<byte, Stride1D.Dense> remainder3View,
        byte lastIsSevenFlag,
        ArrayView1D<byte, Stride1D.Dense> maskView,
        Span<byte> destination,
        Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, byte, ArrayView<byte>> maskKernel)
    {
        if (count <= 0)
        {
            destination.Clear();
            return;
        }

        maskKernel(count, remainder10View, remainder8View, remainder5View, remainder3View, lastIsSevenFlag, maskView);
        maskView.CopyToCPU(ref MemoryMarshal.GetReference(destination), count);
    }

    private static byte FetchLastRemainder(ArrayView1D<byte, Stride1D.Dense> remainderView, int count)
    {
        if (count <= 0)
        {
            return 0;
        }

        byte value = 0;
        remainderView.SubView(count - 1, 1).CopyToCPU(ref value, 1);
        return value;
    }

    private static byte AddModGpu(byte value, byte delta, byte modulus)
    {
        int result = value + delta;
        int firstWrap = result >= modulus ? 1 : 0;
        result -= firstWrap * modulus;
        int secondWrap = result >= modulus ? 1 : 0;
        result -= secondWrap * modulus;
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
        private Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>> _kernel;
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
            _capacity = Math.Max(1, owner._gpuBatchSize);
        }

        internal void Reset()
        {
            _disposed = false;
        }

        private void EnsureExecutionResourcesLocked()
        {
            _kernel = _owner.GetKernelByPrimeExponent(_accelerator);
            _exponentsBuffer = _accelerator.Allocate1D<ulong>(_capacity);
            _resultsBuffer = _accelerator.Allocate1D<ulong>(_capacity);
            _hostBuffer = ThreadStaticPools.UlongPool.Rent(_capacity);
        }

        public void CheckDivisor(ulong divisor, in MontgomeryDivisorData divisorData, ulong divisorCycle, in ReadOnlySpan<ulong> primes, Span<byte> hits)
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

            if (divisorData.Modulus <= 1UL || (divisorData.Modulus & 1UL) == 0UL)
            {
                hits.Clear();
                return;
            }

            Monitor.Enter(_lease.ExecutionLock);

            if (_kernel == null)
            {
                EnsureExecutionResourcesLocked();
            }

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

        return new BatchResources(this, accelerator, capacity);
    }

    private void ReturnBatchResources(Accelerator accelerator, BatchResources resources) => _resourcePools.GetOrAdd(accelerator, static _ => []).Add(resources);

    private sealed class BatchResources : IDisposable
    {
        internal BatchResources(MersenneNumberDivisorByDivisorGpuTester owner, Accelerator accelerator, int capacity)
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
            DivisorDeltaBuffer = accelerator.Allocate1D<ulong>(actualCapacity);
            RemainderDeltaBuffer = accelerator.Allocate1D<byte>(actualCapacity);
            RemainderBuffer = accelerator.Allocate1D<byte>(actualCapacity * RemainderSlotCount);
            ArrayPool<ulong> ulongPool = ThreadStaticPools.UlongPool;
            Divisors = ulongPool.Rent(actualCapacity);
            Exponents = ulongPool.Rent(actualCapacity);
            Hits = ThreadStaticPools.BytePool.Rent(actualCapacity);
            DivisorData = ThreadStaticPools.MontgomeryDivisorDataPool.Rent(actualCapacity);
            CycleCandidates = ulongPool.Rent(cycleCapacity);
            CycleLengths = ulongPool.Rent(cycleCapacity);
            RemainderDeltaKernel = owner.GetRemainderDeltaKernel(accelerator);
            RemainderScanKernel = owner.GetRemainderScanKernel(accelerator);
            CandidateMaskKernel = owner.GetCandidateMaskKernel(accelerator);
            Capacity = actualCapacity;
            CycleCapacity = cycleCapacity;
        }

        internal readonly MemoryBuffer1D<MontgomeryDivisorData, Stride1D.Dense> DivisorsBuffer;

        internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> ExponentBuffer;

        internal readonly MemoryBuffer1D<byte, Stride1D.Dense> HitsBuffer;

        internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> DivisorDeltaBuffer;

        internal readonly MemoryBuffer1D<byte, Stride1D.Dense> RemainderDeltaBuffer;

        internal readonly MemoryBuffer1D<byte, Stride1D.Dense> RemainderBuffer;

        internal readonly ulong[] Divisors;

        internal readonly ulong[] Exponents;

        internal readonly byte[] Hits;

        internal readonly MontgomeryDivisorData[] DivisorData;

        internal readonly ulong[] CycleCandidates;

        internal readonly ulong[] CycleLengths;

        internal readonly Action<Index1D, ArrayView<ulong>, ArrayView<byte>, byte> RemainderDeltaKernel;

        internal readonly Action<KernelConfig, Index1D, ArrayView<byte>, ArrayView<byte>, byte, byte> RemainderScanKernel;
        internal readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, byte, ArrayView<byte>> CandidateMaskKernel;

        internal readonly int Capacity;

        internal readonly int CycleCapacity;

        public void Dispose()
        {
            DivisorsBuffer.Dispose();
            ExponentBuffer.Dispose();
            HitsBuffer.Dispose();
            DivisorDeltaBuffer.Dispose();
            RemainderDeltaBuffer.Dispose();
            RemainderBuffer.Dispose();
            ArrayPool<ulong> ulongPool = ThreadStaticPools.UlongPool;
            ArrayPool<byte> bytePool = ThreadStaticPools.BytePool;
            ArrayPool<MontgomeryDivisorData> divisorPool = ThreadStaticPools.MontgomeryDivisorDataPool;
            ulongPool.Return(Divisors, clearArray: false);
            ulongPool.Return(Exponents, clearArray: false);
            bytePool.Return(Hits, clearArray: false);
            divisorPool.Return(DivisorData, clearArray: false);
            ulongPool.Return(CycleCandidates, clearArray: false);
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




