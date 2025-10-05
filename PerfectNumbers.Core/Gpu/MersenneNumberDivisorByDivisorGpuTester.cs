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

/// <summary>
/// Scans Mersenne divisors on the GPU for prime exponents p >= 31 using per-divisor Montgomery data.
/// </summary>
public sealed class MersenneNumberDivisorByDivisorGpuTester : IMersenneNumberDivisorByDivisorTester
{
    private int _gpuBatchSize = GpuConstants.ScanBatchSize;
    private readonly object _sync = new();
    private ulong _divisorLimit;
    private bool _isConfigured;

    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<MontgomeryDivisorData>, ArrayView<ulong>, ArrayView<byte>>> _kernelCache = new();
    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>>> _kernelExponentCache = new();
    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView<ulong>, ArrayView<byte>, byte>> _remainderDeltaKernelCache = new(AcceleratorReferenceComparer.Instance);
    private readonly ConcurrentDictionary<Accelerator, Action<KernelConfig, Index1D, ArrayView<byte>, ArrayView<byte>, byte, byte>> _remainderScanKernelCache = new(AcceleratorReferenceComparer.Instance);
    private readonly ConcurrentDictionary<Accelerator, ConcurrentBag<BatchResources>> _resourcePools = new(AcceleratorReferenceComparer.Instance);
    private readonly ConcurrentBag<DivisorScanSession> _sessionPool = new();

    private Action<Index1D, ArrayView<MontgomeryDivisorData>, ArrayView<ulong>, ArrayView<byte>> GetKernel(Accelerator accelerator) =>
        _kernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<MontgomeryDivisorData>, ArrayView<ulong>, ArrayView<byte>>(CheckKernel));

    private Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>> GetKernelByPrimeExponent(Accelerator accelerator) =>
        _kernelExponentCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>>(ComputeMontgomeryExponentKernel));

    private Action<Index1D, ArrayView<ulong>, ArrayView<byte>, byte> GetRemainderDeltaKernel(Accelerator accelerator) =>
        _remainderDeltaKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<byte>, byte>(ComputeRemainderDeltasKernel));

    private Action<KernelConfig, Index1D, ArrayView<byte>, ArrayView<byte>, byte, byte> GetRemainderScanKernel(Accelerator accelerator) =>
        _remainderScanKernelCache.GetOrAdd(
            accelerator,
            acc => acc.LoadStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, byte, byte>(AccumulateRemaindersKernel));

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
        lock (_sync)
        {
            _divisorLimit = ComputeDivisorLimitFromMaxPrime(maxPrime);
            _isConfigured = true;
        }
    }

    public bool IsPrime(ulong prime, out bool divisorsExhausted)
    {
        ulong allowedMax;
        int batchCapacity;

        lock (_sync)
        {
            if (!_isConfigured)
            {
                throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
            }

            allowedMax = ComputeAllowedMaxDivisor(prime, _divisorLimit);
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
        in ulong[] divisors,
        ulong[] exponents,
        byte[] hits,
        MontgomeryDivisorData[] divisorData,
        out ulong lastProcessed,
        out bool coveredRange,
        out ulong processedCount)
    {
        int batchCapacity = (int)divisorsBuffer.Length;
        bool composite = false;
        bool processedAll = false;
        processedCount = 0UL;
        lastProcessed = 0UL;

        Span<MontgomeryDivisorData> divisorDataSpan;
        Span<ulong> divisorSpan;
        Span<ulong> exponentSpan;
        Span<byte> hitsSpan;
        ArrayView1D<MontgomeryDivisorData, Stride1D.Dense> divisorView;
        ArrayView1D<ulong, Stride1D.Dense> exponentView;
        ArrayView1D<byte, Stride1D.Dense> hitsView;

        int chunkCapacity = Math.Min(batchCapacity, (int)accelerator.MaxNumThreadsPerGroup);
        if (chunkCapacity <= 0)
        {
            chunkCapacity = batchCapacity;
        }

        ulong[] candidateDivisors = ArrayPool<ulong>.Shared.Rent(chunkCapacity);
        ulong[] filteredDivisors = ArrayPool<ulong>.Shared.Rent(chunkCapacity);
        ulong[] divisorGaps = ArrayPool<ulong>.Shared.Rent(chunkCapacity);
        byte[] remainder10Buffer = ArrayPool<byte>.Shared.Rent(chunkCapacity);
        byte[] remainder8Buffer = ArrayPool<byte>.Shared.Rent(chunkCapacity);
        byte[] remainder5Buffer = ArrayPool<byte>.Shared.Rent(chunkCapacity);
        byte[] remainder3Buffer = ArrayPool<byte>.Shared.Rent(chunkCapacity);

        UInt128 twoP128 = (UInt128)prime << 1;
        if (twoP128 == UInt128.Zero)
        {
            coveredRange = true;
            return false;
        }

        UInt128 allowedMax128 = allowedMax;
        UInt128 firstDivisor128 = twoP128 + UInt128.One;
        if (firstDivisor128 > allowedMax128)
        {
            coveredRange = true;
            return false;
        }

        UInt128 maxK128 = (allowedMax128 - UInt128.One) / twoP128;
        if (maxK128 == UInt128.Zero)
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

        try
        {
            while (currentK <= maxK)
            {
                int chunkCount = Math.Min(chunkCapacity, batchCapacity);
                ulong remainingK = maxK - currentK + 1UL;
                if ((ulong)chunkCount > remainingK)
                {
                    chunkCount = (int)remainingK;
                }

                if (chunkCount <= 0)
                {
                    break;
                }

                Span<ulong> candidateSpan = candidateDivisors.AsSpan(0, chunkCount);
                Span<ulong> gapSpan = divisorGaps.AsSpan(0, chunkCount);
                Span<byte> rem10Span = remainder10Buffer.AsSpan(0, chunkCount);
                Span<byte> rem8Span = remainder8Buffer.AsSpan(0, chunkCount);
                Span<byte> rem5Span = remainder5Buffer.AsSpan(0, chunkCount);
                Span<byte> rem3Span = remainder3Buffer.AsSpan(0, chunkCount);

                UInt128 localDivisor = currentDivisor128;

                for (int i = 0; i < chunkCount; i++)
                {
                    ulong divisorValue = (ulong)localDivisor;
                    candidateSpan[i] = divisorValue;

                    processedCount++;
                    lastProcessed = divisorValue;

                    localDivisor += twoP128;
                }

                gapSpan[0] = 0UL;
                for (int i = 1; i < chunkCount; i++)
                {
                    gapSpan[i] = candidateSpan[i] - candidateSpan[i - 1];
                }

                currentDivisor128 = localDivisor;

                ArrayView1D<ulong, Stride1D.Dense> gapView = divisorDeltaBuffer.View.SubView(0, chunkCount);
                ArrayView1D<byte, Stride1D.Dense> deltaView = remainderDeltaBuffer.View.SubView(0, chunkCount);
                ArrayView1D<byte, Stride1D.Dense> remainderView = remainderBuffer.View.SubView(0, chunkCount);

                gapView.CopyFromCPU(ref MemoryMarshal.GetReference(gapSpan), chunkCount);

                ComputeRemaindersOnGpu(accelerator, chunkCount, remainder10, 10, gapView, deltaView, remainderView, remainderDeltaKernel, remainderScanKernel, rem10Span);
                ComputeRemaindersOnGpu(accelerator, chunkCount, remainder8, 8, gapView, deltaView, remainderView, remainderDeltaKernel, remainderScanKernel, rem8Span);
                ComputeRemaindersOnGpu(accelerator, chunkCount, remainder5, 5, gapView, deltaView, remainderView, remainderDeltaKernel, remainderScanKernel, rem5Span);
                ComputeRemaindersOnGpu(accelerator, chunkCount, remainder3, 3, gapView, deltaView, remainderView, remainderDeltaKernel, remainderScanKernel, rem3Span);

                remainder10 = AddMod(rem10Span[chunkCount - 1], step10, 10);
                remainder8 = AddMod(rem8Span[chunkCount - 1], step8, 8);
                remainder5 = AddMod(rem5Span[chunkCount - 1], step5, 5);
                remainder3 = AddMod(rem3Span[chunkCount - 1], step3, 3);

                int filteredCount = 0;
                Span<ulong> filteredDivisorsSpan = filteredDivisors.AsSpan(0, chunkCount);

                for (int i = 0; i < chunkCount; i++)
                {
                    byte r10 = rem10Span[i];
                    if (lastIsSeven)
                    {
                        if (r10 != 3 && r10 != 7 && r10 != 9)
                        {
                            continue;
                        }
                    }
                    else
                    {
                        if (r10 != 1 && r10 != 3 && r10 != 7 && r10 != 9)
                        {
                            continue;
                        }
                    }

                    byte r8 = rem8Span[i];
                    if (r8 != 1 && r8 != 7)
                    {
                        continue;
                    }

                    if (rem3Span[i] == 0 || rem5Span[i] == 0)
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

                int batchSize = 0;
                for (int i = 0; i < filteredCount; i++)
                {
                    ulong divisorValue = filteredDivisorsSpan[i];
                    divisors[batchSize] = divisorValue;
                    exponents[batchSize] = prime;
                    batchSize++;
                }

                if (batchSize == 0)
                {
                    processedAll = currentK > maxK;
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

                processedAll = currentK > maxK;
            }
        }
        finally
        {
            ArrayPool<ulong>.Shared.Return(candidateDivisors, clearArray: false);
            ArrayPool<ulong>.Shared.Return(filteredDivisors, clearArray: false);
            ArrayPool<ulong>.Shared.Return(divisorGaps, clearArray: false);
            ArrayPool<byte>.Shared.Return(remainder10Buffer, clearArray: false);
            ArrayPool<byte>.Shared.Return(remainder8Buffer, clearArray: false);
            ArrayPool<byte>.Shared.Return(remainder5Buffer, clearArray: false);
            ArrayPool<byte>.Shared.Return(remainder3Buffer, clearArray: false);
        }

        coveredRange = composite || processedAll || (currentDivisor128 > allowedMax128);
        return composite;
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ComputeRemaindersOnGpu(
        Accelerator accelerator,
        int count,
        byte baseRemainder,
        byte modulus,
        ArrayView1D<ulong, Stride1D.Dense> gapView,
        ArrayView1D<byte, Stride1D.Dense> deltaView,
        ArrayView1D<byte, Stride1D.Dense> remainderView,
        Action<Index1D, ArrayView<ulong>, ArrayView<byte>, byte> deltaKernel,
        Action<KernelConfig, Index1D, ArrayView<byte>, ArrayView<byte>, byte, byte> scanKernel,
        Span<byte> destination)
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
        accelerator.Synchronize();
        remainderView.CopyToCPU(ref MemoryMarshal.GetReference(destination), count);
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
                result %= modulus;
            }
        }

        return (byte)result;
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
        hits[index] = exponent.Pow2MontgomeryModWindowed(divisor, keepMontgomery: false) == 1UL ? (byte)1 : (byte)0;
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

        public void CheckDivisor(ulong divisor, ulong divisorCycle, ReadOnlySpan<ulong> primes, Span<byte> hits)
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

            MontgomeryDivisorData divisorData = MontgomeryDivisorDataCache.Get(divisor);
            if (divisorData.Modulus <= 1UL || (divisorData.Modulus & 1UL) == 0UL)
            {
                hits.Clear();
                return;
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
    private static void ComputeRemainderDeltasKernel(Index1D index, ArrayView<ulong> gaps, ArrayView<byte> deltas, byte modulus)
    {
        deltas[index] = (byte)(gaps[index] % modulus);
    }

    private static void AccumulateRemaindersKernel(
        Index1D index,
        ArrayView<byte> deltas,
        ArrayView<byte> remainders,
        byte baseRemainder,
        byte modulus)
    {
        int globalIndex = index;
        int length = (int)remainders.Length;
        if (globalIndex >= length)
        {
            return;
        }

        int localIndex = Group.IdxX;
        int groupSize = Group.Dimension.X;
        var shared = SharedMemory.GetDynamic<int>();

        shared[localIndex] = deltas[globalIndex];
        Group.Barrier();

        int offset = 1;
        int mod = modulus;
        while (offset < groupSize)
        {
            int addend = 0;
            if (localIndex >= offset)
            {
                addend = shared[localIndex - offset];
            }

            Group.Barrier();

            if (localIndex >= offset)
            {
                int sum = shared[localIndex] + addend;
                if (sum >= mod)
                {
                    sum %= mod;
                }

                shared[localIndex] = sum;
            }

            Group.Barrier();

            offset <<= 1;
        }

        if (globalIndex == 0)
        {
            remainders[0] = baseRemainder;
            return;
        }

        int remainder = baseRemainder + shared[localIndex];
        if (remainder >= mod)
        {
            remainder %= mod;
        }

        remainders[globalIndex] = (byte)remainder;
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

        results[index] = exponent.Pow2MontgomeryModWindowed(divisor, keepMontgomery: true);
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

        return new BatchResources(this, accelerator, capacity);
    }

    private void ReturnBatchResources(Accelerator accelerator, BatchResources resources) => _resourcePools.GetOrAdd(accelerator, static _ => []).Add(resources);

    private sealed class BatchResources : IDisposable
    {
        internal BatchResources(MersenneNumberDivisorByDivisorGpuTester owner, Accelerator accelerator, int capacity)
        {
            DivisorsBuffer = accelerator.Allocate1D<MontgomeryDivisorData>(capacity);
            ExponentBuffer = accelerator.Allocate1D<ulong>(capacity);
            HitsBuffer = accelerator.Allocate1D<byte>(capacity);
            DivisorDeltaBuffer = accelerator.Allocate1D<ulong>(capacity);
            RemainderDeltaBuffer = accelerator.Allocate1D<byte>(capacity);
            RemainderBuffer = accelerator.Allocate1D<byte>(capacity);
            Divisors = ArrayPool<ulong>.Shared.Rent(capacity);
            Exponents = ArrayPool<ulong>.Shared.Rent(capacity);
            Hits = ArrayPool<byte>.Shared.Rent(capacity);
            DivisorData = ArrayPool<MontgomeryDivisorData>.Shared.Rent(capacity);
            RemainderDeltaKernel = owner.GetRemainderDeltaKernel(accelerator);
            RemainderScanKernel = owner.GetRemainderScanKernel(accelerator);
            Capacity = capacity;
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

        internal readonly Action<Index1D, ArrayView<ulong>, ArrayView<byte>, byte> RemainderDeltaKernel;

        internal readonly Action<KernelConfig, Index1D, ArrayView<byte>, ArrayView<byte>, byte, byte> RemainderScanKernel;

        internal readonly int Capacity;

        public void Dispose()
        {
            DivisorsBuffer.Dispose();
            ExponentBuffer.Dispose();
            HitsBuffer.Dispose();
            DivisorDeltaBuffer.Dispose();
            RemainderDeltaBuffer.Dispose();
            RemainderBuffer.Dispose();
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




