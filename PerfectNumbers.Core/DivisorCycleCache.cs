using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

public sealed class DivisorCycleCache
{
    private const byte ByteZero = 0;
    private const byte ByteOne = 1;
    private const int StackBufferThreshold = 128;

    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>>> _gpuKernelCache = new(AcceleratorReferenceComparer.Instance);
    private volatile ulong[] _snapshot;
    private volatile bool _useGpuGeneration = true;
    private readonly int _divisorCyclesBatchSize;
    private static int _sharedDivisorCyclesBatchSize = GpuConstants.GpuCycleStepsPerInvocation;

    private const int CycleCacheTrackingLimit = 500_000;
    private static readonly HashSet<ulong> CycleCacheTrackedDivisors = new();
    private static readonly object CycleCacheTrackingLock = new();
    private static ulong CycleCacheHitCount;
    private static bool CycleCacheTrackingDisabled;

    public static void SetDivisorCyclesBatchSize(int divisorCyclesBatchSize)
    {
        _sharedDivisorCyclesBatchSize = Math.Max(1, divisorCyclesBatchSize);
    }

    public static DivisorCycleCache Shared { get; } = new DivisorCycleCache(_sharedDivisorCyclesBatchSize);

    public int PreferredBatchSize => _divisorCyclesBatchSize;

    private DivisorCycleCache(int divisorCyclesBatchSize)
    {
        _divisorCyclesBatchSize = Math.Max(1, divisorCyclesBatchSize);
        _snapshot = MersenneDivisorCycles.Shared.ExportSmallCyclesSnapshot();
    }

    public void ConfigureGeneratorDevice(bool useGpu)
    {
        _useGpuGeneration = useGpu;
    }

    public void RefreshSnapshot()
    {
        _snapshot = MersenneDivisorCycles.Shared.ExportSmallCyclesSnapshot();
    }

    public ulong GetCycleLength(ulong divisor, bool skipPrimeOrderHeuristic = false)
    {
        if (!skipPrimeOrderHeuristic)
        {
            Span<ulong> result = stackalloc ulong[1];
            ReadOnlySpan<ulong> singleDivisor = stackalloc ulong[1] { divisor };
            GetCycleLengths(singleDivisor, result);
            return result[0];
        }

        // The skipPrimeOrderHeuristic path is only used for divisors produced by the CPU by-divisor scanner,
        // which always generates values greater than one. Keep this guard disabled here to avoid repeating the
        // validation already performed by the general GetCycleLengths path.
        // if (divisor <= 1UL)
        // {
        //     throw new InvalidDataException("Divisor must be > 1");
        // }

        ulong[] snapshot = _snapshot;
        if (divisor < (ulong)snapshot.Length)
        {
            ulong cached = snapshot[divisor];
            if (cached == 0UL)
            {
                throw new InvalidDataException($"Divisor cycle is missing for {divisor}");
            }

            ObserveCycleCacheDivisor(divisor);
            return cached;
        }

        MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(divisor);
        return MersenneDivisorCycles.CalculateCycleLength(divisor, divisorData, skipPrimeOrderHeuristic: true);
    }

    public void GetCycleLengths(ReadOnlySpan<ulong> divisors, Span<ulong> cycles)
    {
        ulong[] snapshot = _snapshot;
        int length = divisors.Length;
        int[]? rentedMissing = null;
        Span<int> missingBuffer = length <= StackBufferThreshold
            ? stackalloc int[length]
            : new Span<int>(rentedMissing = ArrayPool<int>.Shared.Rent(length), 0, length);

        int missingCount = 0;

        for (int i = 0; i < length; i++)
        {
            ulong divisor = divisors[i];
            if (divisor <= 1UL)
            {
                throw new InvalidDataException("Divisor must be > 1");
            }

            if (divisor < (ulong)snapshot.Length)
            {
                ulong cached = snapshot[divisor];
                if (cached == 0UL)
                {
                    throw new InvalidDataException($"Divisor cycle is missing for {divisor}");
                }

                ObserveCycleCacheDivisor(divisor);
                cycles[i] = cached;
            }
            else
            {
                missingBuffer[missingCount++] = i;
            }
        }

        if (missingCount != 0)
        {
            ReadOnlySpan<int> missingIndices = missingBuffer[..missingCount];
            if (_useGpuGeneration)
            {
                ComputeCyclesGpu(divisors, cycles, missingIndices);
            }
            else
            {
                ComputeCyclesCpu(divisors, cycles, missingIndices);
            }
        }

        if (rentedMissing is not null)
        {
            ArrayPool<int>.Shared.Return(rentedMissing, clearArray: false);
        }
    }

    private static void ObserveCycleCacheDivisor(ulong divisor)
    {
        bool logHit = false;
        ulong hitNumber = 0UL;
        bool trackingDisabledNow = false;

        lock (CycleCacheTrackingLock)
        {
            if (CycleCacheTrackingDisabled)
            {
                return;
            }

            if (!CycleCacheTrackedDivisors.Add(divisor))
            {
                CycleCacheHitCount++;
                hitNumber = CycleCacheHitCount;
                logHit = true;
            }
            else if (CycleCacheTrackedDivisors.Count >= CycleCacheTrackingLimit)
            {
                CycleCacheTrackingDisabled = true;
                CycleCacheTrackedDivisors.Clear();
                trackingDisabledNow = true;
            }
        }

        if (logHit)
        {
            Console.WriteLine($"Cycle cache hit for divisor {divisor} ({hitNumber})");
        }
        else if (trackingDisabledNow)
        {
            Console.WriteLine("Cycle cache hit tracking disabled after exceeding the tracking limit.");
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ComputeCyclesCpu(ReadOnlySpan<ulong> divisors, Span<ulong> cycles, ReadOnlySpan<int> indices)
    {
        for (int i = 0; i < indices.Length; i++)
        {
            int targetIndex = indices[i];
            ulong divisor = divisors[targetIndex];
            MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(divisor);
            cycles[targetIndex] = MersenneDivisorCycles.CalculateCycleLength(divisor, divisorData);
        }
    }

    private void ComputeCyclesGpu(ReadOnlySpan<ulong> divisors, Span<ulong> cycles, ReadOnlySpan<int> indices)
    {
        int count = indices.Length;
        ulong[]? rentedDivisors = null;
        Span<ulong> workDivisors = count <= StackBufferThreshold
            ? stackalloc ulong[count]
            : new Span<ulong>(rentedDivisors = ArrayPool<ulong>.Shared.Rent(count), 0, count);

        ulong[]? rentedResults = null;
        Span<ulong> workResults = count <= StackBufferThreshold
            ? stackalloc ulong[count]
            : new Span<ulong>(rentedResults = ArrayPool<ulong>.Shared.Rent(count), 0, count);

        for (int i = 0; i < count; i++)
        {
            workDivisors[i] = divisors[indices[i]];
        }

        ComputeCyclesGpuCore(workDivisors, workResults);

        for (int i = 0; i < count; i++)
        {
            cycles[indices[i]] = workResults[i];
        }

        if (rentedDivisors is not null)
        {
            ArrayPool<ulong>.Shared.Return(rentedDivisors, clearArray: false);
            ArrayPool<ulong>.Shared.Return(rentedResults!, clearArray: false);
        }
    }

    private void ComputeCyclesGpuCore(ReadOnlySpan<ulong> divisors, Span<ulong> destination)
    {
        int length = divisors.Length;
        var gpuLease = GpuKernelPool.GetKernel(useGpuOrder: true);
        var execution = gpuLease.EnterExecutionScope();

        Accelerator accelerator = gpuLease.Accelerator;
        var kernel = _gpuKernelCache.GetOrAdd(accelerator, LoadKernel);

        using MemoryBuffer1D<ulong, Stride1D.Dense> divisorBuffer = accelerator.Allocate1D<ulong>(length);
        using MemoryBuffer1D<ulong, Stride1D.Dense> powBuffer = accelerator.Allocate1D<ulong>(length);
        using MemoryBuffer1D<ulong, Stride1D.Dense> orderBuffer = accelerator.Allocate1D<ulong>(length);
        using MemoryBuffer1D<ulong, Stride1D.Dense> resultBuffer = accelerator.Allocate1D<ulong>(length);
        using MemoryBuffer1D<byte, Stride1D.Dense> statusBuffer = accelerator.Allocate1D<byte>(length);

        ulong[]? rentedPow = null;
        Span<ulong> powSpan = length <= StackBufferThreshold
            ? stackalloc ulong[length]
            : new Span<ulong>(rentedPow = ArrayPool<ulong>.Shared.Rent(length), 0, length);

        ulong[]? rentedOrder = null;
        Span<ulong> orderSpan = length <= StackBufferThreshold
            ? stackalloc ulong[length]
            : new Span<ulong>(rentedOrder = ArrayPool<ulong>.Shared.Rent(length), 0, length);

        ulong[]? rentedResult = null;
        Span<ulong> resultSpan = length <= StackBufferThreshold
            ? stackalloc ulong[length]
            : new Span<ulong>(rentedResult = ArrayPool<ulong>.Shared.Rent(length), 0, length);

        byte[]? rentedStatus = null;
        Span<byte> statusSpan = length <= StackBufferThreshold
            ? stackalloc byte[length]
            : new Span<byte>(rentedStatus = ArrayPool<byte>.Shared.Rent(length), 0, length);

        ref ulong divisorRef = ref MemoryMarshal.GetReference(divisors);
        divisorBuffer.View.CopyFromCPU(ref divisorRef, length);

        for (int i = 0; i < length; i++)
        {
            ulong divisor = divisors[i];
            ulong initialPow = 2UL;
            if (initialPow >= divisor)
            {
                initialPow -= divisor;
            }

            powSpan[i] = initialPow;
            orderSpan[i] = 1UL;
            resultSpan[i] = 0UL;
            statusSpan[i] = ByteZero;
        }

        powBuffer.View.CopyFromCPU(ref MemoryMarshal.GetReference(powSpan), length);
        orderBuffer.View.CopyFromCPU(ref MemoryMarshal.GetReference(orderSpan), length);
        resultBuffer.View.CopyFromCPU(ref MemoryMarshal.GetReference(resultSpan), length);
        statusBuffer.View.CopyFromCPU(ref MemoryMarshal.GetReference(statusSpan), length);

        int pending;
        do
        {
            kernel(length, _divisorCyclesBatchSize, divisorBuffer.View, powBuffer.View, orderBuffer.View, resultBuffer.View, statusBuffer.View);
            accelerator.Synchronize();

            statusBuffer.View.CopyToCPU(ref MemoryMarshal.GetReference(statusSpan), length);
            pending = 0;
            for (int i = 0; i < length; i++)
            {
                if (statusSpan[i] == ByteZero)
                {
                    pending++;
                }
            }
        }
        while (pending > 0);

        resultBuffer.View.CopyToCPU(ref MemoryMarshal.GetReference(resultSpan), length);
        resultSpan.CopyTo(destination);
        if (rentedPow is not null)
        {
            ArrayPool<ulong>.Shared.Return(rentedPow, clearArray: false);
            ArrayPool<ulong>.Shared.Return(rentedOrder!, clearArray: false);
            ArrayPool<ulong>.Shared.Return(rentedResult!, clearArray: false);
            ArrayPool<byte>.Shared.Return(rentedStatus!, clearArray: false);
        }

        execution.Dispose();
        gpuLease.Dispose();
    }

    private static Action<Index1D, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>> LoadKernel(Accelerator accelerator)
    {
        return accelerator.LoadAutoGroupedStreamKernel<Index1D, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>>(DivisorCycleKernels.GpuAdvanceDivisorCyclesKernel);
    }

    private sealed class AcceleratorReferenceComparer : IEqualityComparer<Accelerator>
    {
        internal static AcceleratorReferenceComparer Instance { get; } = new();

        public bool Equals(Accelerator? x, Accelerator? y) => ReferenceEquals(x, y);

        public int GetHashCode(Accelerator obj) => RuntimeHelpers.GetHashCode(obj);
    }
}




