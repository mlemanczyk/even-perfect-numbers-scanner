using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

public sealed class DivisorCycleCache
{
    public sealed class CycleBlock : IDisposable
    {
        private readonly DivisorCycleCache _owner;
        private int _referenceCount;

        internal CycleBlock(DivisorCycleCache owner, int index, ulong start, in ulong[] cycles)
        {
            _owner = owner;
            Index = index;
            Start = start;
            End = start + (ulong)cycles.Length - 1UL;
            Cycles = cycles;
        }

        internal int Index { get; }

        internal ulong Start { get; }

        internal ulong End { get; }

        internal ulong[] Cycles { get; }

        internal int ReferenceCount => Volatile.Read(ref _referenceCount);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ulong GetCycle(ulong divisor) => Cycles[(int)(divisor - Start)];

        internal void Retain()
        {
            Interlocked.Increment(ref _referenceCount);
        }

        public void Dispose()
        {
            int remaining = Interlocked.Decrement(ref _referenceCount);
            if (remaining < 0)
            {
                throw new InvalidOperationException("Cycle block released more times than it was acquired.");
            }

            if (remaining == 0)
            {
                _owner.OnCycleBlockReleased(this);
            }
        }
    }

    private const int CycleGenerationBatchSize = 262_144; // TODO: Remove the block-sized batch once the single-cycle generator
                                                          // replaces block hydration so we stop materializing large buffers
                                                          // just to satisfy one missing divisor lookup.
    private const byte ByteZero = 0;
    private const byte ByteOne = 1;

    private readonly object _sync = new(); // TODO: Delete the synchronization object once the cache exposes only snapshot reads and single-cycle computation so callers no longer contend on locks.
    private readonly ConcurrentDictionary<int, Task<CycleBlock>> _pending = new(); // TODO: Drop the concurrent dictionary entirely when the block pipeline is replaced with single-cycle generation; the new plan never queues background work.
    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>>> _gpuKernelCache = new(AcceleratorReferenceComparer.Instance); // TODO: Swap this ConcurrentDictionary for a simple accelerator-indexed array once the single-cycle helper precompiles kernels during startup, eliminating the need for thread-safe lookups.
    private CycleBlock _baseBlock = null!;
    private CycleBlock? _activeBlock; // TODO: Remove dynamic blocks altogether once lookups route to the snapshot or compute an individual cycle without mutating shared state.
    private CycleBlock? _prefetchedBlock; // TODO: Delete prefetch support so the cache never materializes extra blocks and callers always compute the missing cycle inline.
    private bool _useGpuGeneration = true;
	private readonly int _divisorCyclesBatchSize;
	private static int _sharedDivisorCyclesBatchSize = GpuConstants.GpuCycleStepsPerInvocation;

	public static void SetDivisorCyclesBatchSize(int divisorCyclesBatchSize) => _sharedDivisorCyclesBatchSize = divisorCyclesBatchSize;
    public static DivisorCycleCache Shared { get; } = new DivisorCycleCache(_sharedDivisorCyclesBatchSize);

    private DivisorCycleCache(int divisorCyclesBatchSize)
    {
		_divisorCyclesBatchSize = divisorCyclesBatchSize;
        ulong[] snapshot = MersenneDivisorCycles.Shared.ExportSmallCyclesSnapshot();
        Initialize(snapshot);
	}

    // TODO: Remove the ability to reload at runtime once the cache becomes a read-only snapshot plus
    // transient single-cycle generator; future callers should load the snapshot during startup only.
    public void ReloadFromCurrentSnapshot()
    {
        ulong[] snapshot = MersenneDivisorCycles.Shared.ExportSmallCyclesSnapshot();
        Initialize(snapshot);
    }

    public void ConfigureGeneratorDevice(bool useGpu)
    {
        lock (_sync)
        {
            _useGpuGeneration = useGpu;
        }
        // TODO: Remove the lock once single-cycle generation eliminates shared mutable state; a simple
        // volatile write will be enough when the cache becomes read-only.
    }

    private void Initialize(ulong[] snapshot)
    {
        CycleBlock? activeToDispose = null;
        CycleBlock? prefetchedToDispose = null;

        lock (_sync)
        {
            if (_activeBlock is not null && _activeBlock.Index != 0)
            {
                activeToDispose = _activeBlock;
            }

            if (_prefetchedBlock is not null && _prefetchedBlock.Index != 0)
            {
                prefetchedToDispose = _prefetchedBlock;
            }

            _pending.Clear();
            // TODO: Remove the dictionary once prefetching and background block tracking disappear; with single-cycle generation we will keep only the snapshot and skip additional bookkeeping entirely.
            _baseBlock = new CycleBlock(this, index: 0, start: 0UL, snapshot);
            _activeBlock = null;
            _prefetchedBlock = null;
            StartPrefetchLocked(1);
        }

        activeToDispose?.Dispose();
        prefetchedToDispose?.Dispose();
    }

    public CycleBlock Acquire(ulong divisor)
    {
        CycleBlock block = divisor <= _baseBlock.End ? _baseBlock : AcquireDynamicBlock(divisor);
        // TODO: Replace this block-based acquisition with a direct GetCycle API that computes the missing cycle synchronously on the configured device so the cache never hands out mutable blocks or requires disposal.
        block.Retain();
        return block;
    }

    private CycleBlock AcquireDynamicBlock(ulong divisor)
    {
        int blockIndex = GetBlockIndex(divisor);

        while (true)
        {
            Task<CycleBlock>? pendingTask = null;
            lock (_sync)
            {
                if (TryGetCachedBlockLocked(blockIndex, out CycleBlock? cached))
                {
                    return cached!;
                }

                if (!_pending.TryGetValue(blockIndex, out pendingTask))
                {
                    // TODO: Replace Task.Run with an inline single-cycle computation that runs on the configured
                    // device so we materialize exactly one missing cycle and stop scheduling thread-pool work for
                    // entire blocks.
                    pendingTask = Task.Run(() => GenerateBlock(blockIndex));
                    _pending[blockIndex] = pendingTask;
                }
            }

            CycleBlock block = pendingTask.GetAwaiter().GetResult();
            lock (_sync)
            {
                _pending.TryRemove(blockIndex, out _);

                if (TryGetCachedBlockLocked(blockIndex, out CycleBlock? cachedAfterAwait))
                {
                    return cachedAfterAwait!;
                }

                SetActiveBlockLocked(block);
                // TODO: Stop promoting freshly generated blocks into the shared cache and, instead of
                // returning a block, surface the computed cycle directly to the caller so no extra cache
                // state survives beyond the single snapshot block.
                return block;
            }
        }
    }

    private bool TryGetCachedBlockLocked(int blockIndex, out CycleBlock? block)
    {
        CycleBlock? active = _activeBlock;
        if (active is not null && active.Index == blockIndex)
        {
            block = active;
            return true;
        }

        CycleBlock? prefetched = _prefetchedBlock;
        if (prefetched is not null && prefetched.Index == blockIndex)
        {
            PromotePrefetchedLocked();
            block = _activeBlock!;
            return true;
        }

        block = null;
        return false;
    }

    private void PromotePrefetchedLocked()
    {
        CycleBlock? prefetched = _prefetchedBlock;
        if (prefetched is null)
        {
            return;
        }

        CycleBlock? previousActive = _activeBlock;
        _activeBlock = prefetched;
        _prefetchedBlock = null;
        prefetched.Retain();
        prefetched.Dispose();

        if (previousActive is not null && previousActive.Index != 0 && previousActive.Index != prefetched.Index)
        {
            previousActive.Dispose();
        }

        StartPrefetchLocked(prefetched.Index + 1); // TODO: Remove this call once prefetching is dropped
                                                   // so we strictly keep a single active block in memory.
    }

    private void SetActiveBlockLocked(CycleBlock block)
    {
        CycleBlock? previousActive = _activeBlock;
        _activeBlock = block;
        block.Retain();

        CycleBlock? prefetched = _prefetchedBlock;
        if (prefetched is not null && prefetched.Index != block.Index + 1)
        {
            if (prefetched.Index != 0)
            {
                prefetched.Dispose();
            }

            _prefetchedBlock = null;
        }

        if (previousActive is not null && previousActive.Index != 0 && previousActive.Index != block.Index)
        {
            previousActive.Dispose();
        }

        StartPrefetchLocked(block.Index + 1); // TODO: Remove once prefetching is removed and the cache
                                              // no longer speculatively requests the next block after a miss.
    }

    private void StartPrefetchLocked(int blockIndex)
    {
        // TODO: Remove prefetching entirely so the cache maintains a single snapshot block and computes
        // individual cycles on demand instead of triggering background generation for future indices.
        if (blockIndex <= 0)
        {
            return;
        }

        CycleBlock? prefetched = _prefetchedBlock;
        if (prefetched is not null && prefetched.Index == blockIndex)
        {
            return;
        }

        if (_pending.TryGetValue(blockIndex, out Task<CycleBlock>? existing))
        {
            if (existing.IsCompletedSuccessfully)
            {
                _pending.TryRemove(blockIndex, out _);
                AssignPrefetchedBlockLocked(blockIndex, existing.Result);
            }

            return;
        }

        // TODO: Delete this once prefetching is removed; single-cycle generation will happen inline and
        // no asynchronous work should be scheduled.
        Task<CycleBlock> task = Task.Run(() => GenerateBlock(blockIndex));
        _pending[blockIndex] = task;
        task.ContinueWith(t => OnPrefetchCompleted(blockIndex, t), TaskScheduler.Default);
    }

    private void OnPrefetchCompleted(int blockIndex, Task<CycleBlock> task)
    {
        // TODO: Delete this callback when we remove prefetching. Single-cycle generation will run inline,
        // eliminating the need for asynchronous continuation handlers and concurrent dictionaries.
        if (!task.IsCompletedSuccessfully)
        {
            _pending.TryRemove(blockIndex, out _);
            return;
        }

        lock (_sync)
        {
            _pending.TryRemove(blockIndex, out _);
            AssignPrefetchedBlockLocked(blockIndex, task.Result);
        }
    }

    private void AssignPrefetchedBlockLocked(int blockIndex, CycleBlock block)
    {
        // TODO: Remove this entire method when prefetching goes away. The single-cycle flow should never
        // allocate or retain extra blocks beyond the startup snapshot, so prefetched block bookkeeping
        // becomes unnecessary.
        CycleBlock? active = _activeBlock;
        if (active is not null && blockIndex != active.Index + 1)
        {
            MontgomeryDivisorDataCache.ReleaseBlock(block);
            return;
        }

        CycleBlock? previousPrefetched = _prefetchedBlock;
        if (ReferenceEquals(previousPrefetched, block))
        {
            return;
        }

        if (_prefetchedBlock is null || _prefetchedBlock.Index <= blockIndex)
        {
            if (previousPrefetched is not null && previousPrefetched.Index != 0)
            {
                previousPrefetched.Dispose();
            }

            _prefetchedBlock = block;
            block.Retain();
        }
        else
        {
            MontgomeryDivisorDataCache.ReleaseBlock(block);
        }
    }

    private void OnCycleBlockReleased(CycleBlock block)
    {
        if (block.Index == 0)
        {
            return;
        }

        for (int attempt = 0; attempt < 8; attempt++)
        {
            if (block.ReferenceCount != 0)
            {
                return;
            }

            Thread.SpinWait(1 << attempt);
        }

        bool stillTracked;

        lock (_sync)
        {
            stillTracked = ReferenceEquals(_activeBlock, block) || ReferenceEquals(_prefetchedBlock, block);
        }

        if (!stillTracked && block.ReferenceCount == 0)
        {
            MontgomeryDivisorDataCache.ReleaseBlock(block);
        }
    }

    private CycleBlock GenerateBlock(int blockIndex)
    {
        if (blockIndex == 0)
        {
            return _baseBlock;
        }

        ulong start = GetBlockStart(blockIndex);
        int length = _baseBlock.Cycles.Length;
        ulong[] cycles = new ulong[length];

        if (_useGpuGeneration)
        {
            ComputeCyclesGpu(start, cycles, _divisorCyclesBatchSize);
        }
        else
        {
            ComputeCyclesCpu(start, cycles);
        }

        // TODO: Replace this block generator with a single-cycle helper that reuses shared buffers,
        // returns just the requested divisor's cycle, and avoids mutating cache state beyond the
        // preloaded snapshot.
        return new CycleBlock(this, blockIndex, start, cycles);
    }

    private int GetBlockIndex(ulong divisor)
    {
        if (divisor <= _baseBlock.End)
        {
            return 0;
        }

        ulong offset = divisor - (_baseBlock.End + 1UL);
        return (int)(offset / (ulong)PerfectNumberConstants.MaxQForDivisorCycles) + 1;
    }

    private ulong GetBlockStart(int blockIndex)
    {
        if (blockIndex == 0)
        {
            return 0UL;
        }

        ulong baseEnd = _baseBlock.End;
        return baseEnd + 1UL + (ulong)(blockIndex - 1) * (ulong)PerfectNumberConstants.MaxQForDivisorCycles;
    }

    private void ComputeCyclesCpu(ulong start, ulong[] destination)
    {
        ulong divisor = start;
        for (int i = 0; i < destination.Length; i++)
        {
            // TODO: Port this CPU fallback to the unrolled-hex cycle generator once it is shared so misses outside the snapshot
            // stop looping through CalculateCycleLength, which benchmarks showed is far slower than the GPU-assisted pipeline.
            destination[i] = MersenneDivisorCycles.CalculateCycleLength(divisor);
            divisor++;
        }
    }

    private void ComputeCyclesGpu(ulong start, ulong[] destination, int divisorCyclesBatchSize)
    {
        var gpuLease = GpuKernelPool.GetKernel(_useGpuGeneration);
        var execution = gpuLease.EnterExecutionScope();

        try
        {
            Accelerator accelerator = gpuLease.Accelerator;
            Action<Index1D, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>> kernel = _gpuKernelCache.GetOrAdd(accelerator, LoadKernel);
            // TODO: Once single-cycle generation is in place, bypass this block-sized kernel entirely and
            // call the ProcessEightBitWindows-based GPU helper that materializes only the requested
            // divisor, reusing a pinned upload buffer instead of allocating per block.

            int length = destination.Length;
            int chunkCapacity = Math.Min(length, CycleGenerationBatchSize);
            ulong[] divisors = ArrayPool<ulong>.Shared.Rent(chunkCapacity);
            ulong[] powScratch = ArrayPool<ulong>.Shared.Rent(chunkCapacity);
            ulong[] orderScratch = ArrayPool<ulong>.Shared.Rent(chunkCapacity);
            byte[] statusScratch = ArrayPool<byte>.Shared.Rent(chunkCapacity);

            try
            {
                using MemoryBuffer1D<ulong, Stride1D.Dense> divisorBuffer = accelerator.Allocate1D<ulong>(chunkCapacity);
                using MemoryBuffer1D<ulong, Stride1D.Dense> powBuffer = accelerator.Allocate1D<ulong>(chunkCapacity);
                using MemoryBuffer1D<ulong, Stride1D.Dense> orderBuffer = accelerator.Allocate1D<ulong>(chunkCapacity);
                using MemoryBuffer1D<ulong, Stride1D.Dense> resultBuffer = accelerator.Allocate1D<ulong>(chunkCapacity);
                using MemoryBuffer1D<byte, Stride1D.Dense> statusBuffer = accelerator.Allocate1D<byte>(chunkCapacity);

                int offset = 0;
                while (offset < length)
                {
                    int batchSize = Math.Min(chunkCapacity, length - offset);
                    Span<ulong> divisorsSpan = divisors.AsSpan(0, batchSize);
                    Span<ulong> destinationSpan = destination.AsSpan(offset, batchSize);
                    Span<ulong> powSpan = powScratch.AsSpan(0, batchSize);
                    Span<ulong> orderSpan = orderScratch.AsSpan(0, batchSize);
                    Span<byte> statusSpan = statusScratch.AsSpan(0, batchSize);

                    int pending = 0;
                    for (int i = 0; i < batchSize; i++)
                    {
                        ulong divisorValue = start + (ulong)(offset + i);
                        divisorsSpan[i] = divisorValue;

                        if ((divisorValue & (divisorValue - 1UL)) == 0UL)
                        {
                            destinationSpan[i] = 1UL;
                            powSpan[i] = 0UL;
                            orderSpan[i] = 1UL;
                            statusSpan[i] = ByteOne;
                        }
                        else
                        {
                            ulong initialPow = 2UL;
                            if (initialPow >= divisorValue)
                            {
                                initialPow -= divisorValue;
                            }

                            destinationSpan[i] = 0UL;
                            powSpan[i] = initialPow;
                            orderSpan[i] = 1UL;
                            statusSpan[i] = ByteZero;
                            pending++;
                        }
                    }

                    ArrayView1D<ulong, Stride1D.Dense> divisorView = divisorBuffer.View.SubView(0, batchSize);
                    ArrayView1D<ulong, Stride1D.Dense> powView = powBuffer.View.SubView(0, batchSize);
                    ArrayView1D<ulong, Stride1D.Dense> orderView = orderBuffer.View.SubView(0, batchSize);
                    ArrayView1D<ulong, Stride1D.Dense> resultView = resultBuffer.View.SubView(0, batchSize);
                    ArrayView1D<byte, Stride1D.Dense> statusView = statusBuffer.View.SubView(0, batchSize);

                    divisorView.CopyFromCPU(ref MemoryMarshal.GetReference(divisorsSpan), batchSize);
                    powView.CopyFromCPU(ref MemoryMarshal.GetReference(powSpan), batchSize);
                    orderView.CopyFromCPU(ref MemoryMarshal.GetReference(orderSpan), batchSize);
                    resultView.CopyFromCPU(ref MemoryMarshal.GetReference(destinationSpan), batchSize);
                    statusView.CopyFromCPU(ref MemoryMarshal.GetReference(statusSpan), batchSize);

                    while (pending > 0)
                    {
                        kernel(batchSize, divisorCyclesBatchSize, divisorView, powView, orderView, resultView, statusView);
                        accelerator.Synchronize();

                        statusView.CopyToCPU(ref MemoryMarshal.GetReference(statusSpan), batchSize);

                        pending = 0;
                        for (int i = 0; i < batchSize; i++)
                        {
                            if (statusSpan[i] == ByteZero)
                            {
                                pending++;
                            }
                        }
                    }

                    resultView.CopyToCPU(ref MemoryMarshal.GetReference(destinationSpan), batchSize);

                    offset += batchSize;
                }
            }
            finally
            {
                ArrayPool<ulong>.Shared.Return(divisors, clearArray: false);
                ArrayPool<ulong>.Shared.Return(powScratch, clearArray: false);
                ArrayPool<ulong>.Shared.Return(orderScratch, clearArray: false);
                ArrayPool<byte>.Shared.Return(statusScratch, clearArray: false);
            }
        }
        finally
        {
            execution.Dispose();
            gpuLease.Dispose();
        }
    }

    private static Action<Index1D, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>> LoadKernel(Accelerator accelerator)
    {
        return accelerator.LoadAutoGroupedStreamKernel<Index1D, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>>(GpuAdvanceDivisorCyclesKernel);
    }

    private static void GpuAdvanceDivisorCyclesKernel(
        Index1D index,
		int steps,
        ArrayView1D<ulong, Stride1D.Dense> divisors,
        ArrayView1D<ulong, Stride1D.Dense> pow,
        ArrayView1D<ulong, Stride1D.Dense> order,
        ArrayView1D<ulong, Stride1D.Dense> cycles,
        ArrayView1D<byte, Stride1D.Dense> status)
    {
        if (status[index] != ByteZero)
        {
            return;
        }

        ulong divisor = divisors[index];
        ulong currentPow = pow[index];
        ulong currentOrder = order[index];

        do
        {
            currentPow += currentPow;
            if (currentPow >= divisor)
            {
                currentPow -= divisor;
            }

            currentOrder++;

            if (currentPow == 1UL)
            {
                cycles[index] = currentOrder;
                status[index] = ByteOne;
                pow[index] = currentPow;
                order[index] = currentOrder;
                return;
            }
        }
        while (--steps != 0);

        pow[index] = currentPow;
        order[index] = currentOrder;
    }

    private sealed class AcceleratorReferenceComparer : IEqualityComparer<Accelerator>
    {
        internal static AcceleratorReferenceComparer Instance { get; } = new();

        public bool Equals(Accelerator? x, Accelerator? y) => ReferenceEquals(x, y);

        public int GetHashCode(Accelerator obj) => RuntimeHelpers.GetHashCode(obj);
    }
}

