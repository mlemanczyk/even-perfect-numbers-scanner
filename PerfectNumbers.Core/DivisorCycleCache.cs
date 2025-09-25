using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

public sealed class DivisorCycleCache
{
    internal sealed class CycleBlock
    {
        internal CycleBlock(int index, ulong start, ulong[] cycles)
        {
            Index = index;
            Start = start;
            Cycles = cycles;
            End = start + (ulong)cycles.Length - 1UL;
        }

        internal int Index { get; }

        internal ulong Start { get; }

        internal ulong End { get; }

        internal ulong[] Cycles { get; }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal ulong GetCycle(ulong divisor)
        {
            if (divisor < Start || divisor > End)
            {
                return 0UL;
            }

            return Cycles[(int)(divisor - Start)];
        }
    }

    public struct Lease : IDisposable
    {
        private CycleBlock? _block;

        internal Lease(CycleBlock block)
        {
            _block = block;
        }

        public readonly bool IsValid => _block is not null;

        public readonly ulong Start => _block?.Start ?? 0UL;

        public readonly ulong End => _block?.End ?? 0UL;

        public readonly ulong[]? Values => _block?.Cycles;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly ulong GetCycle(ulong divisor) => _block?.GetCycle(divisor) ?? 0UL;

        public void Dispose()
        {
            _block = null;
        }
    }

    private readonly object _sync = new();
    private readonly ConcurrentDictionary<int, Task<CycleBlock>> _pending = new();
    private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>> _gpuKernelCache = new(AcceleratorReferenceComparer.Instance);
    private CycleBlock _baseBlock = null!;
    private CycleBlock? _activeBlock;
    private CycleBlock? _prefetchedBlock;
    private bool _useGpuGeneration = true;

    public static DivisorCycleCache Shared { get; } = new DivisorCycleCache();

    private DivisorCycleCache()
    {
        ulong[] snapshot = MersenneDivisorCycles.Shared.ExportSmallCyclesSnapshot();
        Initialize(snapshot);
    }

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
    }

    private void Initialize(ulong[] snapshot)
    {
        lock (_sync)
        {
            _pending.Clear();
            _baseBlock = new CycleBlock(index: 0, start: 0UL, snapshot);
            _activeBlock = null;
            _prefetchedBlock = null;
            StartPrefetchLocked(1);
        }
    }

    public Lease Acquire(ulong divisor)
    {
        if (divisor <= _baseBlock.End)
        {
            return new Lease(_baseBlock);
        }

        CycleBlock block = AcquireDynamicBlock(divisor);
        return new Lease(block);
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

        _activeBlock = prefetched;
        _prefetchedBlock = null;
        StartPrefetchLocked(prefetched.Index + 1);
    }

    private void SetActiveBlockLocked(CycleBlock block)
    {
        _activeBlock = block;

        CycleBlock? prefetched = _prefetchedBlock;
        if (prefetched is not null && prefetched.Index != block.Index + 1)
        {
            _prefetchedBlock = null;
        }

        StartPrefetchLocked(block.Index + 1);
    }

    private void StartPrefetchLocked(int blockIndex)
    {
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

        Task<CycleBlock> task = Task.Run(() => GenerateBlock(blockIndex));
        _pending[blockIndex] = task;
        task.ContinueWith(t => OnPrefetchCompleted(blockIndex, t), TaskScheduler.Default);
    }

    private void OnPrefetchCompleted(int blockIndex, Task<CycleBlock> task)
    {
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
        CycleBlock? active = _activeBlock;
        if (active is not null && blockIndex != active.Index + 1)
        {
            return;
        }

        if (_prefetchedBlock is null || _prefetchedBlock.Index < blockIndex)
        {
            _prefetchedBlock = block;
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
            ComputeCyclesGpu(start, cycles);
        }
        else
        {
            ComputeCyclesCpu(start, cycles);
        }

        return new CycleBlock(blockIndex, start, cycles);
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
            destination[i] = MersenneDivisorCycles.CalculateCycleLength(divisor);
            divisor++;
        }
    }

    private void ComputeCyclesGpu(ulong start, ulong[] destination)
    {
        var gpuLease = GpuKernelPool.GetKernel(_useGpuGeneration);
        var execution = gpuLease.EnterExecutionScope();

        try
        {
            Accelerator accelerator = gpuLease.Accelerator;
            Action<Index1D, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>> kernel = _gpuKernelCache.GetOrAdd(accelerator, LoadKernel);

            int length = destination.Length;
            ulong[] divisors = ArrayPool<ulong>.Shared.Rent(length);

            try
            {
                for (int i = 0; i < length; i++)
                {
                    divisors[i] = start + (ulong)i;
                }

                using MemoryBuffer1D<ulong, Stride1D.Dense> divisorBuffer = accelerator.Allocate1D<ulong>(length);
                using MemoryBuffer1D<ulong, Stride1D.Dense> resultBuffer = accelerator.Allocate1D<ulong>(length);

                divisorBuffer.View.CopyFromCPU(ref divisors[0], length);
                kernel(length, divisorBuffer.View, resultBuffer.View);
                accelerator.Synchronize();
                resultBuffer.View.CopyToCPU(ref destination[0], length);
            }
            finally
            {
                ArrayPool<ulong>.Shared.Return(divisors, clearArray: false);
            }
        }
        finally
        {
            execution.Dispose();
            gpuLease.Dispose();
        }
    }

    private static Action<Index1D, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>> LoadKernel(Accelerator accelerator)
    {
        return accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>(GpuDivisorCycleKernel);
    }

    private static void GpuDivisorCycleKernel(Index1D index, ArrayView1D<ulong, Stride1D.Dense> divisors, ArrayView1D<ulong, Stride1D.Dense> cycles)
    {
        cycles[index] = MersenneDivisorCycles.CalculateCycleLengthGpu(divisors[index]);
    }

    private sealed class AcceleratorReferenceComparer : IEqualityComparer<Accelerator>
    {
        internal static AcceleratorReferenceComparer Instance { get; } = new();

        public bool Equals(Accelerator? x, Accelerator? y) => ReferenceEquals(x, y);

        public int GetHashCode(Accelerator obj) => RuntimeHelpers.GetHashCode(obj);
    }
}
