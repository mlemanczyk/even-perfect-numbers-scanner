using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace PerfectNumbers.Core;

public sealed class DivisorCycleCache
{
        internal sealed class CycleBlock
        {
                private int _referenceCount;

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
                internal void AddRef()
                {
                        Interlocked.Increment(ref _referenceCount);
                }

                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                internal void Release()
                {
                        Interlocked.Decrement(ref _referenceCount);
                }

                internal bool IsInUse => Volatile.Read(ref _referenceCount) > 0;

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
                private readonly DivisorCycleCache? _owner;
                private CycleBlock? _block;

                internal Lease(DivisorCycleCache owner, CycleBlock block)
                {
                        _owner = owner;
                        _block = block;
                        _block.AddRef();
                }

                public bool IsValid => _block is not null;

                public ulong Start => _block?.Start ?? 0UL;

                public ulong End => _block?.End ?? 0UL;

                public ulong[]? Values => _block?.Cycles;

                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                public ulong GetCycle(ulong divisor)
                {
                        return _block?.GetCycle(divisor) ?? 0UL;
                }

                public void Dispose()
                {
                        CycleBlock? block = _block;
                        if (block is null)
                        {
                                return;
                        }

                        block.Release();
                        _owner?.Release(block);
                        _block = null;
                }
        }

        private readonly object _sync = new();
        private readonly ConcurrentDictionary<int, Task<CycleBlock>> _pending = new();
        private CycleBlock _baseBlock = null!;
        private CycleBlock? _previous;
        private CycleBlock? _current;
        private CycleBlock? _next;
        private bool _initialized;

        private static readonly Lazy<DivisorCycleCache> _lazy = new(() => new DivisorCycleCache());

        public static DivisorCycleCache Shared => _lazy.Value;

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

        private void Initialize(ulong[] snapshot)
        {
                lock (_sync)
                {
                        _pending.Clear();
                        _baseBlock = new CycleBlock(index: 0, start: 0UL, snapshot);
                        _previous = _baseBlock;
                        _current = null;
                        _next = null;
                        _initialized = true;
                        StartPrefetchLocked(1);
                }
        }

        public Lease Acquire(ulong divisor)
        {
                if (!_initialized)
                {
                        throw new InvalidOperationException("DivisorCycleCache must be initialized before use.");
                }

                if (divisor <= _baseBlock.End)
                {
                        return new Lease(this, _baseBlock);
                }

                CycleBlock block = AcquireDynamicBlock(divisor);
                return new Lease(this, block);
        }

        private CycleBlock AcquireDynamicBlock(ulong divisor)
        {
                int blockIndex = GetBlockIndex(divisor);
                while (true)
                {
                        Task<CycleBlock>? taskToAwait = null;
                        lock (_sync)
                        {
                                if (TryGetCachedBlockLocked(blockIndex, out CycleBlock? cached))
                                {
                                        return cached!;
                                }

                                if (!_pending.TryGetValue(blockIndex, out taskToAwait))
                                {
                                        taskToAwait = Task.Run(() => GenerateBlock(blockIndex));
                                        _pending[blockIndex] = taskToAwait;
                                }
                        }

                        CycleBlock block = taskToAwait.GetAwaiter().GetResult();
                        lock (_sync)
                        {
                                _pending.TryRemove(blockIndex, out _);

                                if (TryGetCachedBlockLocked(blockIndex, out CycleBlock? cachedAfterAwait))
                                {
                                        return cachedAfterAwait!;
                                }

                                PromoteBlockLocked(block);
                                return block;
                        }
                }
        }

        private bool TryGetCachedBlockLocked(int blockIndex, out CycleBlock? block)
        {
                if (_current is { Index: var currentIndex } current && currentIndex == blockIndex)
                {
                        block = current;
                        return true;
                }

                if (_previous is { Index: var previousIndex } previous && previousIndex == blockIndex)
                {
                        block = previous;
                        return true;
                }

                if (_next is { Index: var nextIndex } next && nextIndex == blockIndex)
                {
                        PromoteNextLocked();
                        block = _current!;
                        return true;
                }

                block = null;
                return false;
        }

        private void PromoteNextLocked()
        {
                if (_next is null)
                {
                        return;
                }

                CycleBlock? oldPrevious = _previous;
                _previous = _current ?? _baseBlock;
                _current = _next;
                _next = null;
                StartPrefetchLocked(_current!.Index + 1);
                TryReleaseBlockLocked(oldPrevious);
        }

        private void PromoteBlockLocked(CycleBlock block)
        {
                if (_current is null)
                {
                        _previous = _baseBlock;
                        _current = block;
                        StartPrefetchLocked(block.Index + 1);
                        return;
                }

                if (block.Index > _current.Index)
                {
                        CycleBlock? oldPrevious = _previous;
                        _previous = _current;
                        _current = block;
                        StartPrefetchLocked(block.Index + 1);
                        TryReleaseBlockLocked(oldPrevious);
                        return;
                }

                if (_previous is null || block.Index >= _previous.Index)
                {
                        CycleBlock? oldPrevious = _previous;
                        _previous = block;
                        TryReleaseBlockLocked(oldPrevious);
                }
        }

        private void StartPrefetchLocked(int blockIndex)
        {
                if (blockIndex <= 0)
                {
                        return;
                }

                if (_next is { Index: var nextIndex } next && nextIndex == blockIndex)
                {
                        return;
                }

                if (_pending.TryGetValue(blockIndex, out Task<CycleBlock>? existing))
                {
                        if (existing.IsCompletedSuccessfully)
                        {
                                if (_next is null || _next.Index < blockIndex)
                                {
                                        _next = existing.Result;
                                }

                                _pending.TryRemove(blockIndex, out _);
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
                        if (_next is null || _next.Index < blockIndex)
                        {
                                _next = task.Result;
                        }

                        _pending.TryRemove(blockIndex, out _);
                        TrimCacheLocked();
                }
        }

        private void TrimCacheLocked()
        {
                if (_previous is null)
                {
                        return;
                }

                if (ReferenceEquals(_previous, _baseBlock))
                {
                        return;
                }

                if (_previous.IsInUse)
                {
                        return;
                }

                if (_next is not null && _current is not null)
                {
                        _previous = null;
                }
        }

        private void TryReleaseBlockLocked(CycleBlock? block)
        {
                if (block is null || ReferenceEquals(block, _baseBlock))
                {
                        return;
                }

                if (ReferenceEquals(block, _current) || ReferenceEquals(block, _next))
                {
                        return;
                }

                if (block.IsInUse)
                {
                        return;
                }

                if (ReferenceEquals(block, _previous))
                {
                        _previous = null;
                }
        }

        private void Release(CycleBlock block)
        {
                if (!block.IsInUse)
                {
                        lock (_sync)
                        {
                                TryReleaseBlockLocked(block);
                        }
                }
        }

        private CycleBlock GenerateBlock(int blockIndex)
        {
                if (blockIndex == 0)
                {
                        return _baseBlock;
                }

                ulong start = GetBlockStart(blockIndex);
                int length = GetBlockLength(blockIndex);
                ulong[] cycles = new ulong[length];
                ulong divisor = start;
                for (int i = 0; i < length; i++)
                {
                        cycles[i] = MersenneDivisorCycles.CalculateCycleLength(divisor);
                        divisor++;
                }

                return new CycleBlock(blockIndex, start, cycles);
        }

        private int GetBlockLength(int blockIndex)
        {
                if (blockIndex == 0)
                {
                        return _baseBlock.Cycles.Length;
                }

                return PerfectNumberConstants.MaxQForDivisorCycles;
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

        private int GetBlockIndex(ulong divisor)
        {
                if (divisor <= _baseBlock.End)
                {
                        return 0;
                }

                ulong offset = divisor - (_baseBlock.End + 1UL);
                return (int)(offset / (ulong)PerfectNumberConstants.MaxQForDivisorCycles) + 1;
        }
}
