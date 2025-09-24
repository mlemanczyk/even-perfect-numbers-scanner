using System.Collections.Concurrent;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public sealed class DivisorCycleCache
{
	internal sealed class CycleBlock
	{
		// TODO: This can be removed. See other comments.
		private int _referenceCount;

		internal CycleBlock(int index, ulong start, ulong[] cycles)
		{
			Index = index;
			Start = start;
			Cycles = cycles;
			End = start + (ulong)cycles.Length - 1UL;
		}

		// TODO: Let's use fields instead of the properties below for better performance
		internal int Index { get; }

		internal ulong Start { get; }

		internal ulong End { get; }

		internal ulong[] Cycles { get; }

		// TODO: We don't need full reference tracking. It's enough to assign divisor block to a task, instead. We can remove excessive blocks as required.
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		internal void AddRef()
		{
			Interlocked.Increment(ref _referenceCount);
		}

		// TODO: We don't need full reference tracking. It's enough to assign divisor block to a task, instead. We can remove excessive blocks as required.
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		internal void Release()
		{
			Interlocked.Decrement(ref _referenceCount);
		}

		// TODO: We don't need full reference tracking. It's enough to assign divisor block to a task, instead. This can be removed
		internal bool IsInUse => Volatile.Read(ref _referenceCount) > 0;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		internal ulong GetCycle(ulong divisor)
		{
			// TODO: The production code should guarantee that this will never occur and we'll use divisors with their corresponding blocks, only. This can be removed and we can simplify this method to =>
			if (divisor < Start || divisor > End)
			{
				return 0UL;
			}

			return Cycles[(int)(divisor - Start)];
		}
	}

	public struct Lease : IDisposable
	{
		// TODO: We can probably remove _owner, if we don't need full cycle management which will help GC to clear these out
		private readonly DivisorCycleCache? _owner;
		private CycleBlock? _block;

		internal Lease(DivisorCycleCache owner, CycleBlock block)
		{
			_owner = owner;
			_block = block;
			_block.AddRef();
		}

		// TODO: We're highly over-complicating this. _block is always assigned by the constructor. We don't need to make it nullable. This can be removed.
		public readonly bool IsValid => _block is not null;

		// TODO: Once we make _block non-nullable we can simplify these to just fields assigned during the creation for the best performance.
		public readonly ulong Start => _block?.Start ?? 0UL;

		public readonly ulong End => _block?.End ?? 0UL;

		public readonly ulong[]? Values => _block?.Cycles;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public readonly ulong GetCycle(ulong divisor) => _block?.GetCycle(divisor) ?? 0UL;

		public void Dispose()
		{
			// TODO: We don't need full cycle block management. We can rely on GC to do its job. Thanks to that we could possibly make the struct read-only.
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
	// TODO: _baseBlock should be like no other block. Why do we need a separate property for it? It should become current upon initialization
	private CycleBlock _baseBlock = null!;
	private CycleBlock? _previous;
	private CycleBlock? _current;
	private CycleBlock? _next;
	private bool _initialized;

	// TODO: Let's not use lazy but directly create the instance, instead for better performance. Then we can remove _lazy and just assign the instance to Shared.
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
		// TODO: We don't need _initialized. The constructor calls Initialize and makes sure it's always initialized, before use. This check can be removed, too.
		if (!_initialized)
		{
			throw new InvalidOperationException("DivisorCycleCache must be initialized before use.");
		}

		// TODO: Do we need Lease instances, if we don't care who is using our blocks once they were given and just remove them internally from future use?
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
		// TODO: Why do we need this check? This should never happen in production code.
		if (blockIndex <= 0)
		{
			return;
		}

		// TODO: Why do we need this check? This should never happen in production code.
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
		// TODO: We're over-complicating this again. We should just remember and use the _baseBlock.Cycles.Length. Remember it in a field for better performance during initialization / creation
		if (blockIndex == 0)
		{
			return _baseBlock.Cycles.Length;
		}

		return PerfectNumberConstants.MaxQForDivisorCycles;
	}

	private ulong GetBlockStart(int blockIndex)
	{
		// TODO: We're over-complicating this. There should be only 3 blocks kept. We should simply check all 3 to return the correct one, instead of expensive multiplications.
		if (blockIndex == 0)
		{
			return 0UL;
		}

		ulong baseEnd = _baseBlock.End;
		return baseEnd + 1UL + (ulong)(blockIndex - 1) * (ulong)PerfectNumberConstants.MaxQForDivisorCycles;
	}

	private int GetBlockIndex(ulong divisor)
	{
		// TODO: We're over-complicating this. There should be only 3 blocks kept. We should simply check all 3 to return the correct one, instead of expensive multiplications.
		if (divisor <= _baseBlock.End)
		{
			return 0;
		}

		ulong offset = divisor - (_baseBlock.End + 1UL);
		return (int)(offset / (ulong)PerfectNumberConstants.MaxQForDivisorCycles) + 1;
	}
}
