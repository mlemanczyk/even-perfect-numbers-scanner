using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

/// <summary>
/// Lightweight semaphore-like guard that uses spinning and optional sleeps instead of locks.
/// It allows a small transient oversubscription while contending threads back off between checks.
/// </summary>
public sealed class PollingSemaphore
{
	private readonly int _maximumConcurrency;
	private readonly TimeSpan _pauseBetweenChecks;
	private readonly int _attemptsBeforeSleep;

	private readonly bool _hasPause;
	private int _currentHolders;

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public PollingSemaphore(int maximumConcurrency, TimeSpan pauseBetweenChecks, int attemptsBeforeSleep = 10)
	{
		ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(maximumConcurrency, 0);
		ArgumentOutOfRangeException.ThrowIfLessThan(pauseBetweenChecks, TimeSpan.Zero);

		_maximumConcurrency = maximumConcurrency;
		_pauseBetweenChecks = pauseBetweenChecks;
		_attemptsBeforeSleep = attemptsBeforeSleep;

		_hasPause = pauseBetweenChecks > TimeSpan.Zero;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public bool TryEnter()
	{
		int newValue = Interlocked.Increment(ref _currentHolders);
		if (newValue <= _maximumConcurrency)
		{
			return true;
		}

		Interlocked.Decrement(ref _currentHolders);
		return false;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public bool Wait(TimeSpan timeout, CancellationToken cancellationToken = default)
	{
		ArgumentOutOfRangeException.ThrowIfLessThan(timeout, Timeout.InfiniteTimeSpan);

		bool hasPause = _hasPause;
		bool hasStopwatch = timeout != Timeout.InfiniteTimeSpan;
		int attemptsBeforeSleep = _attemptsBeforeSleep;
		SpinWait spinner = new();
		TimeSpan pauseBetweenChecks = _pauseBetweenChecks;
		Stopwatch? stopwatch = hasStopwatch ? Stopwatch.StartNew() : null;
		while (true)
		{
			cancellationToken.ThrowIfCancellationRequested();

			if (TryEnter())
			{
				return true;
			}

			if (hasStopwatch && stopwatch!.Elapsed >= timeout)
			{
				return false;
			}

			if (spinner.Count < attemptsBeforeSleep)
			{
				spinner.SpinOnce();
				continue;
			}

			if (hasPause)
			{
				Thread.Sleep(pauseBetweenChecks);
				spinner.Reset();
			}
			else
			{
				Thread.Yield();
			}
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public bool Wait(CancellationToken cancellationToken) => Wait(Timeout.InfiniteTimeSpan, cancellationToken);

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public bool Wait() => Wait(Timeout.InfiniteTimeSpan, CancellationToken.None);

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public void Release() => _ = Interlocked.Decrement(ref _currentHolders);
}
