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
    private readonly bool _hasPause;
    private int _currentHolders;

    public PollingSemaphore(int maximumConcurrency, TimeSpan pauseBetweenChecks)
    {
        if (maximumConcurrency <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maximumConcurrency));
        }

        if (pauseBetweenChecks < TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(pauseBetweenChecks));
        }

        _maximumConcurrency = maximumConcurrency;
        _pauseBetweenChecks = pauseBetweenChecks;
        _hasPause = pauseBetweenChecks > TimeSpan.Zero;
    }

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

    public bool Wait(TimeSpan timeout, CancellationToken cancellationToken = default)
    {
		ArgumentOutOfRangeException.ThrowIfLessThan(timeout, Timeout.InfiniteTimeSpan);

		Stopwatch? stopwatch = timeout == Timeout.InfiniteTimeSpan ? null : Stopwatch.StartNew();
        SpinWait spinner = new();

        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (TryEnter())
            {
                return true;
            }

            if (stopwatch is not null && stopwatch.Elapsed >= timeout)
            {
                return false;
            }

            if (spinner.Count < 10)
            {
                spinner.SpinOnce();
                continue;
            }

            if (_hasPause)
            {
                Thread.Sleep(_pauseBetweenChecks);
                spinner.Reset();
            }
            else
            {
                Thread.Yield();
            }
        }
    }

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public bool Wait(CancellationToken cancellationToken) => Wait(Timeout.InfiniteTimeSpan, cancellationToken);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public bool Wait() => Wait(Timeout.InfiniteTimeSpan, CancellationToken.None);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void Release()
    {
        int newValue = Interlocked.Decrement(ref _currentHolders);

        if (newValue < 0)
        {
            Interlocked.Increment(ref _currentHolders);
            throw new SynchronizationLockException("Release called more times than Wait.");
        }
    }
}
