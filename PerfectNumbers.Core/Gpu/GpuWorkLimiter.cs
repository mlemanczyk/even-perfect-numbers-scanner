namespace PerfectNumbers.Core.Gpu;

public static class GpuWorkLimiter
{
    private static SemaphoreSlim _semaphore = new(1, int.MaxValue);
    private static readonly object _lock = new();
    private static int _currentLimit = 1;

    public static IDisposable Acquire()
    {
        var sem = _semaphore;
        sem.Wait();
        return new Releaser(sem);
    }

    public static void SetLimit(int value)
    {
        if (value < 1)
        {
            value = 1;
        }

        lock (_lock)
        {
            if (value == _currentLimit)
            {
                return;
            }

            // Swap to a new semaphore for the new limit. We intentionally
            // do not dispose the old semaphore to avoid releasing handles
            // that may still be in use by existing Releaser instances.
            _semaphore = new SemaphoreSlim(value, value);
            _currentLimit = value;
        }
    }

    private sealed class Releaser : IDisposable
    {
        private readonly SemaphoreSlim _sem;
        private bool _disposed;

        public Releaser(SemaphoreSlim sem)
        {
            _sem = sem;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _sem.Release();
            _disposed = true;
        }
    }
}

