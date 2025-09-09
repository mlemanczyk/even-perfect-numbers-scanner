namespace PerfectNumbers.Core.Gpu;

public static class GpuPrimeWorkLimiter
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

