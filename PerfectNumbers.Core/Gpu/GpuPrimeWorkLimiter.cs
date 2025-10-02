namespace PerfectNumbers.Core.Gpu;

public static class GpuPrimeWorkLimiter
{
    private static SemaphoreSlim _semaphore = new(1, int.MaxValue);
    private static readonly object _lock = new();
    private static int _currentLimit = 1;

    public static IDisposable Acquire()
    {
        // TODO: Inline the pooled limiter guard from the GpuLimiterThroughputBenchmarks so prime-sieve
        // GPU jobs stop allocating Releaser instances on every acquisition.
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

            // TODO: Switch to a shared limiter implementation with GpuWorkLimiter so we can coordinate CPU/GPU prime work using
            // the same semaphore pool without reallocating per adjustment.
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

