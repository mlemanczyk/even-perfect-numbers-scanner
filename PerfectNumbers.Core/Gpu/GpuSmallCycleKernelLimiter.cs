using System.Threading;

namespace PerfectNumbers.Core.Gpu;

public static class GpuSmallCycleKernelLimiter
{
    private static SemaphoreSlim _semaphore = new(512, int.MaxValue);
    private static readonly object _lock = new();
    private static int _currentLimit = 512;

    public static Lease Acquire()
    {
        var sem = _semaphore;
        sem.Wait();
        return new Lease(sem);
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

    public sealed class Lease : IDisposable
    {
        private readonly SemaphoreSlim _semaphore;
        private bool _disposed;

        internal Lease(SemaphoreSlim semaphore)
        {
            _semaphore = semaphore;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _semaphore.Release();
            _disposed = true;
        }
    }
}
