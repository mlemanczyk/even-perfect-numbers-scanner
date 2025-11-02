using System.Threading;

namespace PerfectNumbers.Core.Gpu;

public static class GpuPrimeLeaseLimiter
{
        private static SemaphoreSlim _semaphore = new(int.MaxValue, int.MaxValue);
        private static int _currentLimit = int.MaxValue;

        public static void Initialize(int limit)
        {
                if (limit <= 0)
                {
                        limit = 1;
                }

                if (limit == _currentLimit)
                {
                        return;
                }

                var previous = _semaphore;
                _semaphore = new SemaphoreSlim(limit, limit);
                _currentLimit = limit;
                previous.Dispose();
        }

        public static void Enter()
        {
                _semaphore.Wait();
        }

        public static void Exit()
        {
                _semaphore.Release();
        }
}
