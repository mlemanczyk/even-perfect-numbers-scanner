namespace PerfectNumbers.Core.Gpu;

public static class GpuPrimeWorkLimiter
{
	private static readonly TimeSpan _pauseBetweenChecks = TimeSpan.FromSeconds(10);
	private static PollingSemaphore _semaphore = new(1, _pauseBetweenChecks);
	private static int _currentLimit = 1;

	public static void Acquire() => _semaphore.Wait();
	public static void Release() => _semaphore.Release();

	public static int CurrentLimit => _currentLimit;

	public static void SetLimit(int value)
	{
		if (value <= 0)
		{
			value = 1;
		}

		if (value == _currentLimit)
		{
			return;
		}

		// TODO: Switch to a shared limiter implementation with GpuWorkLimiter so we can coordinate CPU/GPU prime work using
		// the same semaphore pool without reallocating per adjustment.
		// var previous = _semaphore;
		_semaphore = new PollingSemaphore(value, _pauseBetweenChecks);
		_currentLimit = value;
		// previous.Dispose();
	}
}
