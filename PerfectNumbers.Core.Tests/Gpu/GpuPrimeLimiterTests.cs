using System.Diagnostics;
using FluentAssertions;
using PerfectNumbers.Core.Gpu;
using Xunit;

namespace PerfectNumbers.Core.Tests.Gpu;

[Collection("GpuNtt")]
[Trait("Category", "Fast")]
public class GpuPrimeLimiterTests
{
    [Fact]
    public async Task Acquire_blocks_when_limit_reached()
    {
        // Ensure limit = 1 so second acquire must wait.
        GpuPrimeWorkLimiter.SetLimit(1);

        var startedSecond = new TaskCompletionSource<bool>();
        var sw = new Stopwatch();

        GpuPrimeWorkLimiter.Acquire();
        sw.Start();

        var secondTask = Task.Run(() =>
        {
            GpuPrimeWorkLimiter.Acquire();
            try
            {
                startedSecond.TrySetResult(true);
            }
            finally
            {
				GpuPrimeWorkLimiter.Release();
            }
        });

        try
        {
            // Give secondTask a moment to attempt acquire (it should block).
            await Task.Delay(50);
            startedSecond.Task.IsCompleted.Should().BeFalse();
        }
        finally
        {
            // Release first after the initial wait so the second task can proceed.
			GpuPrimeWorkLimiter.Release();
        }

        await secondTask.WaitAsync(TimeSpan.FromSeconds(1));
        sw.Stop();

        // Should be > ~50ms because second had to wait for first to release.
        sw.ElapsedMilliseconds.Should().BeGreaterThanOrEqualTo(50);
    }
}

