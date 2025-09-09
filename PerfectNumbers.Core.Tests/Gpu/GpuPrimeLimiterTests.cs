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

        using var first = GpuPrimeWorkLimiter.Acquire();
        sw.Start();

        var secondTask = Task.Run(() =>
        {
            using var second = GpuPrimeWorkLimiter.Acquire();
            startedSecond.TrySetResult(true);
        });

        // Give secondTask a moment to attempt acquire (it should block).
        await Task.Delay(50);
        startedSecond.Task.IsCompleted.Should().BeFalse();

        // Release first after short delay, allowing second to continue.
        // (Dispose at scope end, but we want to end now.)
        first.Dispose();

        await secondTask.WaitAsync(TimeSpan.FromSeconds(1));
        sw.Stop();

        // Should be > ~50ms because second had to wait for first to release.
        sw.ElapsedMilliseconds.Should().BeGreaterThanOrEqualTo(50);
    }
}

