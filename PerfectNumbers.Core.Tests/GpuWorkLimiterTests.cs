using System.Diagnostics;
using FluentAssertions;
using PerfectNumbers.Core.Gpu;
using Xunit;

namespace PerfectNumbers.Core.Tests;

[Collection("GpuNtt")]
public class GpuWorkLimiterTests
{
    [Fact]
    [Trait("Category", "Fast")]
    public async Task Acquire_blocks_when_limit_reached()
    {
        GpuWorkLimiter.SetLimit(1);

        var startedSecond = new TaskCompletionSource<bool>();
        var sw = new Stopwatch();

        using var first = GpuWorkLimiter.Acquire();
        sw.Start();

        var secondTask = Task.Run(() =>
        {
            using var second = GpuWorkLimiter.Acquire();
            startedSecond.TrySetResult(true);
        });

        await Task.Delay(50);
        startedSecond.Task.IsCompleted.Should().BeFalse();

        first.Dispose();

        await secondTask.WaitAsync(TimeSpan.FromSeconds(1));
        sw.Stop();

        sw.ElapsedMilliseconds.Should().BeGreaterThanOrEqualTo(50);
    }
}
