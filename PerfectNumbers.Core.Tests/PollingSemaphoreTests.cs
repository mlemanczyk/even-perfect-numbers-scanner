using FluentAssertions;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class PollingSemaphoreTests
{
    [Fact]
    [Trait("Category", "Fast")]
    public void TryEnter_AllowsConfiguredConcurrency()
    {
        var semaphore = new PollingSemaphore(2, TimeSpan.Zero);

        semaphore.TryEnter().Should().BeTrue();
        semaphore.TryEnter().Should().BeTrue();
        semaphore.TryEnter().Should().BeFalse();

        semaphore.Release();
        semaphore.Release();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public async Task Wait_BlocksUntilRelease()
    {
        var semaphore = new PollingSemaphore(1, TimeSpan.FromMilliseconds(1));

        semaphore.TryEnter().Should().BeTrue();

        Task<bool> waiter = Task.Run(() => semaphore.Wait(TimeSpan.FromMilliseconds(200)));

        await Task.Delay(20);
        semaphore.Release();

        bool entered = await waiter;
        entered.Should().BeTrue();

        semaphore.Release();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void Wait_ReturnsFalseWhenTimeoutHits()
    {
        var semaphore = new PollingSemaphore(1, TimeSpan.FromMilliseconds(2));

        semaphore.TryEnter().Should().BeTrue();

        bool acquired = semaphore.Wait(TimeSpan.FromMilliseconds(30));

        acquired.Should().BeFalse();

        semaphore.Release();
    }
}
