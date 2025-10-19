using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace PerfectNumbers.Core;

public sealed class UnboundedTaskScheduler : TaskScheduler
{
    public static UnboundedTaskScheduler Instance { get; } = new();

    private readonly TaskThreadPool _threadPool;

    private UnboundedTaskScheduler()
    {
        _threadPool = new TaskThreadPool(10_340, ExecuteTask);
    }

    public override int MaximumConcurrencyLevel => int.MaxValue;

    protected override void QueueTask(Task task)
    {
        _threadPool.Queue(task);
    }

    private void ExecuteTask(Task task)
    {
        TryExecuteTask(task);
    }

    protected override bool TryExecuteTaskInline(Task task, bool taskWasPreviouslyQueued)
    {
        return TryExecuteTask(task);
    }

    protected override IEnumerable<Task>? GetScheduledTasks()
    {
        return _threadPool.GetScheduledTasks();
    }
}
