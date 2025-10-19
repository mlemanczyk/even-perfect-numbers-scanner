using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace PerfectNumbers.Core;

public sealed class UnboundedTaskScheduler : TaskScheduler
{
    private static readonly object InstanceLock = new();
    private static UnboundedTaskScheduler? _instance;
    private static int _configuredThreadCount = Environment.ProcessorCount;

    public static UnboundedTaskScheduler Instance
    {
        get
        {
            lock (InstanceLock)
            {
                _instance ??= new UnboundedTaskScheduler(GetNormalizedThreadCount(_configuredThreadCount));
                return _instance;
            }
        }
    }

    public static int ConfiguredThreadCount => GetNormalizedThreadCount(Volatile.Read(ref _configuredThreadCount));

    private readonly object _poolLock = new();
    private TaskThreadPool _threadPool;

    private UnboundedTaskScheduler(int threadCount)
    {
        _threadPool = new TaskThreadPool(threadCount, ExecuteTask);
    }

    public static void ConfigureThreadCount(int threadCount)
    {
        int normalizedThreadCount = GetNormalizedThreadCount(threadCount);

        lock (InstanceLock)
        {
            Volatile.Write(ref _configuredThreadCount, normalizedThreadCount);
            PrimeOrderCalculator.PrimeOrderSearchConfig.ConfigureHeuristicDefault(normalizedThreadCount);

            if (_instance is null)
            {
                return;
            }

            _instance.UpdateThreadPool(normalizedThreadCount);
        }
    }

    public override int MaximumConcurrencyLevel => int.MaxValue;

    protected override void QueueTask(Task task)
    {
        _threadPool.Queue(task);
    }

    private void UpdateThreadPool(int threadCount)
    {
        lock (_poolLock)
        {
            if (_threadPool.ThreadCount == threadCount)
            {
                return;
            }

            TaskThreadPool newPool = new(threadCount, ExecuteTask);
            TaskThreadPool oldPool = _threadPool;
            _threadPool = newPool;
            oldPool.Dispose();
        }
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

    private static int GetNormalizedThreadCount(int threadCount)
    {
        return threadCount < 1 ? 1 : threadCount;
    }
}
