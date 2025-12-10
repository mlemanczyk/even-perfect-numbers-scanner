namespace PerfectNumbers.Core;

public sealed class UnboundedTaskScheduler : TaskScheduler
{
    private static UnboundedTaskScheduler? _instance;
    private static int _configuredThreadCount = Environment.ProcessorCount;

    public static UnboundedTaskScheduler Instance => _instance ??= new UnboundedTaskScheduler(_configuredThreadCount);

    public static int ConfiguredThreadCount => _configuredThreadCount;

    private TaskThreadPool _threadPool;

    private UnboundedTaskScheduler(int threadCount)
    {
        _threadPool = new TaskThreadPool(threadCount, ExecuteTask);
    }

    public static void ConfigureThreadCount(int threadCount)
    {
        _configuredThreadCount = threadCount;
        PrimeOrderCalculator.PrimeOrderCalculatorConfig.ConfigureHeuristicDefault(threadCount);

        // On EvenPerfectBitScanner's execution path the scheduler instance is created after configuration,
        // so updating an existing pool would never run. Leave the update path disabled for now.
        // if (_instance is not null)
        // {
        //     _instance.UpdateThreadPool(threadCount);
        // }
    }

    public override int MaximumConcurrencyLevel => int.MaxValue;

    protected override void QueueTask(Task task)
    {
        _threadPool.Queue(task);
    }

    private void UpdateThreadPool(int threadCount)
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
