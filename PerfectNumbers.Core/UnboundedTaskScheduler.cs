namespace PerfectNumbers.Core;

public sealed class UnboundedTaskScheduler : TaskScheduler
{
	public static UnboundedTaskScheduler Instance { get; } = new();

	private UnboundedTaskScheduler()
	{
		ThreadPool.SetMaxThreads(100_000, 100_000);
		ThreadPool.SetMinThreads(10_340, 100);
	}

	public override int MaximumConcurrencyLevel => int.MaxValue;

	protected override void QueueTask(Task task)
	{
		ThreadPool.UnsafeQueueUserWorkItem( (state) =>
		{
			var (scheduler, task) = ((UnboundedTaskScheduler, Task))state!;
			scheduler.TryExecuteTask(task);
		}, (this, task));
	}

	protected override bool TryExecuteTaskInline(Task task, bool taskWasPreviouslyQueued) => TryExecuteTask(task);

	protected override IEnumerable<Task>? GetScheduledTasks() => null;
}
