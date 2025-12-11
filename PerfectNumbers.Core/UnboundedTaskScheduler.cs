using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public sealed class UnboundedTaskScheduler : TaskScheduler
{
	private static UnboundedTaskScheduler? _instance;
	private static int _configuredThreadCount = Environment.ProcessorCount;

	public static UnboundedTaskScheduler Instance
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		get => _instance ??= new UnboundedTaskScheduler(_configuredThreadCount);
	}

	public static int ConfiguredThreadCount
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		get => _configuredThreadCount;
	}

	private TaskThreadPool _threadPool;

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private UnboundedTaskScheduler(int threadCount) => _threadPool = new TaskThreadPool(threadCount, ExecuteTask);

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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

	public override int MaximumConcurrencyLevel
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		get => int.MaxValue;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	protected override void QueueTask(Task task) => _threadPool.Queue(task);

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

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private void ExecuteTask(Task task) => TryExecuteTask(task);

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	protected override bool TryExecuteTaskInline(Task task, bool taskWasPreviouslyQueued) => TryExecuteTask(task);

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	protected override IEnumerable<Task>? GetScheduledTasks() => _threadPool.GetScheduledTasks();

}
