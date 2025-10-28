using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace PerfectNumbers.Core;

internal sealed class TaskThreadPool
{
    private readonly struct WorkItem
    {
        public WorkItem(Task task)
        {
            Task = task;
        }

        public Task Task { get; }
    }

    private readonly ConcurrentQueue<WorkItem> _pendingTaskQueue = new();
    private readonly BlockingCollection<WorkItem> _pendingTasks;
    private readonly Thread[] _threads;
    private readonly Action<Task> _taskExecutor;
    public TaskThreadPool(int minimumThreads, Action<Task> taskExecutor)
    {
        if (minimumThreads <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(minimumThreads));
        }

        _taskExecutor = taskExecutor ?? throw new ArgumentNullException(nameof(taskExecutor));
        _pendingTasks = new BlockingCollection<WorkItem>(_pendingTaskQueue);
        _threads = new Thread[minimumThreads];

        for (int i = 0; i < minimumThreads; i++)
        {
            Thread thread = new(WorkerLoop)
            {
                IsBackground = true,
                Name = $"{nameof(TaskThreadPool)}-{i + 1}"
            };

            _threads[i] = thread;
            thread.Start();
        }
    }

    public int ThreadCount => _threads.Length;

    public void Queue(Task task)
    {
        if (task is null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        try
        {
            _pendingTasks.Add(new WorkItem(task));
        }
        catch (Exception ex)
        {
            if (ex is InvalidOperationException || ex is ObjectDisposedException)
            {
                throw new ObjectDisposedException(nameof(TaskThreadPool), ex);
            }

            throw;
        }
    }

    public IEnumerable<Task> GetScheduledTasks()
    {
        WorkItem[] snapshot = _pendingTaskQueue.ToArray();

        if (snapshot.Length == 0)
        {
            return Array.Empty<Task>();
        }

        Task[] tasks = new Task[snapshot.Length];

        for (int i = 0; i < snapshot.Length; i++)
        {
            tasks[i] = snapshot[i].Task;
        }

        return tasks;
    }

    public void Dispose()
    {
        _pendingTasks.CompleteAdding();

        foreach (Thread thread in _threads)
        {
            thread.Join();
        }

        _pendingTasks.Dispose();
    }

    private void WorkerLoop()
    {
        try
        {
            foreach (WorkItem workItem in _pendingTasks.GetConsumingEnumerable())
            {
                _taskExecutor(workItem.Task);
            }
        }
        catch (ObjectDisposedException)
        {
            // The collection was disposed while the worker was waiting. Exit gracefully.
        }
    }
}
