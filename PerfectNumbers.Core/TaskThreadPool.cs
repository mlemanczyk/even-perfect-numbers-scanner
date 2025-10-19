using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace PerfectNumbers.Core;

internal sealed class TaskThreadPool : IDisposable
{
    private readonly struct WorkItem
    {
        public WorkItem(Task task)
        {
            Task = task;
        }

        public Task Task { get; }
    }

    private readonly ConcurrentQueue<WorkItem> _pendingTasks = new();
    private readonly SemaphoreSlim _availableTasks = new(0);
    private readonly Thread[] _threads;
    private readonly Action<Task> _taskExecutor;
    private int _disposed;

    public TaskThreadPool(int minimumThreads, Action<Task> taskExecutor)
    {
        _taskExecutor = taskExecutor ?? throw new ArgumentNullException(nameof(taskExecutor));
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
        if (Volatile.Read(ref _disposed) != 0)
        {
            throw new ObjectDisposedException(nameof(TaskThreadPool));
        }

        _pendingTasks.Enqueue(new WorkItem(task));
        _availableTasks.Release();
    }

    public IEnumerable<Task> GetScheduledTasks()
    {
        WorkItem[] snapshot = _pendingTasks.ToArray();

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
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        for (int i = 0; i < _threads.Length; i++)
        {
            _availableTasks.Release();
        }

        foreach (Thread thread in _threads)
        {
            thread.Join();
        }

        _availableTasks.Dispose();
    }

    private void WorkerLoop()
    {
        while (true)
        {
            _availableTasks.Wait();

            if (Volatile.Read(ref _disposed) != 0)
            {
                return;
            }

            if (!_pendingTasks.TryDequeue(out WorkItem workItem))
            {
                continue;
            }

            _taskExecutor(workItem.Task);
        }
    }
}
