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
    private readonly HashSet<Thread> _runningThreads = new();
    private readonly object _lock = new();
    private readonly Action<Task> _taskExecutor;
    private readonly int _maximumConcurrency;
    private readonly SemaphoreSlim _workAvailable = new(0);
    private readonly CancellationTokenSource _disposeCancellation = new();
    private int _pendingTaskCount;
    private int _currentThreads;
    private int _threadId;

    public TaskThreadPool(int minimumThreads, Action<Task> taskExecutor)
    {
        if (minimumThreads <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(minimumThreads));
        }

        _taskExecutor = taskExecutor ?? throw new ArgumentNullException(nameof(taskExecutor));
        _maximumConcurrency = minimumThreads;

        List<Thread> threadsToStart = new(minimumThreads);

        lock (_lock)
        {
            for (int i = 0; i < minimumThreads; i++)
            {
                Thread thread = CreateWorkerThread();
                _runningThreads.Add(thread);
                _currentThreads++;
                threadsToStart.Add(thread);
            }
        }

        foreach (Thread thread in threadsToStart)
        {
            thread.Start();
        }
    }

    public int ThreadCount => _maximumConcurrency;

    public void Queue(Task task)
    {
        if (task is null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        List<Thread>? threadsToStart;

        lock (_lock)
        {
            if (_disposeCancellation.IsCancellationRequested)
            {
                throw new ObjectDisposedException(nameof(TaskThreadPool));
            }

            _pendingTaskQueue.Enqueue(new WorkItem(task));
            _pendingTaskCount++;

            threadsToStart = EnsureWorkerCapacityLocked(null);
        }

        _workAvailable.Release();

        if (threadsToStart is not null)
        {
            foreach (Thread thread in threadsToStart)
            {
                thread.Start();
            }
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
        Thread[] threadsToJoin;

        lock (_lock)
        {
            _disposeCancellation.Cancel();

            threadsToJoin = new Thread[_runningThreads.Count];
            _runningThreads.CopyTo(threadsToJoin);
        }

        if (threadsToJoin.Length > 0)
        {
            _workAvailable.Release(threadsToJoin.Length);
        }

        foreach (Thread thread in threadsToJoin)
        {
            thread.Join();
        }

        _workAvailable.Dispose();
        _disposeCancellation.Dispose();

        lock (_lock)
        {
            _runningThreads.Clear();

            while (_pendingTaskQueue.TryDequeue(out _))
            {
            }

            _pendingTaskCount = 0;
            _currentThreads = 0;
        }
    }

    private bool TryDequeueWork(out WorkItem workItem)
    {
        CancellationToken cancellationToken = _disposeCancellation.Token;

        while (true)
        {
            try
            {
                _workAvailable.Wait(cancellationToken);
            }
            catch (OperationCanceledException)
            {
                workItem = default;
                return false;
            }

            if (_pendingTaskQueue.TryDequeue(out workItem))
            {
                Interlocked.Decrement(ref _pendingTaskCount);
                return true;
            }
        }
    }

    private void WorkerLoop()
    {
        try
        {
            if (!TryDequeueWork(out WorkItem workItem))
            {
                return;
            }

            _taskExecutor(workItem.Task);
        }
        finally
        {
            CompleteWorker();
        }
    }

    private void CompleteWorker()
    {
        List<Thread>? threadsToStart = null;

        lock (_lock)
        {
            _runningThreads.Remove(Thread.CurrentThread);
            _currentThreads--;

            if (!_disposeCancellation.IsCancellationRequested)
            {
                threadsToStart = EnsureWorkerCapacityLocked(null);
            }
        }

        if (threadsToStart is not null)
        {
            foreach (Thread thread in threadsToStart)
            {
                thread.Start();
            }
        }
    }

    private List<Thread>? EnsureWorkerCapacityLocked(List<Thread>? threadsToStart)
    {
        int pendingTasks = Math.Min(_maximumConcurrency, Volatile.Read(ref _pendingTaskCount));

        while (!_disposeCancellation.IsCancellationRequested && _currentThreads < pendingTasks)
        {
            Thread thread = CreateWorkerThread();
            _runningThreads.Add(thread);
            _currentThreads++;

            threadsToStart ??= new List<Thread>();
            threadsToStart.Add(thread);
        }

        return threadsToStart;
    }

    private Thread CreateWorkerThread()
    {
        int threadIndex = Interlocked.Increment(ref _threadId);

        return new Thread(WorkerLoop)
        {
            IsBackground = true,
            Name = $"{nameof(TaskThreadPool)}-{threadIndex}"
        };
    }
}
