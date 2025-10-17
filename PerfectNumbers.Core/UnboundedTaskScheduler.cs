using System.Threading;
using System.Threading.Tasks;

namespace PerfectNumbers.Core;

public sealed class UnboundedTaskScheduler : TaskScheduler
{
        public static UnboundedTaskScheduler Instance { get; } = new();

        private UnboundedTaskScheduler()
        {
        }

        public override int MaximumConcurrencyLevel => int.MaxValue;

        protected override void QueueTask(Task task)
        {
                ThreadPool.UnsafeQueueUserWorkItem(static state =>
                {
                        (UnboundedTaskScheduler Scheduler, Task Task) tuple = ((UnboundedTaskScheduler, Task))state!;
                        tuple.Scheduler.TryExecuteTask(tuple.Task);
                }, (this, task));
        }

        protected override bool TryExecuteTaskInline(Task task, bool taskWasPreviouslyQueued)
        {
                return TryExecuteTask(task);
        }

        protected override IEnumerable<Task>? GetScheduledTasks()
        {
                return null;
        }
}
