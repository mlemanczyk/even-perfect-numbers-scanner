using System.Collections.Concurrent;
using System.Text;

namespace PerfectNumbers.Core;

public static class StringBuilderPool
{
    private static readonly ConcurrentQueue<StringBuilder> _stringBuilderPool = new();

    public static StringBuilder Rent()
    {
        // TODO: Ensure oversized builders remain eligible for reuse without trimming so call sites keep their full capacity.
        return _stringBuilderPool.TryDequeue(out var sb) ? sb : new();
    }

    public static void Return(StringBuilder sb)
    {
        _ = sb.Clear();
        // TODO: Preserve the builder's capacity when returning it so the pool hands the same buffer back without shrinkage.
        _stringBuilderPool.Enqueue(sb);
    }
}

