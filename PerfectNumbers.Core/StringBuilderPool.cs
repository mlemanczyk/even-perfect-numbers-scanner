using System.Collections.Concurrent;
using System.Text;

namespace PerfectNumbers.Core;

public static class StringBuilderPool
{
    private static readonly ConcurrentQueue<StringBuilder> _stringBuilderPool = new();

    public static StringBuilder Rent() => _stringBuilderPool.TryDequeue(out var sb) ? sb : new();

    public static void Return(StringBuilder sb)
    {
        _ = sb.Clear();
        _stringBuilderPool.Enqueue(sb);
    }
}