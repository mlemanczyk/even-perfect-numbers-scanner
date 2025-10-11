using System;
using System.Buffers;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

internal static class ThreadLocalArrayPool<T>
{
    [ThreadStatic]
    private static ArrayPool<T>? s_threadPool;

    public static ArrayPool<T> Shared
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            ArrayPool<T>? pool = s_threadPool;
            if (pool is null)
            {
                pool = ArrayPool<T>.Create();
                s_threadPool = pool;
            }

            return pool;
        }
    }
}
