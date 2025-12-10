using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core.Cpu;

internal static class DeterministicRandomCpu
{
    [ThreadStatic]
    private static ulong s_threadState;

    private static long s_seedCounter;

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static ulong NextUInt64()
    {
        ulong state = s_threadState;
        if (state == 0UL)
        {
            state = InitializeThreadState();
        }

        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        s_threadState = state;
        return state * 2685821657736338717UL;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static UInt128 NextUInt128()
    {
        UInt128 high = (UInt128)NextUInt64();
        UInt128 low = (UInt128)NextUInt64();
        return (high << 64) | low;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static ulong InitializeThreadState()
    {
        ulong seed = (ulong)Interlocked.Increment(ref s_seedCounter);
        seed += 0x9E3779B97F4A7C15UL;
        seed = (seed ^ (seed >> 30)) * 0xBF58476D1CE4E5B9UL;
        seed = (seed ^ (seed >> 27)) * 0x94D049BB133111EBUL;
        seed ^= seed >> 31;
        if (seed == 0UL)
        {
            seed = 0x9E3779B97F4A7C15UL;
        }

        return seed;
    }
}
