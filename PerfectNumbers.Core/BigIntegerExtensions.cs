using System.Numerics;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

/// <summary>
/// BigInteger helpers including cycle-remainder reduction using the subtract ladder that benchmarked best.
/// Shared stats/warm set stay here so ulong reductions can reuse them through ULongExtensions.
/// </summary>
public static partial class BigIntegerExtensions
{
    // private const int UnrolledSubtractCount = 6;
    // private static readonly HashSet<ulong> WarmModuli;
    // private static long _hits;
    // private static long _misses;

    // static BigIntegerExtensions()
    // {
    //     ulong[] snapshot = MersenneDivisorCycles.Shared.ExportSmallCyclesSnapshot();
    //     WarmModuli = new HashSet<ulong>(snapshot.Length);
    //     for (int i = 0; i < snapshot.Length; i++)
    //     {
    //         ulong cycle = snapshot[i];
    //         if (cycle > 1UL)
    //         {
    //             WarmModuli.Add(cycle);
    //         }
    //     }
    // }

    // public static void DumpCycleRemainderStats()
    // {
    //     long hits = Interlocked.Read(ref _hits);
    //     long misses = Interlocked.Read(ref _misses);
    //     Console.WriteLine($"CycleRemainderReducerCache: hits={hits}, misses={misses}");
    // }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static BigInteger ReduceCycleRemainder(this BigInteger value, BigInteger modulus)
    {
        // Track(modulus);
        // DumpCycleRemainderStats();

        if (value < modulus)
        {
            return value;
        }

        value -= modulus;
        if (value < modulus)
        {
            return value;
        }

        value -= modulus;
        if (value < modulus)
        {
            return value;
        }

        value -= modulus;
        if (value < modulus)
        {
            return value;
        }

        value -= modulus;
        if (value < modulus)
        {
            return value;
        }

        value -= modulus;
        if (value < modulus)
        {
            return value;
        }

        value -= modulus;
        if (value < modulus)
        {
            return value;
        }

        return value % modulus;
    }

    // [MethodImpl(MethodImplOptions.AggressiveInlining)]
    // internal static void Track(in BigInteger modulus)
    // {
    //     if (modulus <= ulong.MaxValue)
    //     {
    //         Track((ulong)modulus);
    //     }
    //     else
    //     {
    //         Interlocked.Increment(ref _misses);
    //     }
    // }

    // [MethodImpl(MethodImplOptions.AggressiveInlining)]
    // internal static void Track(ulong modulus)
    // {
    //     if (WarmModuli.Contains(modulus))
    //     {
    //         Interlocked.Increment(ref _hits);
    //     }
    //     else
    //     {
    //         Interlocked.Increment(ref _misses);
    //     }
    // }
}
