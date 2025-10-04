using System;
using System.Collections.Concurrent;

namespace PerfectNumbers.Core.Gpu;

internal enum GpuPow2ModStatus
{
    Success,
    Overflow,
    Unavailable,
}

internal static class PrimeOrderGpuHeuristics
{
    private static readonly ConcurrentDictionary<ulong, byte> OverflowedPrimes = new();

    internal static void MarkOverflow(ulong prime) => OverflowedPrimes[prime] = 0;

    internal static void ClearOverflow(ulong prime) => OverflowedPrimes.TryRemove(prime, out _);

    internal static void ClearAllOverflowForTesting() => OverflowedPrimes.Clear();

    public static GpuPow2ModStatus TryPow2Mod(ulong exponent, ulong prime, out ulong remainder)
    {
        remainder = 0UL;
        if (prime <= 1UL)
        {
            return GpuPow2ModStatus.Unavailable;
        }

        if (OverflowedPrimes.ContainsKey(prime))
        {
            return GpuPow2ModStatus.Overflow;
        }

        // TODO: Implement the GPU heuristic order calculator so this path mirrors PrimeOrderCalculator before enabling device execution.
        return GpuPow2ModStatus.Unavailable;
    }

    public static GpuPow2ModStatus TryPow2ModBatch(ReadOnlySpan<ulong> exponents, ulong prime, Span<ulong> remainders)
    {
        if (exponents.Length == 0)
        {
            return GpuPow2ModStatus.Success;
        }

        if (remainders.Length < exponents.Length)
        {
            throw new ArgumentException("Remainder span is shorter than the exponent span.", nameof(remainders));
        }

        remainders.Slice(0, exponents.Length).Clear();

        if (prime <= 1UL)
        {
            return GpuPow2ModStatus.Unavailable;
        }

        if (OverflowedPrimes.ContainsKey(prime))
        {
            return GpuPow2ModStatus.Overflow;
        }

        // TODO: Implement the GPU heuristic order calculator so this path mirrors PrimeOrderCalculator before enabling device execution.
        return GpuPow2ModStatus.Unavailable;
    }
}
