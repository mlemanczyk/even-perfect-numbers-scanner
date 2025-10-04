using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Threading;

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
    private static int s_modulusBitLimit = PrimeOrderGpuCapability.Default.ModulusBits;
    private static int s_exponentBitLimit = PrimeOrderGpuCapability.Default.ExponentBits;

    internal static void OverrideCapabilitiesForTesting(PrimeOrderGpuCapability capability)
    {
        Volatile.Write(ref s_modulusBitLimit, capability.ModulusBits);
        Volatile.Write(ref s_exponentBitLimit, capability.ExponentBits);
    }

    internal static void ResetCapabilitiesForTesting()
    {
        OverrideCapabilitiesForTesting(PrimeOrderGpuCapability.Default);
    }

    private static PrimeOrderGpuCapability Capabilities
        => new(Volatile.Read(ref s_modulusBitLimit), Volatile.Read(ref s_exponentBitLimit));

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

        PrimeOrderGpuCapability capability = Capabilities;

        if (!SupportsPrime(prime, capability))
        {
            MarkOverflow(prime);
            return GpuPow2ModStatus.Overflow;
        }

        if (!SupportsExponent(exponent, capability))
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

        PrimeOrderGpuCapability capability = Capabilities;

        if (!SupportsPrime(prime, capability))
        {
            MarkOverflow(prime);
            return GpuPow2ModStatus.Overflow;
        }

        for (int i = 0; i < exponents.Length; i++)
        {
            if (!SupportsExponent(exponents[i], capability))
            {
                return GpuPow2ModStatus.Overflow;
            }
        }

        // TODO: Implement the GPU heuristic order calculator so this path mirrors PrimeOrderCalculator before enabling device execution.
        return GpuPow2ModStatus.Unavailable;
    }

    private static bool SupportsPrime(ulong prime, PrimeOrderGpuCapability capability)
    {
        return GetBitLength(prime) <= capability.ModulusBits;
    }

    private static bool SupportsExponent(ulong exponent, PrimeOrderGpuCapability capability)
    {
        return GetBitLength(exponent) <= capability.ExponentBits;
    }

    private static int GetBitLength(ulong value)
    {
        if (value == 0UL)
        {
            return 0;
        }

        return 64 - BitOperations.LeadingZeroCount(value);
    }

    internal readonly record struct PrimeOrderGpuCapability(int ModulusBits, int ExponentBits)
    {
        public static PrimeOrderGpuCapability Default => new(128, 64);
    }

    }
