using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public readonly struct MontgomeryDivisorData(ulong modulus, ulong nPrime, ulong montgomeryOne, ulong montgomeryTwo)
{
    public ulong Modulus { get; } = modulus;

    public ulong NPrime { get; } = nPrime;

    public ulong MontgomeryOne { get; } = montgomeryOne;

    public ulong MontgomeryTwo { get; } = montgomeryTwo;
}

internal static class MontgomeryDivisorDataCache
{
    private const ulong BaseBlockEnd = PerfectNumberConstants.MaxQForDivisorCycles;

    private static readonly ConcurrentDictionary<ulong, CacheEntry> Cache = new();
    private static readonly ConcurrentDictionary<int, ConcurrentDictionary<ulong, byte>> BlockMembership = new();

    public static MontgomeryDivisorData Get(ulong modulus)
    {
        if (modulus <= 1UL || (modulus & 1UL) == 0UL)
        {
            return new MontgomeryDivisorData(modulus, 0UL, 0UL, 0UL);
        }

        int blockIndex = ComputeBlockIndex(modulus);
        return GetManaged(modulus, blockIndex);
    }

    internal static void ReleaseBlock(DivisorCycleCache.CycleBlock block)
    {
        if (block.Index <= 0)
        {
            return;
        }

        if (BlockMembership.TryRemove(block.Index, out ConcurrentDictionary<ulong, byte>? members))
        {
            foreach (ulong modulus in members.Keys)
            {
                Cache.TryRemove(modulus, out _);
            }
        }
    }

    private static MontgomeryDivisorData GetManaged(ulong modulus, int blockIndex)
    {
        while (true)
        {
            if (Cache.TryGetValue(modulus, out CacheEntry entry))
            {
                if (entry.BlockIndex != blockIndex)
                {
                    if (blockIndex > 0 && entry.BlockIndex <= 0)
                    {
                        CacheEntry updated = new(entry.Data, blockIndex);
                        if (Cache.TryUpdate(modulus, updated, entry))
                        {
                            RegisterBlockMembership(modulus, blockIndex);
                            return entry.Data;
                        }

                        continue;
                    }

                    if (blockIndex > 0)
                    {
                        RegisterBlockMembership(modulus, blockIndex);
                    }

                    return entry.Data;
                }
                return entry.Data;
            }

            MontgomeryDivisorData created = Create(modulus);
            if (Cache.TryAdd(modulus, new CacheEntry(created, blockIndex)))
            {
                RegisterBlockMembership(modulus, blockIndex);
                return created;
            }
        }
    }

    private static void RegisterBlockMembership(ulong modulus, int blockIndex)
    {
        if (blockIndex <= 0)
        {
            return;
        }

        ConcurrentDictionary<ulong, byte> set = BlockMembership.GetOrAdd(blockIndex, static _ => new ConcurrentDictionary<ulong, byte>());
        set.TryAdd(modulus, 0);
    }

    private static int ComputeBlockIndex(ulong modulus)
    {
        if (modulus <= BaseBlockEnd)
        {
            return 0;
        }

        return (int)((modulus - (BaseBlockEnd + 1UL)) / (ulong)PerfectNumberConstants.MaxQForDivisorCycles) + 1;
    }

    private static MontgomeryDivisorData Create(ulong modulus)
    {
        return new MontgomeryDivisorData(
            modulus,
            ComputeMontgomeryNPrime(modulus),
            ComputeMontgomeryResidue(1UL, modulus),
            ComputeMontgomeryResidue(2UL, modulus));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    // TODO: Replace this generic `%` reduction with the UInt128 Montgomery folding helper measured faster in
    // MontgomeryMultiplyBenchmarks so CPU paths avoid the slow BigInteger-style remainder in hot divisor cache lookups.
    private static ulong ComputeMontgomeryResidue(ulong value, ulong modulus) => (ulong)((UInt128)value * (UInt128.One << 64) % modulus);

    private static ulong ComputeMontgomeryNPrime(ulong modulus)
    {
        ulong inv = modulus;
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        return unchecked(0UL - inv);
    }

    private readonly struct CacheEntry
    {
        internal CacheEntry(MontgomeryDivisorData data, int blockIndex)
        {
            Data = data;
            BlockIndex = blockIndex;
        }

        internal MontgomeryDivisorData Data { get; }

        internal int BlockIndex { get; }
    }
}
