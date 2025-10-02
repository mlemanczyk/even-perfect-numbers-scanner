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

    private static readonly ConcurrentDictionary<ulong, CacheEntry> Cache = new(); // TODO: Replace this with a plain Dictionary that never mutates after startup once the divisor-cycle snapshot becomes read-only so Montgomery lookups avoid any concurrent collections.
    // TODO: Delete the block-membership tracking entirely when the divisor-cycle cache stops producing
    // dynamic blocks. The single-cycle plan keeps the snapshot immutable, so we should not retain block
    // indices or concurrent maps at all.
    private static readonly ConcurrentDictionary<int, ConcurrentDictionary<ulong, byte>> BlockMembership = new();

    public static MontgomeryDivisorData Get(ulong modulus)
    {
        if (modulus <= 1UL || (modulus & 1UL) == 0UL)
        {
            return new MontgomeryDivisorData(modulus, 0UL, 0UL, 0UL);
        }

        int blockIndex = ComputeBlockIndex(modulus);
        // TODO: Short-circuit block indices greater than zero once the cache stops prefetching future
        // blocks; when only the startup snapshot remains we should bypass block arithmetic entirely and
        // compute single cycles inline on cache misses without updating shared state.
        return GetManaged(modulus, blockIndex);
    }

    internal static void ReleaseBlock(DivisorCycleCache.CycleBlock block)
    {
        // TODO: Remove this entire release hook once the cycle cache no longer hands out dynamic blocks;
        // the single-cycle pipeline will never transfer ownership back to Montgomery metadata.
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
        // TODO: Collapse this retry loop once the cache becomes a simple snapshot dictionary. With the
        // single-cycle policy we can compute the requested cycle, populate the dictionary exactly once,
        // and return without spinning through concurrent retries.
    }

    private static void RegisterBlockMembership(ulong modulus, int blockIndex)
    {
        // TODO: Delete this method once block membership goes away; the single snapshot model must avoid
        // tracking additional indices or allocating concurrent dictionaries for Montgomery lookups.
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

        // TODO: Return 0 here once the divisor-cycle cache no longer tracks additional blocks so
        // MontgomeryDivisorData lookups never attempt to follow prefetched ranges or trigger new cycle
        // computations for entire blocks.
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
