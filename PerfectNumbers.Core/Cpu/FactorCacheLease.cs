using System.Collections.Generic;

namespace PerfectNumbers.Core.Cpu;

internal struct FactorCacheLease
{
    private Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>? _cache;

    public void EnsureInitialized(ref Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>? cache)
    {
        Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>? current = cache;
        if (current is null)
        {
            current = ThreadStaticPools.RentMersenneFactorCacheDictionary();
            cache = current;
        }

        _cache = current;
    }

    public void Dispose()
    {
        Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>? cache = _cache;
        if (cache is null)
        {
            return;
        }

        ThreadStaticPools.ReturnMersenneFactorCacheDictionary(cache);
        _cache = null;
    }
}
