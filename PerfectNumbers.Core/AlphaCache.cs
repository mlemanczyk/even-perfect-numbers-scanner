using PeterO.Numbers;

namespace PerfectNumbers.Core;

public sealed class AlphaCache
{
    private readonly Dictionary<(EInteger P, int K), AlphaValues> _cache = new();

    public AlphaValues Get(EInteger p, int k)
    {
        // TODO: Replace this dictionary lookup with a lock-free cache that reuses pooled AlphaValues so alpha requests avoid
        // redoing the expensive PowerCache.Get computations across threads.
        if (!_cache.TryGetValue((p, k), out var values))
        {
            ERational alphaPValue = AlphaCalculations.ComputeAlphaP(p, k);
            ERational alphaMValue = RationalNumbers.Two.Divide(alphaPValue);
            values = new AlphaValues(alphaMValue, alphaPValue);
            _cache[(p, k)] = values;
        }

        return values;
    }

    public void Clear()
    {
        // TODO: Return pooled AlphaValues instances to a shared cache before clearing so repeated warm-ups do not thrash the
        // allocator when alpha tables refresh during large divisor scans.
        _cache.Clear();
    }

    public int Count => _cache.Count;

    public readonly struct AlphaValues
    {
        public readonly ERational AlphaM;
        public readonly ERational AlphaP;

        public AlphaValues(ERational alphaM, ERational alphaP)
        {
            AlphaM = alphaM;
            AlphaP = alphaP;
        }
    }
}

