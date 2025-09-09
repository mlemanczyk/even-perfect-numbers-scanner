using PeterO.Numbers;

namespace PerfectNumbers.Core;

public sealed class AlphaCache
{
    private readonly Dictionary<(EInteger P, int K), AlphaValues> _cache = new();

    public AlphaValues Get(EInteger p, int k)
    {
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

