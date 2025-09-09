using System.Buffers;
using PeterO.Numbers;
using Open.Numeric.Primes;

namespace PerfectNumbers.Core;

public sealed class PrimeCache
{
    private readonly List<EInteger> _primes = new();
    private readonly List<byte> _primeMod4 = new();
    private IEnumerator<ulong> _enumerator = Prime.Numbers.GetEnumerator();
    private static readonly EInteger[] EmptyEulerPrimesArray = Array.Empty<EInteger>();

    private void EnsurePrimesUpTo(EInteger limit)
    {
        EInteger last = _primes.Count > 0 ? _primes[^1] : EInteger.Zero;
        if (last.CompareTo(limit) >= 0)
        {
            return;
        }

        while (last.CompareTo(limit) < 0 && _enumerator.MoveNext())
        {
            ulong current = _enumerator.Current;
            last = current;
            _primes.Add(last);
            _primeMod4.Add((byte)(current & 3));
        }
    }

    public EInteger[] GetEulerPrimes(EInteger start, EInteger end)
    {
        EnsurePrimesUpTo(end);

        int index = _primes.BinarySearch(start);
        if (index < 0)
        {
            index = ~index;
        }

        EInteger[] primesArray = ArrayPool<EInteger>.Shared.Rent(_primes.Count);
        try
        {
            Span<EInteger> primes = new(primesArray);
            int count = 0;
            for (; index < _primes.Count; index++)
            {
                EInteger val = _primes[index];
                if (val.CompareTo(end) > 0)
                {
                    break;
                }
                if (_primeMod4[index] == 1)
                {
                    primes[count++] = val;
                }
            }

            return count > 0 ? primes[..count].ToArray() : EmptyEulerPrimesArray;
        }
        finally
        {
            ArrayPool<EInteger>.Shared.Return(primesArray);
        }
    }

    public IEnumerable<EInteger> EnumeratePrimes(EInteger start)
    {
        ulong s = (ulong)start;
        EnsurePrimesUpTo(start);

        int index = _primes.BinarySearch(s);
        if (index < 0)
        {
            index = ~index;
        }

        for (; index < _primes.Count; index++)
        {
            yield return _primes[index];
        }

        while (_enumerator.MoveNext())
        {
            ulong current = _enumerator.Current;
            EInteger val = EInteger.FromUInt64(current);
            _primes.Add(val);
            _primeMod4.Add((byte)(current & 3));
            if (current >= s)
            {
                yield return val;
            }
        }
    }
}

