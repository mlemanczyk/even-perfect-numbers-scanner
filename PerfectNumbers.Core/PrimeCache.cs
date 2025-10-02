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
            // TODO: Swap the Open.Numeric enumerator for the staged sieve batches we benchmarked; the
            // iterator allocations here throttle the cache fill when we extend the search past 138M.
            // TODO: Fold in the Mod6 stride planner that topped Mod6ComparisonBenchmarks so the CPU cache
            // can skip even/composite candidates instead of walking each enumerator element.
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
            // TODO: Move this incremental append to the shared sieve batches so we amortize the
            // conversions; walking the enumerator element-by-element is noticeably slower in the
            // updated prime cache benchmarks.
            // TODO: Apply the Mod6 stride scheduler here as well so on-demand extensions mirror the
            // Mod6ComparisonBenchmarks winner when pulling primes for divisor-cycle generation.
            if (current >= s)
            {
                yield return val;
            }
        }
    }
}

