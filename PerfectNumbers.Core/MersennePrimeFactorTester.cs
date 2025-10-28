using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Numerics;
using UInt128 = System.UInt128;
using ILGPU;

namespace PerfectNumbers.Core;

public static class MersennePrimeFactorTester
{
	// private static ulong _isPrimeFactorHits;
	// private static ulong _factor64Hits;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsPrimeFactor(ulong p, ulong q, CancellationToken ct)
    {
        if (q < 2UL)
        {
            return false;
        }

        if ((q & 7UL) != 1UL && (q & 7UL) != 7UL)
        {
            return false;
        }

        if (ct.IsCancellationRequested)
        {
            return false;
        }

        // if (!HeuristicPrimeTester.Exclusive.IsPrime(q, ct))

		// Atomic.Add(ref _isPrimeFactorHits, 1UL);
		// Console.WriteLine($"MersennePrimeFactorTester.IsPrimeFactor hits {Volatile.Read(ref _isPrimeFactorHits)}");

        if (!PrimeTester.IsPrimeCpu(q, ct))
        {
            return false;
        }

        if (p < 2UL || (p & 1UL) == 0)
        {
            return false;
        }

        ulong value = checked(p << 1);
        // TODO: Replace this `%` with the benchmarked Math.DivRem fast path so the hot filter avoids
        // 128-bit division when evaluating candidates.
        if ((q - 1UL) % value != 0UL)
        {
            return false;
        }

        ulong order = GetOrderOf2ModPrime(q, ct);
        return !ct.IsCancellationRequested && order == p;
    }

    // TODO: Replace this dictionary with the divisor-cycle order cache once the benchmarks confirm the
    // shared cache can stream snapshot results without extra locking.
    private static readonly Dictionary<UInt128, UInt128> _orderCache = [];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ulong GetOrderOf2ModPrime(ulong q, CancellationToken ct) =>
        // TODO: Collapse this wrapper once all callers promote operands to UInt128 so we
        // can jump straight to OrderOf2ModPrime without bouncing through multiple overloads.
        (ulong)GetOrderOf2ModPrime((UInt128)q, ct);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static UInt128 GetOrderOf2ModPrime(UInt128 q, CancellationToken ct) =>
        // TODO: Inline this shim once callers can invoke OrderOf2ModPrime directly; the extra
        // hop shows up on hot perf traces while we chase large-order factors.
        OrderOf2ModPrime(q, ct);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static UInt128 OrderOf2ModPrime(UInt128 q, CancellationToken ct)
    {
        lock (_orderCache)
        {
            // TODO: Replace this lock with the lock-free order cache once the divisor-cycle snapshot exposes
            // deterministic ordering for single-cycle computations; we measured heavy contention in the
            // factor benchmarks when many threads warm the cache concurrently.
            if (_orderCache.TryGetValue(q, out var cached))
            {
                return cached;
            }
        }

        UInt128 phi = q - 1;
        UInt128[] factors = phi <= ulong.MaxValue
            ? Array.ConvertAll(Factor64((ulong)phi, ct), x => (UInt128)x)
            : Factor128(phi, ct);
        // TODO: Pull these factor arrays from ArrayPool once the factoring helpers adopt the pooled
        // buffers measured faster in FactorizationBenchmarks.
        UInt128 order = phi;

        UInt128 candidate, prime;
        int i, factorsCount = factors.Length;
        for (i = 0; i < factorsCount; i++)
        {
            prime = factors[i];
            if (ct.IsCancellationRequested)
            {
                return 0;
            }

            // TODO: Swap this `%`/`/` loop for the Math.DivRem-based reducer highlighted in the
            // order benchmarks so we avoid repeated 128-bit division while trimming the order.
            while (order % prime == 0)
            {
                if (ct.IsCancellationRequested)
                {
                    return 0;
                }

                candidate = order / prime;
                if (ModPowWithCancellation(UInt128Numbers.Two, candidate, q, ct) == UInt128.One)
                {
                    order = candidate;
                }
                else
                {
                    break;
                }
            }
        }

        // TODO: Integrate divisor-cycle data here so repeated order refinement uses cached cycle
        // lengths when available and computes missing orders on the configured device without
        // persisting them in the shared cache.
        if (!ct.IsCancellationRequested)
        {
            lock (_orderCache)
            {
                _orderCache[q] = order;
            }
        }

        return order;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static UInt128 ModPowWithCancellation(UInt128 baseValue, UInt128 exponent, UInt128 modulus, CancellationToken ct)
    {
        // Keep this helper local to the factoring path so we can preserve the token-aware early outs
        // without carrying a separate overload on the shared UInt128 extensions.
        if (modulus <= UInt128.One)
        {
            return UInt128.Zero;
        }

        UInt128 result = UInt128.One;
        UInt128 baseResidue = baseValue % modulus;

        while (exponent != UInt128.Zero)
        {
            if (ct.IsCancellationRequested)
            {
                return UInt128.Zero;
            }

            if ((exponent & UInt128.One) != UInt128.Zero)
            {
                result = result.MulMod(baseResidue, modulus);
            }

            if (ct.IsCancellationRequested)
            {
                return UInt128.Zero;
            }

            baseResidue = baseResidue.MulMod(baseResidue, modulus);
            exponent >>= 1;
        }

        return result;
    }

    private static readonly ConcurrentBag<List<ulong>> ListPool = new();
    private static readonly ConcurrentBag<Dictionary<ulong, byte>> DictPool = new();
    private static readonly Dictionary<ulong, ulong[]> FactorCache = new();
    private static readonly object FactorCacheLock = new();

    private static List<ulong> RentList() => ListPool.TryTake(out var list) ? list : new List<ulong>(4);

    private static void ReturnList(List<ulong> list)
    {
        list.Clear();
        ListPool.Add(list);
    }

    private static Dictionary<ulong, byte> RentDictionary() => DictPool.TryTake(out var dict) ? dict : new Dictionary<ulong, byte>();

    private static void ReturnDictionary(Dictionary<ulong, byte> dict)
    {
        dict.Clear();
        DictPool.Add(dict);
    }

    private static readonly ConcurrentBag<Dictionary<UInt128, byte>> DictPool128 = new();

    private static Dictionary<UInt128, byte> RentDictionary128() =>
        DictPool128.TryTake(out var dict) ? dict : new Dictionary<UInt128, byte>();

    private static void ReturnDictionary128(Dictionary<UInt128, byte> dict)
    {
        dict.Clear();
        DictPool128.Add(dict);
    }

    private static ulong[] Factor64(ulong n, CancellationToken ct)
    {
        // lock (FactorCacheLock)
        // {
            // if (FactorCache.TryGetValue(n, out var cached))
            // {
            //     return cached;
            // }
        // }

        if (n == 1UL)
        {
            return [];
        }

        var dict = RentDictionary();
        var queue = new Queue<ulong>(); // TODO: Replace this Queue with the pooled stack from the
                                        // FactorizationBenchmarks fast path so factoring large
                                        // composites avoids per-node allocations.
        queue.Enqueue(n);

        while (queue.Count != 0)
        {
            if (ct.IsCancellationRequested)
            {
                break;
            }

            ulong m = queue.Dequeue();
            if (m == 1UL)
            {
                continue;
            }


			// Atomic.Add(ref _factor64Hits, 1UL);
			// Console.WriteLine($"MersennePrimeFactorTester.Factor64 hits {Volatile.Read(ref _factor64Hits)}");

            // if (HeuristicPrimeTester.Exclusive.IsPrime(m, ct))
            if (PrimeTester.IsPrimeCpu(m, ct))
            {
                dict[m] = 1;
                continue;
            }

            if (ct.IsCancellationRequested)
            {
                break;
            }

            ulong d = PollardRho(m, ct);
            if (ct.IsCancellationRequested)
            {
                break;
            }

            queue.Enqueue(d);
            queue.Enqueue(m / d);
        }

        ulong[] result = [.. dict.Keys];
        // TODO: Rent the result array from ArrayPool once the factoring pipeline consumes spans so we
        // can recycle buffers across large factorizations.
        // lock (FactorCacheLock)
        // {
        //     FactorCache[n] = result;
        // }

        ReturnDictionary(dict);
        return result;
    }

    private static UInt128[] Factor128(UInt128 n, CancellationToken ct)
    {
        if (n == 1)
        {
            return [];
        }

        var dict = RentDictionary128();
        var queue = new Queue<UInt128>(); // TODO: Replace this with the pooled UInt128 stack once the
                                          // wide-factorization benchmarks finalize the faster span
                                          // based traversal.
        queue.Enqueue(n);

        while (queue.Count != 0)
        {
            if (ct.IsCancellationRequested)
            {
                break;
            }

            UInt128 m = queue.Dequeue();
            if (m == 1)
            {
                continue;
            }

            if (IsPrime128(m, ct))
            {
                dict[m] = 1;
                continue;
            }

            if (ct.IsCancellationRequested)
            {
                break;
            }

            UInt128 d = PollardRho128(m, ct);
            if (ct.IsCancellationRequested)
            {
                break;
            }

            queue.Enqueue(d);
            queue.Enqueue(m / d);
        }

        var result = new UInt128[dict.Count]; // TODO: Rent this array from ArrayPool<UInt128> so
                                              // wide-factorization batches stop allocating fresh
                                              // buffers per candidate.
        int index = 0;
        foreach (var key in dict.Keys)
        {
            result[index++] = key;
        }

        ReturnDictionary128(dict);
        return result;
    }

    private static ulong PollardRho(ulong n, CancellationToken ct)
    {
        if ((n & 1UL) == 0UL)
        {
            return 2UL;
        }

        ulong c, d, x, y, diff;
        while (true)
        {
            if (ct.IsCancellationRequested)
            {
                return 0UL;
            }

            c = (DeterministicRandom.NextUInt64() % (n - 1UL)) + 1UL;
            x = (DeterministicRandom.NextUInt64() % (n - 2UL)) + 2UL;
            y = x;
            d = 1UL;

            while (d == 1UL)
            {
                if (ct.IsCancellationRequested)
                {
                    return 0UL;
                }

                x = x.MulMod64(x, n).AddMod64(c, n);
                if (ct.IsCancellationRequested)
                {
                    return 0UL;
                }

                y = y.MulMod64(y, n).AddMod64(c, n);
                if (ct.IsCancellationRequested)
                {
                    return 0UL;
                }

                y = y.MulMod64(y, n).AddMod64(c, n);
                diff = x > y ? x - y : y - x;
                if (ct.IsCancellationRequested)
                {
                    return 0UL;
                }

                d = Gcd(diff, n, ct);
            }

            if (d != n)
            {
                return d;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Gcd(ulong a, ulong b, CancellationToken ct)
    {
        ulong t;
        while (b != 0UL)
        {
            if (ct.IsCancellationRequested)
            {
                return 1UL; // We're returning 1 to avoid division by 0 in any place
            }

            t = a % b; // TODO: Replace this with the binary GCD helper used in the gcd benchmarks so we
                        // avoid slow modulo operations while factoring large composites.
            a = b;
            b = t;
        }

        return a;
    }

    // TODO: Port this Miller–Rabin routine to UInt128 intrinsics so we avoid the BigInteger.ModPow calls that lag by orders of magnitude on the large inputs highlighted in the Pow2MontgomeryMod benchmarks.
    private static bool IsPrime128(UInt128 n, CancellationToken ct)
    {
        if (n <= 3)
        {
            return n >= 2;
        }

        if ((n & 1) == 0)
        {
            return false;
        }

        BigInteger bn = (BigInteger)n;
        BigInteger d = bn - 1;
        int s = 0;
        while ((d & 1) == 0)
        {
            d >>= 1;
            s++;
        }

        BigInteger[] bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
        for (int i = 0; i < bases.Length; i++)
        {
            BigInteger a = bases[i];
            if (a >= bn)
            {
                continue;
            }

            if (ct.IsCancellationRequested)
            {
                return false;
            }

            BigInteger x = BigInteger.ModPow(a, d, bn);
            if (x == 1 || x == bn - 1)
            {
                continue;
            }

            bool cont = false;
            for (int r = 1; r < s; r++)
            {
                if (ct.IsCancellationRequested)
                {
                    return false;
                }

                x = BigInteger.ModPow(x, 2, bn);
                if (x == bn - 1)
                {
                    cont = true;
                    break;
                }
            }

            if (!cont)
            {
                return false;
            }
        }

        return true;
    }

    private static UInt128 PollardRho128(UInt128 n, CancellationToken ct)
    {
        if ((n & 1) == 0)
        {
            return 2;
        }

        UInt128 c, d, x, y, diff;
        while (true)
        {
            if (ct.IsCancellationRequested)
            {
                return 0;
            }

            c = (DeterministicRandom.NextUInt128() % (n - 1)) + 1;
            x = (DeterministicRandom.NextUInt128() % (n - 2)) + 2;
            y = x;
            d = 1;

            while (d == 1)
            {
                if (ct.IsCancellationRequested)
                {
                    return 0;
                }

                x = x.MulMod(x, n).AddMod(c, n);
                if (ct.IsCancellationRequested)
                {
                    return 0;
                }

                y = y.MulMod(y, n).AddMod(c, n);
                if (ct.IsCancellationRequested)
                {
                    return 0;
                }

                y = y.MulMod(y, n).AddMod(c, n);
                diff = x > y ? x - y : y - x;
                if (ct.IsCancellationRequested)
                {
                    return 0;
                }

                d = Gcd128(diff, n, ct); // TODO: Move this to the binary GCD implementation once the
                                          // UInt128 benchmarks validate the optimized routine.
            }

            if (d != n)
            {
                return d;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static UInt128 Gcd128(UInt128 a, UInt128 b, CancellationToken ct)
    {
        UInt128 t;
        while (b != 0)
        {
            if (ct.IsCancellationRequested)
            {
                return 1;
            }

            t = a % b; // TODO: Replace with the binary GCD helper once the UInt128 benchmarks land so
                        // we remove the slow `%` from the wide gcd loop.
            a = b;
            b = t;
        }

        return a;
    }
}
