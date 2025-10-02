using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Numerics;
using UInt128 = System.UInt128;

namespace PerfectNumbers.Core;

public static class MersennePrimeFactorTester
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsPrimeFactor(ulong p, ulong q, CancellationToken ct)
    {
        if (q < 2UL || (q & 1UL) == 0)
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

        if (!PrimeTester.IsPrimeInternal(q, ct))
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
                if (ModPow128(2, candidate, q, ct) == 1)
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
    // TODO: Replace this duplicate powmod with ULongExtensions.ModPow64 so we inherit the optimized MulMod64 path that led the MulMod64Benchmarks on large operands.
    private static ulong ModPow64(ulong value, ulong exponent, ulong modulus, CancellationToken ct)
    {
        ulong result = 1UL;
        ulong baseValue = value % modulus;
        ulong exp = exponent;

        while (exp != 0UL)
        {
            if (ct.IsCancellationRequested)
            {
                return 0UL;
            }

            if ((exp & 1UL) != 0UL)
            {
                result = MulMod64(result, baseValue, modulus);
            }

            if (ct.IsCancellationRequested)
            {
                return 0UL;
            }

            baseValue = MulMod64(baseValue, baseValue, modulus);
            exp >>= 1;
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MulMod64(ulong a, ulong b, ulong mod) => (ulong)((UInt128)a * b % mod); // TODO: Route this through ULongExtensions.MulMod64 so the powmod above adopts the inline UInt128 implementation that dominated the MulMod64Benchmarks.

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    // TODO: Switch to UInt128Extensions.ModPow once its MulMod backend adopts the faster UInt128BuiltIn path highlighted in MulHighBenchmarks to avoid the BigInteger fallback.
    private static UInt128 ModPow128(UInt128 value, UInt128 exponent, UInt128 modulus, CancellationToken ct)
    {
        UInt128 result = 1;
        UInt128 baseValue = value % modulus;
        UInt128 exp = exponent;

        while (exp != 0)
        {
            if (ct.IsCancellationRequested)
            {
                return 0;
            }

            if ((exp & 1) != 0)
            {
                result = MulMod128(result, baseValue, modulus);
            }

            if (ct.IsCancellationRequested)
            {
                return 0;
            }

            baseValue = MulMod128(baseValue, baseValue, modulus);
            exp >>= 1;
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static UInt128 MulMod128(UInt128 a, UInt128 b, UInt128 mod)
    {
        BigInteger product = (BigInteger)a * b; // TODO: Replace this BigInteger reduction with the forthcoming UInt128 intrinsic helper measured faster in Mul64Benchmarks and MulHighBenchmarks for huge operands.
        product %= (BigInteger)mod;
        return (UInt128)product;
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

            if (PrimeTester.IsPrimeInternal(m, ct))
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

        var rng = new Random(1234567); // TODO: Replace this RNG with the deterministic span-based
                                       // sequence measured faster in PollardRhoBenchmarks so we avoid
                                       // per-call Random allocations.

        ulong c, d, x, y, diff;
        while (true)
        {
            if (ct.IsCancellationRequested)
            {
                return 0UL;
            }

            c = (ulong)rng.Next(2, int.MaxValue);
            x = (ulong)rng.Next(2, int.MaxValue);
            y = x;
            d = 1UL;

            while (d == 1UL)
            {
                if (ct.IsCancellationRequested)
                {
                    return 0UL;
                }

                x = (MulMod64(x, x, n) + c) % n; // TODO: Swap the `% n` with the Montgomery folding
                                                 // helper from the PollardRho benchmarks once the
                                                 // specialized reducer lands.
                if (ct.IsCancellationRequested)
                {
                    return 0UL;
                }

                y = (MulMod64(y, y, n) + c) % n; // TODO: Use the same Montgomery folding helper here
                                                 // to keep the tortoise sequence on the optimized
                                                 // path.
                if (ct.IsCancellationRequested)
                {
                    return 0UL;
                }

                y = (MulMod64(y, y, n) + c) % n; // TODO: Replace this modulo with the optimized helper
                                                 // so both steps share the fast reduction.
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

        var rng = new Random(1234567); // TODO: Replace with the deterministic UInt128 sequence from
                                       // PollardRho128Benchmarks so we eliminate Random allocations.

        UInt128 c, d, x, y, diff;
        while (true)
        {
            if (ct.IsCancellationRequested)
            {
                return 0;
            }

            c = (UInt128)rng.NextInt64(2, long.MaxValue);
            x = (UInt128)rng.NextInt64(2, long.MaxValue);
            y = x;
            d = 1;

            while (d == 1)
            {
                if (ct.IsCancellationRequested)
                {
                    return 0;
                }

                x = (MulMod128(x, x, n) + c) % n; // TODO: Swap this modulo for the UInt128 Montgomery
                                                 // reducer once the optimized helper lands.
                if (ct.IsCancellationRequested)
                {
                    return 0;
                }

                y = (MulMod128(y, y, n) + c) % n; // TODO: Use the same optimized reducer for the
                                                 // tortoise step.
                if (ct.IsCancellationRequested)
                {
                    return 0;
                }

                y = (MulMod128(y, y, n) + c) % n; // TODO: Replace with the optimized reducer so both
                                                 // steps avoid BigInteger-based modulo.
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
