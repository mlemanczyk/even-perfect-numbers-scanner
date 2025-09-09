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
        if ((q - 1UL) % value != 0UL)
        {
            return false;
        }

        ulong order = GetOrderOf2ModPrime(q, ct);
        return !ct.IsCancellationRequested && order == p;
    }

    private static readonly Dictionary<UInt128, UInt128> _orderCache = [];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ulong GetOrderOf2ModPrime(ulong q, CancellationToken ct) =>
        (ulong)GetOrderOf2ModPrime((UInt128)q, ct);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static UInt128 GetOrderOf2ModPrime(UInt128 q, CancellationToken ct) =>
        OrderOf2ModPrime(q, ct);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static UInt128 OrderOf2ModPrime(UInt128 q, CancellationToken ct)
    {
        lock (_orderCache)
        {
            if (_orderCache.TryGetValue(q, out var cached))
            {
                return cached;
            }
        }

        UInt128 phi = q - 1;
        UInt128[] factors = phi <= ulong.MaxValue
            ? Array.ConvertAll(Factor64((ulong)phi, ct), x => (UInt128)x)
            : Factor128(phi, ct);
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
    private static ulong MulMod64(ulong a, ulong b, ulong mod) => (ulong)((UInt128)a * b % mod);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
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
        BigInteger product = (BigInteger)a * b;
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
        var queue = new Queue<ulong>();
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
        var queue = new Queue<UInt128>();
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

        var result = new UInt128[dict.Count];
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

        var rng = new Random(1234567);

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

                x = (MulMod64(x, x, n) + c) % n;
                if (ct.IsCancellationRequested)
                {
                    return 0UL;
                }

                y = (MulMod64(y, y, n) + c) % n;
                if (ct.IsCancellationRequested)
                {
                    return 0UL;
                }

                y = (MulMod64(y, y, n) + c) % n;
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

            t = a % b;
            a = b;
            b = t;
        }

        return a;
    }

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

        var rng = new Random(1234567);

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

                x = (MulMod128(x, x, n) + c) % n;
                if (ct.IsCancellationRequested)
                {
                    return 0;
                }

                y = (MulMod128(y, y, n) + c) % n;
                if (ct.IsCancellationRequested)
                {
                    return 0;
                }

                y = (MulMod128(y, y, n) + c) % n;
                diff = x > y ? x - y : y - x;
                if (ct.IsCancellationRequested)
                {
                    return 0;
                }

                d = Gcd128(diff, n, ct);
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

            t = a % b;
            a = b;
            b = t;
        }

        return a;
    }
}
