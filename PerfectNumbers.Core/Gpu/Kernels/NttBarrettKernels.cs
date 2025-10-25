using System.Runtime.CompilerServices;
using ILGPU;

namespace PerfectNumbers.Core.Gpu;

internal static class NttBarrettKernels
{
    // Barrett-based scaling: v = v * scale mod n for 128-bit n
    public static void ScaleBarrett128Kernel(Index1D index, ArrayView<GpuUInt128> data, GpuUInt128 scale, ulong modHigh, ulong modLow, ulong muHigh, ulong muLow)
    {
        var v = data[index];
        // 128x128 -> 256
        Mul128Full(v.High, v.Low, scale.High, scale.Low, out var z3, out var z2, out var z1, out var z0);
        // q approx
        Mul3x2Top(z3, z2, z1, muHigh, muLow, out var qHigh, out var qLow);
        // q*n
        Mul128Full(qHigh, qLow, modHigh, modLow, out var qq3, out var qq2, out var qq1, out var qq0);
        // r = z - q*n
        Sub256(z3, z2, z1, z0, qq3, qq2, qq1, qq0, out var r3, out var r2, out var r1, out var r0);
        ulong resHigh = r1;
        ulong resLow = r0;
        if (Geq128(resHigh, resLow, modHigh, modLow))
        {
            Sub128(ref resHigh, ref resLow, modHigh, modLow);
            if (Geq128(resHigh, resLow, modHigh, modLow))
            {
                Sub128(ref resHigh, ref resLow, modHigh, modLow);
            }
        }

        data[index] = new GpuUInt128(resHigh, resLow);
    }

    // Barrett-based squaring: v = v^2 mod n for 128-bit n
    public static void SquareBarrett128Kernel(Index1D index, ArrayView<GpuUInt128> data, ulong modHigh, ulong modLow, ulong muHigh, ulong muLow)
    {
        var v = data[index];
        Mul128Full(v.High, v.Low, v.High, v.Low, out var z3, out var z2, out var z1, out var z0);
        Mul3x2Top(z3, z2, z1, muHigh, muLow, out var qHigh, out var qLow);
        Mul128Full(qHigh, qLow, modHigh, modLow, out var qq3, out var qq2, out var qq1, out var qq0);
        Sub256(z3, z2, z1, z0, qq3, qq2, qq1, qq0, out var r3, out var r2, out var r1, out var r0);
        ulong resHigh = r1;
        ulong resLow = r0;
        if (Geq128(resHigh, resLow, modHigh, modLow))
        {
            Sub128(ref resHigh, ref resLow, modHigh, modLow);
            if (Geq128(resHigh, resLow, modHigh, modLow))
            {
                Sub128(ref resHigh, ref resLow, modHigh, modLow);
            }
        }
        data[index] = new GpuUInt128(resHigh, resLow);
    }

    // Barrett reduction based stage for 128-bit moduli (no '%').
    public static void StageBarrett128Kernel(Index1D index, ArrayView<GpuUInt128> data, int len, int half, int stageOffset, ArrayView<GpuUInt128> twiddles, ulong modHigh, ulong modLow, ulong muHigh, ulong muLow)
    {
        int t = index.X;
        // TODO: Use the bitmask-based remainder helper here as well to remove `%` from the butterfly stage and align with the
        // optimized kernels highlighted in the GPU pow2mod benchmarks.
        int j = t % half;
        int block = t / half;
        int k = block * len;
        int i1 = k + j;
        int i2 = i1 + half;
        var u = data[i1];
        var v = data[i2];
        var w = twiddles[stageOffset + j];

        // Compute v * w (128x128) -> 256-bit z3..z0
        Mul128Full(v.High, v.Low, w.High, w.Low, out var z3, out var z2, out var z1, out var z0);

        // t = floor(z / b) where b=2^64 => take (z3,z2,z1)
        // Compute q = floor((t * mu) / b^3). Only need top two limbs of 5-limb product.
        Mul3x2Top(z3, z2, z1, muHigh, muLow, out var qHigh, out var qLow);

        // q * n (128x128) -> 256-bit qq3..qq0
        Mul128Full(qHigh, qLow, modHigh, modLow, out var qq3, out var qq2, out var qq1, out var qq0);

        // r = z - q*n (256-bit)
        Sub256(z3, z2, z1, z0, qq3, qq2, qq1, qq0, out var r3, out var r2, out var r1, out var r0);

        // Reduce r to [0, n) by at most two subtractions
        // Discard higher limbs (should be zero if r < 2n), keep low 128-bit
        ulong resHigh = r1; // r1 is limb1 after subtraction (since lower two limbs are r1 (high) and r0 (low))
        ulong resLow = r0;

        // while (res >= n) res -= n; (at most twice)
        if (Geq128(resHigh, resLow, modHigh, modLow))
        {
            Sub128(ref resHigh, ref resLow, modHigh, modLow);
            if (Geq128(resHigh, resLow, modHigh, modLow))
            {
                Sub128(ref resHigh, ref resLow, modHigh, modLow);
            }
        }

        var sum = new GpuUInt128(u);
        var mulred = new GpuUInt128(resHigh, resLow);
        sum.AddMod(mulred, new GpuUInt128(modHigh, modLow));
        u.SubMod(mulred, new GpuUInt128(modHigh, modLow));
        data[i1] = sum;
        data[i2] = u;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Mul128Full(ulong aHigh, ulong aLow, ulong bHigh, ulong bLow, out ulong p3, out ulong p2, out ulong p1, out ulong p0)
    {
        // Full 128x128 -> 256 multiply using 64-bit limbs
        Mul64Parts(aLow, bLow, out var h0, out var l0);
        Mul64Parts(aLow, bHigh, out var h1, out var l1);
        Mul64Parts(aHigh, bLow, out var h2, out var l2);
        Mul64Parts(aHigh, bHigh, out var h3, out var l3);

        p0 = l0;
        ulong carry = 0UL;
        ulong sum = h0;
        sum += l1; if (sum < l1) carry++;
        sum += l2; if (sum < l2) carry++;
        p1 = sum;

        sum = h1;
        sum += h2; ulong carry2 = sum < h2 ? 1UL : 0UL;
        sum += l3; if (sum < l3) carry2++;
        sum += carry; if (sum < carry) carry2++;
        p2 = sum;
        p3 = h3 + carry2;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Mul3x2Top(ulong t2, ulong t1, ulong t0, ulong muHigh, ulong muLow, out ulong top1, out ulong top0)
    {
        // Compute top two limbs of ( (t2,t1,t0) * (muHigh,muLow) ) after shifting right by 192 bits.
        // We build full 5 limbs and return p4 (top1) and p3 (top0).
        ulong p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0;

        void Add128To(ref ulong lo, ref ulong hi, ulong addLo, ulong addHi)
        {
            ulong s = lo + addLo;
            ulong c = s < lo ? 1UL : 0UL;
            lo = s;
            ulong s2 = hi + addHi + c;
            hi = s2;
        }

        // t0 * muLow -> contributes to p0,p1
        Mul64Parts(t0, muLow, out var h, out var l);
        Add128To(ref p0, ref p1, l, h);

        // t0 * muHigh -> to p1,p2
        Mul64Parts(t0, muHigh, out h, out l);
        Add128To(ref p1, ref p2, l, h);

        // t1 * muLow -> to p1,p2
        Mul64Parts(t1, muLow, out h, out l);
        Add128To(ref p1, ref p2, l, h);

        // t1 * muHigh -> to p2,p3
        Mul64Parts(t1, muHigh, out h, out l);
        Add128To(ref p2, ref p3, l, h);

        // t2 * muLow -> to p2,p3
        Mul64Parts(t2, muLow, out h, out l);
        Add128To(ref p2, ref p3, l, h);

        // t2 * muHigh -> to p3,p4
        Mul64Parts(t2, muHigh, out h, out l);
        Add128To(ref p3, ref p4, l, h);

        top1 = p4; top0 = p3;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Sub256(ulong a3, ulong a2, ulong a1, ulong a0, ulong b3, ulong b2, ulong b1, ulong b0, out ulong r3, out ulong r2, out ulong r1, out ulong r0)
    {
        ulong borrow = 0;
        r0 = SubWithBorrow(a0, b0, ref borrow);
        r1 = SubWithBorrow(a1, b1, ref borrow);
        r2 = SubWithBorrow(a2, b2, ref borrow);
        r3 = SubWithBorrow(a3, b3, ref borrow);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong SubWithBorrow(ulong a, ulong b, ref ulong borrow)
    {
        ulong res = a - b - borrow;
        borrow = ((a ^ b) & (a ^ res) & 0x8000_0000_0000_0000UL) != 0 ? 1UL : (a < b + borrow ? 1UL : 0UL);
        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool Geq128(ulong aHigh, ulong aLow, ulong bHigh, ulong bLow)
    {
        return aHigh > bHigh || (aHigh == bHigh && aLow >= bLow);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Sub128(ref ulong aHigh, ref ulong aLow, ulong bHigh, ulong bLow)
    {
        ulong borrow = 0;
        ulong low = aLow - bLow;
        borrow = aLow < bLow ? 1UL : 0UL;
        ulong high = aHigh - bHigh - borrow;
        aHigh = high; aLow = low;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Mul64Parts(ulong a, ulong b, out ulong high, out ulong low)
    {
        // 64x64 -> 128 using 32-bit limbs
        ulong a0 = (uint)a;
        ulong a1 = a >> 32;
        ulong b0 = (uint)b;
        ulong b1 = b >> 32;

        ulong lo = a0 * b0;
        ulong mid1 = a1 * b0;
        ulong mid2 = a0 * b1;
        ulong hi = a1 * b1;

        ulong carry = (lo >> 32) + (uint)mid1 + (uint)mid2;
        low = (lo & 0xFFFFFFFFUL) | (carry << 32);
        hi += (mid1 >> 32) + (mid2 >> 32) + (carry >> 32);
        high = hi;
    }
}
