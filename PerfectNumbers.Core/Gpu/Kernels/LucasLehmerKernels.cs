using System;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using UInt128 = System.UInt128;

namespace PerfectNumbers.Core.Gpu;

internal static class LucasLehmerKernels
{
    public static void AddSmallKernel(Index1D index, ulong add, ArrayView<GpuUInt128> value)
    {
        ulong carry = add;
        for (int i = 0; i < value.Length && carry != 0UL; i++)
        {
            var limb = value[i];
            ulong low = limb.Low + carry;
            ulong carryOut = low < carry ? 1UL : 0UL;
            ulong high = limb.High + carryOut;
            carry = high < limb.High ? 1UL : 0UL;
            value[i] = new GpuUInt128(high, low);
        }
    }

    public static void SubtractSmallKernel(Index1D index, ulong subtract, ArrayView<GpuUInt128> value)
    {
        ulong borrow = subtract;
        for (int i = 0; i < value.Length && borrow != 0UL; i++)
        {
            var limb = value[i];
            ulong low = limb.Low;
            ulong high = limb.High;
            if (low >= borrow)
            {
                low -= borrow;
                borrow = 0UL;
            }
            else
            {
                low = unchecked(low - borrow);
                borrow = 1UL;
                if (high != 0UL)
                {
                    high--;
                    borrow = 0UL;
                }
                else
                {
                    high = ulong.MaxValue;
                }
            }

            value[i] = new GpuUInt128(high, low);
        }
    }

    public static void ReduceModMersenneKernel(Index1D index, ulong exponent, ArrayView<GpuUInt128> value)
    {
        int limbCount = (int)((exponent + 127UL) / 128UL);
        for (int i = limbCount; i < value.Length; i++)
        {
            var limb = value[i];
            if (limb.IsZero)
            {
                continue;
            }

            int target = i - limbCount;
            var sum = value[target];
            var original = sum;
            sum.Add(limb);
            value[target] = sum;
            value[i] = new GpuUInt128(0UL, 0UL);
            if (sum.CompareTo(original) < 0)
            {
                int j = target + 1;
                while (j < limbCount)
                {
                    var next = value[j];
                    var nextOriginal = next;
                    next.Add(1UL);
                    value[j] = next;
                    if (next.CompareTo(nextOriginal) >= 0)
                    {
                        break;
                    }

                    j++;
                }
            }
        }

        int topBits = (int)exponent.Mod128();
        if (topBits == 0)
        {
            topBits = 128;
        }

        int topIndex = limbCount - 1;
        ulong topLowMask = topBits >= 64 ? ulong.MaxValue : (1UL << topBits) - 1UL;
        ulong topHighMask = topBits > 64 ? (1UL << (topBits - 64)) - 1UL : 0UL;
        var top = value[topIndex];
        var carryBits = top >> topBits;
        top = new(top.High & topHighMask, top.Low & topLowMask);
        value[topIndex] = top;

        if (!carryBits.IsZero)
        {
            int j = 0;
            while (j < limbCount && !carryBits.IsZero)
            {
                var cur = value[j];
                var curOriginal = cur;
                cur.Add(carryBits);
                value[j] = cur;
                carryBits = cur.CompareTo(curOriginal) < 0 ? new GpuUInt128(0UL, 1UL) : new GpuUInt128(0UL, 0UL);
                j++;
            }
        }

        bool geq = true;
        for (int i = topIndex; i >= 0; i--)
        {
            var limb = value[i];
            ulong modHigh = i == topIndex ? topHighMask : ulong.MaxValue;
            ulong modLow = i == topIndex ? topLowMask : ulong.MaxValue;
            if (limb.High > modHigh || (limb.High == modHigh && limb.Low > modLow))
            {
                geq = true;
                break;
            }

            if (limb.High < modHigh || (limb.High == modHigh && limb.Low < modLow))
            {
                geq = false;
                break;
            }
        }

        if (geq)
        {
            ulong borrow = 0UL;
            for (int i = 0; i < limbCount - 1; i++)
            {
                var limb = value[i];
                ulong newLow = limb.Low - ulong.MaxValue - borrow;
                borrow = limb.Low < ulong.MaxValue + borrow ? 1UL : 0UL;
                ulong newHigh = limb.High - ulong.MaxValue - borrow;
                borrow = limb.High < ulong.MaxValue + borrow ? 1UL : 0UL;
                value[i] = new GpuUInt128(newHigh, newLow);
            }

            var topLimb = value[topIndex];
            ulong newTopLow = topLimb.Low - topLowMask - borrow;
            borrow = topLimb.Low < topLowMask + borrow ? 1UL : 0UL;
            ulong newTopHigh = topLimb.High - topHighMask - borrow;
            value[topIndex] = new GpuUInt128(newTopHigh, newTopLow);
        }

        for (int i = limbCount; i < value.Length; i++)
        {
            value[i] = new GpuUInt128(0UL, 0UL);
        }
    }

    public static void IsZeroKernel(Index1D index, ArrayView<GpuUInt128> value, ArrayView<byte> result)
    {
        byte isZero = 1;
        for (int i = 0; i < value.Length; i++)
        {
            var limb = value[i];
            if (limb.High != 0UL || limb.Low != 0UL)
            {
                isZero = 0;
                break;
            }
        }

        result[0] = isZero;
    }

    private static void ReduceModMersenne(Span<GpuUInt128> value, ulong exponent)
    {
        int limbCount = (int)((exponent + 127UL) / 128UL);
        for (int i = limbCount; i < value.Length; i++)
        {
            var limb = value[i];
            if (limb.IsZero)
            {
                continue;
            }

            int target = i - limbCount;
            UInt128 original = value[target];
            UInt128 sum = original + (UInt128)limb;
            bool carry = sum < original;
            value[target] = new GpuUInt128(sum);
            value[i] = new GpuUInt128(0UL, 0UL);
            if (carry)
            {
                int j = target + 1;
                while (j < limbCount)
                {
                    UInt128 next = value[j];
                    UInt128 nextSum = next + 1UL;
                    value[j] = new GpuUInt128(nextSum);
                    if (nextSum != 0UL)
                    {
                        break;
                    }

                    j++;
                }
            }
        }

        int topBits = (int)exponent.Mod128();
        if (topBits == 0)
        {
            topBits = 128;
        }

        int topIndex = limbCount - 1;
        ulong topLowMask = topBits >= 64 ? ulong.MaxValue : (1UL << topBits) - 1UL;
        ulong topHighMask = topBits > 64 ? (1UL << (topBits - 64)) - 1UL : 0UL;
        UInt128 mask = ((UInt128)topHighMask << 64) | topLowMask;
        UInt128 top = value[topIndex];
        UInt128 carryBits = top >> topBits;
        top &= mask;
        value[topIndex] = new GpuUInt128(top);
        if (carryBits != 0UL)
        {
            int j = 0;
            UInt128 carry = carryBits;
            while (j < limbCount && carry != 0UL)
            {
                UInt128 cur = value[j];
                UInt128 sum = cur + carry;
                bool overflow = sum < cur;
                value[j] = new GpuUInt128(sum);
                carry = overflow ? 1UL : 0UL;
                j++;
            }
        }

        bool geq = true;
        for (int i = topIndex; i >= 0; i--)
        {
            UInt128 limb = value[i];
            UInt128 modLimb = i == topIndex ? mask : UInt128.MaxValue;
            if (limb > modLimb)
            {
                geq = true;
                break;
            }

            if (limb < modLimb)
            {
                geq = false;
                break;
            }
        }

        if (geq)
        {
            ulong borrow = 0UL;
            for (int i = 0; i < limbCount - 1; i++)
            {
                ulong low = value[i].Low;
                ulong high = value[i].High;
                ulong newLow = low - ulong.MaxValue - borrow;
                borrow = low < ulong.MaxValue + borrow ? 1UL : 0UL;
                ulong newHigh = high - ulong.MaxValue - borrow;
                borrow = high < ulong.MaxValue + borrow ? 1UL : 0UL;
                value[i] = new GpuUInt128(newHigh, newLow);
            }

            ulong topLow = value[topIndex].Low;
            ulong topHigh = value[topIndex].High;
            ulong newTopLow = topLow - topLowMask - borrow;
            borrow = topLow < topLowMask + borrow ? 1UL : 0UL;
            ulong newTopHigh = topHigh - topHighMask - borrow;
            value[topIndex] = new GpuUInt128(newTopHigh, newTopLow);
        }

        for (int i = limbCount; i < value.Length; i++)
        {
            value[i] = new GpuUInt128(0UL, 0UL);
        }
    }

    public static void KernelBatch(Index1D index, ArrayView<ulong> exponents, ArrayView<GpuUInt128> moduli, ArrayView<GpuUInt128> states)
    {
        int idx = index.X;
        ulong exponent = exponents[idx];
        GpuUInt128 modulus = moduli[idx];
        var s = new GpuUInt128(0UL, 4UL);
        ulong limit = exponent - 2UL;
        for (ulong i = 0UL; i < limit; i++)
        {
            s = SquareModMersenne128(s, exponent);
            s.SubMod(2UL, modulus);
        }

        states[idx] = s;
    }

    public static void Kernel(Index1D index, ulong exponent, GpuUInt128 modulus, ArrayView<GpuUInt128> state)
    {
        // Lucas–Lehmer iteration in the field GF(2^p-1).
        // For p < 128, use a Mersenne-specific squaring + reduction to avoid
        // the generic 128-bit long-division-like reduction.
        var s = new GpuUInt128(0UL, 4UL);
        ulong i = 0UL, limit = exponent - 2UL;
        if (exponent < 128UL)
        {
            for (; i < limit; i++)
            {
                s = SquareModMersenne128(s, exponent);
                s.SubMod(2UL, modulus);
            }
        }
        else
        {
            for (; i < limit; i++)
            {
                s.SquareMod(modulus);
                s.SubMod(2UL, modulus);
            }
        }

        state[0] = s;
    }

    // Squares a 128-bit value and reduces modulo M_p = 2^p - 1 using
    // Mersenne folding. Valid for 0 < p < 128 and input s in [0, M_p].
    private static GpuUInt128 SquareModMersenne128(GpuUInt128 s, ulong p)
    {
        // Full square: (H*2^64 + L)^2 = H^2*2^128 + 2HL*2^64 + L^2
        SquareFull128(s, out var q3, out var q2, out var q1, out var q0);

        // v = low128 + high128 (fold by 128 bits once)
        var vHigh = new GpuUInt128(q3, q2);
        var vLow = new GpuUInt128(q1, q0);
        vLow.Add(vHigh);

        // Mask to keep only p low bits and fold the carry bits (v >> p)
        int topBits = (int)p.Mod128();
        if (topBits == 0)
        {
            topBits = 128;
        }

        ulong maskLow = topBits >= 64 ? ulong.MaxValue : (1UL << topBits) - 1UL;
        ulong maskHigh = topBits > 64 ? (1UL << (topBits - 64)) - 1UL : 0UL;

        var carryBits = vLow >> topBits;
		vLow = new(vLow.High & maskHigh, vLow.Low & maskLow);

		// Propagate carryBits back into the masked value (single-limb fold)
		GpuUInt128 before;
        while (!carryBits.IsZero)
		{
			before = vLow;
                    vLow.Add(carryBits);
			carryBits = vLow.CompareTo(before) < 0 ? GpuUInt128.One : GpuUInt128.Zero;
		}

        // Final correction if v >= modulus (mask)
        if (vLow.High > maskHigh || (vLow.High == maskHigh && vLow.Low >= maskLow))
        {
            ulong borrow = 0UL;
            ulong newLow = vLow.Low - maskLow;
            borrow = vLow.Low < maskLow ? 1UL : 0UL;
            ulong newHigh = vLow.High - maskHigh - borrow;
            vLow = new GpuUInt128(newHigh, newLow);
        }

        return vLow;
    }

    // Helper: 128-bit square -> 256-bit product as four 64-bit limbs (q3..q0)
    private static void SquareFull128(GpuUInt128 value, out ulong q3, out ulong q2, out ulong q1, out ulong q0)
    {
        ulong L = value.Low;
        ulong H = value.High;
        Mul64Parts(L, L, out var hLL, out var lLL);    // L^2
        Mul64Parts(H, H, out var hHH, out var lHH);    // H^2
        Mul64Parts(L, H, out var hLH, out var lLH);    // L*H

        // double LH
        ulong dLH_low = lLH << 1;
        ulong carry = (lLH >> 63) & 1UL;
        ulong dLH_high = (hLH << 1) | carry;

        q0 = lLL;
        ulong sum1 = hLL + dLH_low;
        ulong c1 = sum1 < hLL ? 1UL : 0UL;
        q1 = sum1;

        ulong sum2 = lHH + dLH_high;
        ulong c2 = sum2 < lHH ? 1UL : 0UL;
        sum2 += c1;
        if (sum2 < c1)
        {
            c2++;
        }
        q2 = sum2;
        q3 = hHH + c2;
    }

    // 64x64 -> 128 multiply (high,low)
    [MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static void Mul64Parts(ulong a, ulong b, out ulong high, out ulong low)
    {
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

    [MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static ulong ModPow64(ulong value, ulong exponent, ulong modulus)
    {
        ulong result = 1UL;
        ulong baseValue = value % modulus;
        ulong exp = exponent;

        while (exp != 0UL)
        {
            if ((exp & 1UL) != 0UL)
            {
                result = (ulong)(((UInt128)result * baseValue) % modulus);
            }

            baseValue = (ulong)(((UInt128)baseValue * baseValue) % modulus);
            exp >>= 1;
        }

        return result;
    }
}
