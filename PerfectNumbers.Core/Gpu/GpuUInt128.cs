using System.Numerics;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core.Gpu;

public struct GpuUInt128 : IComparable<GpuUInt128>, IEquatable<GpuUInt128>
{
    public ulong High;

    public ulong Low;

    private const int NativeModuloChunkBits = 8;
    private const int NativeModuloChunkBitsMinusOne = NativeModuloChunkBits - 1;
    private const int NativeModuloBitMaskTableSize = 1024;
    private const ulong NativeModuloChunkMask = (1UL << NativeModuloChunkBits) - 1UL;
    private static readonly ulong[] NativeModuloBitMasks = CreateNativeModuloBitMasks();

    public static readonly GpuUInt128 Zero = new(0UL, 0UL);
    public static readonly GpuUInt128 One = new(0UL, 1UL);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuUInt128(ulong high, ulong low)
    {
        High = high;
        Low = low;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuUInt128(ulong low)
    {
        Low = low;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuUInt128(UInt128 value)
    {
        High = (ulong)(value >> 64);
        Low = (ulong)value;
    }

    public readonly bool IsZero
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => High == 0UL && Low == 0UL;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator GpuUInt128(UInt128 value) => new(value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator GpuUInt128(ulong value) => new(value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator UInt128(GpuUInt128 value) =>
        ((UInt128)value.High << 64) | value.Low;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuUInt128 operator +(GpuUInt128 left, GpuUInt128 right)
    {
        var low = left.Low + right.Low;
        var high = left.High + right.High + (low < left.Low ? 1UL : 0UL);
        return new(high, low);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuUInt128 operator -(GpuUInt128 left, GpuUInt128 right)
    {
        var borrow = left.Low < right.Low ? 1UL : 0UL;
        var low = left.Low - right.Low;
        var high = left.High - right.High - borrow;
        return new(high, low);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuUInt128 operator ^(GpuUInt128 left, GpuUInt128 right) =>
        new(left.High ^ right.High, left.Low ^ right.Low);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuUInt128 operator *(GpuUInt128 left, GpuUInt128 right)
    {
        Multiply(left, right, out var high, out var low);
        return new(high, low);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuUInt128 operator <<(GpuUInt128 value, int shift)
    {
        shift &= 127;
        if (shift == 0)
        {
            return value;
        }

        if (shift >= 64)
        {
            ulong high = value.Low << (shift - 64);
            return new(high, 0UL);
        }

        ulong highPart = (value.High << shift) | (value.Low >> (64 - shift));
        ulong lowPart = value.Low << shift;
        return new(highPart, lowPart);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuUInt128 operator >>(GpuUInt128 value, int shift)
    {
        shift &= 127;
        if (shift == 0)
        {
            return value;
        }

        if (shift >= 64)
        {
            ulong low = value.High >> (shift - 64);
            return new(0UL, low);
        }

        ulong highPart = value.High >> shift;
        ulong lowPart = (value.Low >> shift) | (value.High << (64 - shift));
        return new(highPart, lowPart);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool operator ==(GpuUInt128 left, GpuUInt128 right) => left.Equals(right);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool operator !=(GpuUInt128 left, GpuUInt128 right) => !left.Equals(right);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool operator <(GpuUInt128 left, GpuUInt128 right) => left.CompareTo(right) < 0;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool operator >(GpuUInt128 left, GpuUInt128 right) => left.CompareTo(right) > 0;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool operator <=(GpuUInt128 left, GpuUInt128 right) => left.CompareTo(right) <= 0;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool operator >=(GpuUInt128 left, GpuUInt128 right) => left.CompareTo(right) >= 0;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static GpuUInt128 operator ++(GpuUInt128 value)
        {
            value.Add(1UL);
            return value;
        }

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly int CompareTo(GpuUInt128 other)
    {
        if (High < other.High)
        {
            return -1;
        }

        if (High > other.High)
        {
            return 1;
        }

        if (Low < other.Low)
        {
            return -1;
        }

        if (Low > other.Low)
        {
            return 1;
        }

        return 0;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly bool Equals(GpuUInt128 other) => High == other.High && Low == other.Low;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override readonly bool Equals(object? obj) => obj is GpuUInt128 value && Equals(value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override readonly int GetHashCode() => HashCode.Combine(High, Low);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Add(GpuUInt128 other)
    {
        ulong low = Low + other.Low;
        High = High + other.High + (low < Low ? 1UL : 0UL);
        Low = low;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Add(ulong value)
    {
        ulong low = Low + value;
        High = High + (low < Low ? 1UL : 0UL);
        Low = low;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void AddMod(GpuUInt128 value, GpuUInt128 modulus)
    {
        if (modulus.High == 0UL)
        {
            AddMod(value, modulus.Low);
            return;
        }

        Add(value);
        if (CompareTo(modulus) >= 0)
        {
            Sub(modulus);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void AddMod(ulong value, GpuUInt128 modulus)
    {
        if (modulus.High == 0UL)
        {
            AddMod(value, modulus.Low);
            return;
        }

        Add(value);
        if (CompareTo(modulus) >= 0)
        {
            Sub(modulus);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuUInt128 Pow2Minus1Mod(ulong exponent, GpuUInt128 modulus)
    {
        if (modulus.IsZero)
        {
            return Zero;
        }

        GpuUInt128 result = Zero;
        ulong i = 0UL;
        while (i < exponent)
        {
            // result = (result * 2) % modulus
            result += result;
            if (result.CompareTo(modulus) >= 0)
            {
                result -= modulus;
            }

            // result = (result + 1) % modulus
            result += One;
            if (result.CompareTo(modulus) >= 0)
            {
                result -= modulus;
            }

            i++;
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Pow2Minus1ModBatch(GpuUInt128 modulus, ReadOnlySpan<ulong> exponents, Span<GpuUInt128> results)
    {
        GpuUInt128 r0 = Zero, r1 = Zero, r2 = Zero, r3 = Zero, r4 = Zero, r5 = Zero, r6 = Zero, r7 = Zero;
        ulong e0 = 0UL, e1 = 0UL, e2 = 0UL, e3 = 0UL, e4 = 0UL, e5 = 0UL, e6 = 0UL, e7 = 0UL;
        bool a0 = false, a1 = false, a2 = false, a3 = false, a4 = false, a5 = false, a6 = false, a7 = false;

        int len = exponents.Length;
        if (len > 0) { e0 = exponents[0]; a0 = true; }
        if (len > 1) { e1 = exponents[1]; a1 = true; }
        if (len > 2) { e2 = exponents[2]; a2 = true; }
        if (len > 3) { e3 = exponents[3]; a3 = true; }
        if (len > 4) { e4 = exponents[4]; a4 = true; }
        if (len > 5) { e5 = exponents[5]; a5 = true; }
        if (len > 6) { e6 = exponents[6]; a6 = true; }
        if (len > 7) { e7 = exponents[7]; a7 = true; }

        ulong max = 0UL;
        if (a0 && e0 > max) max = e0;
        if (a1 && e1 > max) max = e1;
        if (a2 && e2 > max) max = e2;
        if (a3 && e3 > max) max = e3;
        if (a4 && e4 > max) max = e4;
        if (a5 && e5 > max) max = e5;
        if (a6 && e6 > max) max = e6;
        if (a7 && e7 > max) max = e7;

        for (ulong i = 0UL; i < max; i++)
        {
            if (a0)
            {
                r0 += r0; if (r0.CompareTo(modulus) >= 0) r0 -= modulus;
                r0 += One; if (r0.CompareTo(modulus) >= 0) r0 -= modulus;
                if (i + 1UL == e0) a0 = false;
            }
            if (a1)
            {
                r1 += r1; if (r1.CompareTo(modulus) >= 0) r1 -= modulus;
                r1 += One; if (r1.CompareTo(modulus) >= 0) r1 -= modulus;
                if (i + 1UL == e1) a1 = false;
            }
            if (a2)
            {
                r2 += r2; if (r2.CompareTo(modulus) >= 0) r2 -= modulus;
                r2 += One; if (r2.CompareTo(modulus) >= 0) r2 -= modulus;
                if (i + 1UL == e2) a2 = false;
            }
            if (a3)
            {
                r3 += r3; if (r3.CompareTo(modulus) >= 0) r3 -= modulus;
                r3 += One; if (r3.CompareTo(modulus) >= 0) r3 -= modulus;
                if (i + 1UL == e3) a3 = false;
            }
            if (a4)
            {
                r4 += r4; if (r4.CompareTo(modulus) >= 0) r4 -= modulus;
                r4 += One; if (r4.CompareTo(modulus) >= 0) r4 -= modulus;
                if (i + 1UL == e4) a4 = false;
            }
            if (a5)
            {
                r5 += r5; if (r5.CompareTo(modulus) >= 0) r5 -= modulus;
                r5 += One; if (r5.CompareTo(modulus) >= 0) r5 -= modulus;
                if (i + 1UL == e5) a5 = false;
            }
            if (a6)
            {
                r6 += r6; if (r6.CompareTo(modulus) >= 0) r6 -= modulus;
                r6 += One; if (r6.CompareTo(modulus) >= 0) r6 -= modulus;
                if (i + 1UL == e6) a6 = false;
            }
            if (a7)
            {
                r7 += r7; if (r7.CompareTo(modulus) >= 0) r7 -= modulus;
                r7 += One; if (r7.CompareTo(modulus) >= 0) r7 -= modulus;
                if (i + 1UL == e7) a7 = false;
            }
        }

        if (len > 0) results[0] = r0;
        if (len > 1) results[1] = r1;
        if (len > 2) results[2] = r2;
        if (len > 3) results[3] = r3;
        if (len > 4) results[4] = r4;
        if (len > 5) results[5] = r5;
        if (len > 6) results[6] = r6;
        if (len > 7) results[7] = r7;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void AddMod(GpuUInt128 value, ulong modulus)
    {
        ulong a = Low % modulus;
        ulong b = value.Low % modulus;
        ulong sum = a + b;
        if (sum >= modulus || sum < a)
        {
            sum -= modulus;
        }

        High = 0UL;
        Low = sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void AddMod(ulong value, ulong modulus)
    {
        ulong a = Low % modulus;
        ulong b = value % modulus;
        ulong sum = a + b;
        if (sum >= modulus || sum < a)
        {
            sum -= modulus;
        }

        High = 0UL;
        Low = sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Sub(GpuUInt128 other)
    {
        ulong borrow = Low < other.Low ? 1UL : 0UL;
        High = High - other.High - borrow;
        Low -= other.Low;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong MulHigh(ulong x, ulong y)
    {
        ulong xLow = (uint)x;
        ulong xHigh = x >> 32;
        ulong yLow = (uint)y;
        ulong yHigh = y >> 32;

        ulong w1 = xLow * yHigh;
        ulong w2 = xHigh * yLow;
        ulong w3 = xLow * yLow;

        // Keeping the wide partial sum in a dedicated local prevents the JIT from
        // materialising it on the stack before the final carry propagation. The
        // additional store looks redundant in C#, but it shortens the generated
        // instruction sequence by avoiding an extra temporary and results in a
        // measurable throughput win in the MulHigh benchmarks.
        ulong result = (xHigh * yHigh) + (w1 >> 32) + (w2 >> 32);
        result += ((w3 >> 32) + (uint)w1 + (uint)w2) >> 32;
        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Mul64(GpuUInt128 other)
    {
        // Multiply this.Low (assumed 64-bit value) by full 128-bit other
        ulong operand = Low;
            //   otherLow = other.Low;

        Low = operand * other.Low;
        High = operand * other.High + MulHigh(operand, other.Low);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuUInt128 AddMod(GpuUInt128 a, GpuUInt128 b, GpuUInt128 modulus)
    {
        a.Add(b);
        if (a >= modulus)
        {
            a.Sub(modulus);
        }

        return a;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuUInt128 MulMod(GpuUInt128 a, GpuUInt128 b, GpuUInt128 modulus)
    {
        GpuUInt128 result = new(0UL, 0UL);
        while (!b.IsZero)
        {
            if ((b.Low & 1UL) != 0UL)
            {
                result = AddMod(result, a, modulus);
            }

            // a = (a << 1) % modulus
            a <<= 1;
            if (a >= modulus)
            {
                a.Sub(modulus);
            }

            // b >>= 1
            if (b.High == 0UL)
            {
                b = new GpuUInt128(0UL, b.Low >> 1);
            }
            else
            {
                ulong low = (b.Low >> 1) | (b.High << 63);
                ulong high = b.High >> 1;
                b = new GpuUInt128(high, low);
            }
        }

        return result;
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuUInt128 Pow2Mod(ulong exponent, GpuUInt128 modulus)
    {
        if (modulus.IsZero || modulus == One)
        {
            return new(0UL, 0UL);
        }

        GpuUInt128 result = new(0UL, 1UL);
		GpuUInt128 baseVal = new(0UL, 2UL);

        ulong e = exponent;
        while (e != 0UL)
        {
            if ((e & 1UL) != 0UL)
            {
                result = MulMod(result, baseVal, modulus);
            }

            baseVal = MulMod(baseVal, baseVal, modulus);
            e >>= 1;
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int TrailingZeroCount(GpuUInt128 value)
    {
        if (value.Low != 0UL)
        {
            return BitOperations.TrailingZeroCount(value.Low);
        }

        return 64 + BitOperations.TrailingZeroCount(value.High);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuUInt128 BinaryGcd(GpuUInt128 u, GpuUInt128 v)
    {
        if (u.IsZero)
        {
            return v;
        }

        if (v.IsZero)
        {
            return u;
        }

        int shift = TrailingZeroCount(new GpuUInt128(u.High | v.High, u.Low | v.Low));
        // remove factors of 2
        int zu = TrailingZeroCount(u);
        u >>= zu;

        do
        {
            int zv = TrailingZeroCount(v);
            v >>= zv;
            if (u > v)
            {
                (u, v) = (v, u);
            }

            v -= u;
        }
        while (!v.IsZero);

        return u << shift;
    }


    public void Sub(ulong value)
    {
        ulong borrow = Low < value ? 1UL : 0UL;
        High -= borrow;
        Low -= value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void SubMod(GpuUInt128 value, GpuUInt128 modulus)
    {
        if (CompareTo(value) >= 0)
        {
            Sub(value);
        }
        else
        {
            Add(modulus);
            Sub(value);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void SubMod(ulong value, GpuUInt128 modulus)
    {
        ulong low, high;
        if (High == 0UL && Low < value)
        {
            low = Low + modulus.Low;
            ulong carry = low < Low ? 1UL : 0UL;
            high = High + modulus.High + carry;
        }
        else
        {
            high = High;
            low = Low;
        }

        ulong borrow = low < value ? 1UL : 0UL;
        High = high - borrow;
        Low = low - value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Mul(GpuUInt128 other)
    {
        Multiply(this, other, out var high, out var low);
        High = high;
        Low = low;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Mul(ulong value)
    {
        var (highPart, lowPart) = Mul64(Low, value);
        High = highPart + High * value;
        Low = lowPart;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Xor(GpuUInt128 other)
    {
        High ^= other.High;
        Low ^= other.Low;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Xor(ulong value)
    {
        Low ^= value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ShiftLeft(int shift)
    {
        this = this << shift;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ShiftRight(int shift)
    {
        this = this >> shift;
    }

    public void MulMod(GpuUInt128 other, GpuUInt128 modulus)
    {
        if (modulus.High == 0UL)
        {
            MulMod(other, modulus.Low);
            return;
        }

        MultiplyFull(this, other, out var p3, out var p2, out var p1, out var p0);

        GpuUInt128 remainder = new(0UL, 0UL);
        int bit;
        for (bit = 63; bit >= 0; bit--)
        {
            remainder <<= 1;
            remainder = new(remainder.High, remainder.Low | ((p3 >> bit) & 1UL));
            if (remainder.CompareTo(modulus) >= 0)
            {
                remainder.Sub(modulus);
            }
        }

        for (bit = 63; bit >= 0; bit--)
        {
            remainder <<= 1;
            remainder = new(remainder.High, remainder.Low | ((p2 >> bit) & 1UL));
            if (remainder.CompareTo(modulus) >= 0)
            {
                remainder.Sub(modulus);
            }
        }

        for (bit = 63; bit >= 0; bit--)
        {
            remainder <<= 1;
            remainder = new(remainder.High, remainder.Low | ((p1 >> bit) & 1UL));
            if (remainder.CompareTo(modulus) >= 0)
            {
                remainder.Sub(modulus);
            }
        }

        for (bit = 63; bit >= 0; bit--)
        {
            remainder <<= 1;
            remainder = new(remainder.High, remainder.Low | ((p0 >> bit) & 1UL));
            if (remainder.CompareTo(modulus) >= 0)
            {
                remainder.Sub(modulus);
            }
        }

        High = remainder.High;
        Low = remainder.Low;
        // TODO(MOD-OPT): Replace this bitwise long-division style reduction
        // with a faster algorithm suitable for GPU kernels without '%' support.
        // Options:
        // - Montgomery reduction for 128-bit moduli (R=2^128), requires pre-
        //   computed modulus-dependent constants (n' and R^2 mod n). This is
        //   ideal for NTT primes (odd modulus). Cache per-modulus constants.
        // - Barrett reduction with a 256/128 quotient approximation using only
        //   multiplies and shifts. Cache mu = floor(2^k / n) for k=256.
        // Implement a fast path for common 64-bit NTT moduli and a separate
        // path for 128-bit moduli. This will significantly reduce the cost of
        // each butterfly in NTT and LL steps.
        // TODO(MOD-OPT): Plumb constants through caches in NttGpuMath.SquareCacheEntry
        // (e.g., Montgomery n', R2) and provide device-friendly accessors.
    }

    /// <summary>
    /// Modular multiplication using <see cref="BigInteger"/> reduction. This
    /// method is intended for validation only and should not be used inside
    /// GPU kernels.
    /// </summary>
    internal void MulModBigInteger(GpuUInt128 other, GpuUInt128 modulus)
    {
        var left = (BigInteger)(UInt128)this;
        var right = (BigInteger)(UInt128)other;
        var mod = (BigInteger)(UInt128)modulus;
        var reduced = (UInt128)((left * right) % mod);
        High = (ulong)(reduced >> 64);
        Low = (ulong)reduced;
    }

    /// <summary>
    /// Experimental limb-based reduction. The current implementation performs
    /// repeated subtractions and becomes extremely slow for large remainders.
    /// Kept for future optimization work.
    /// </summary>
    internal void MulModByLimb(GpuUInt128 other, GpuUInt128 modulus)
    {
        MultiplyFull(this, other, out var p3, out var p2, out var p1, out var p0);

        GpuUInt128 remainder = new(p3, p2);
        while (remainder.CompareTo(modulus) >= 0)
        {
            remainder.Sub(modulus);
        }

        ulong limb = p1;
        for (int i = 0; i < 2; i++)
        {
            remainder <<= 64;
            remainder = new(remainder.High, limb);
            while (remainder.CompareTo(modulus) >= 0)
            {
                remainder.Sub(modulus);
            }

            limb = p0;
        }

        High = remainder.High;
        Low = remainder.Low;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void MulMod(ulong value, GpuUInt128 modulus)
    {
        if (modulus.High == 0UL)
        {
            MulMod(value, modulus.Low);
            return;
        }

        GpuUInt128 other = new(0UL, value);
        MulMod(other, modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void MulMod(GpuUInt128 other, ulong modulus)
    {
        ulong a = Low % modulus;
        ulong b = other.Low % modulus;
        ulong result;
        if (a == 0UL || b == 0UL)
        {
            result = 0UL;
        }
        else if (b <= ulong.MaxValue / a)
        {
            result = (a * b) % modulus;
        }
        else
        {
            result = MulMod64(a, b, modulus);
        }

        High = 0UL;
        Low = result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void MulMod(ulong value, ulong modulus)
    {
        ulong a = Low % modulus;
        ulong b = value % modulus;
        ulong result;
        if (a == 0UL || b == 0UL)
        {
            result = 0UL;
        }
        else if (b <= ulong.MaxValue / a)
        {
            result = (a * b) % modulus;
        }
        else
        {
            result = MulMod64(a, b, modulus);
        }

        High = 0UL;
        Low = result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void MulModWithNativeModulo(ulong value, ulong modulus)
    {
        High = 0UL;

        ulong a = Low % modulus;
        ulong b = value % modulus;
        if (a == 0UL || b == 0UL)
        {
            Low = 0UL;
            return;
        }

        ulong result = 0UL;
        while (true)
        {
            ulong chunk = b & NativeModuloChunkMask;
            if (chunk != 0UL)
            {
                ulong chunkContribution = MultiplyChunkModulo(a, chunk, modulus);
                result = (result + chunkContribution) % modulus;
            }

            b >>= NativeModuloChunkBits;
            if (b == 0UL)
            {
                break;
            }

            a = ShiftLeftByNativeChunk(a, modulus);
        }

        Low = result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong ShiftLeftByNativeChunk(ulong value, ulong modulus)
    {
        value = (value << 1) % modulus;
        value = (value << 1) % modulus;
        value = (value << 1) % modulus;
        value = (value << 1) % modulus;
        value = (value << 1) % modulus;
        value = (value << 1) % modulus;
        value = (value << 1) % modulus;
        value = (value << 1) % modulus;

        return value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MultiplyChunkModulo(ulong value, ulong chunk, ulong modulus)
    {
        ulong result = 0UL;
        ulong addend = value;

        for (int bit = 0; bit < NativeModuloChunkBits; bit++)
        {
            if ((chunk & NativeModuloBitMasks[bit]) != 0UL)
            {
                result = (result + addend) % modulus;
            }

            if (bit == NativeModuloChunkBitsMinusOne)
            {
                break;
            }

            addend = (addend << 1) % modulus;
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void MulModMontgomery64(GpuUInt128 other, ulong modulus, ulong nPrime, ulong r2)
    {
        ulong a = Low % modulus;
        ulong b = other.Low % modulus;
        if (a == 0UL || b == 0UL)
        {
            High = 0UL;
            Low = 0UL;
            return;
        }

        ulong aR = MontMul64(a, r2, modulus, nPrime);
        ulong bR = MontMul64(b, r2, modulus, nPrime);
        ulong cR = MontMul64(aR, bR, modulus, nPrime);
        ulong c = MontMul64(cR, 1UL, modulus, nPrime);
        High = 0UL;
        Low = c;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void MulModMontgomery64(ulong other, ulong modulus, ulong nPrime, ulong r2)
    {
        ulong a = Low % modulus;
        ulong b = other % modulus;
        if (a == 0UL || b == 0UL)
        {
            High = 0UL;
            Low = 0UL;
            return;
        }

        ulong aR = MontMul64(a, r2, modulus, nPrime);
        ulong bR = MontMul64(b, r2, modulus, nPrime);
        ulong cR = MontMul64(aR, bR, modulus, nPrime);
        ulong c = MontMul64(cR, 1UL, modulus, nPrime);
        High = 0UL;
        Low = c;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void SquareMod(GpuUInt128 modulus)
    {
        if (modulus.High == 0UL)
        {
            MulMod(Low, modulus.Low);
            return;
        }

        SquareFull(this, out var p3, out var p2, out var p1, out var p0);

        GpuUInt128 remainder = new(0UL, 0UL);
        int bit;
        for (bit = 63; bit >= 0; bit--)
        {
            remainder <<= 1;
            remainder = new(remainder.High, remainder.Low | ((p3 >> bit) & 1UL));
            if (remainder.CompareTo(modulus) >= 0)
            {
                remainder.Sub(modulus);
            }
        }

        for (bit = 63; bit >= 0; bit--)
        {
            remainder <<= 1;
            remainder = new(remainder.High, remainder.Low | ((p2 >> bit) & 1UL));
            if (remainder.CompareTo(modulus) >= 0)
            {
                remainder.Sub(modulus);
            }
        }

        for (bit = 63; bit >= 0; bit--)
        {
            remainder <<= 1;
            remainder = new(remainder.High, remainder.Low | ((p1 >> bit) & 1UL));
            if (remainder.CompareTo(modulus) >= 0)
            {
                remainder.Sub(modulus);
            }
        }

        for (bit = 63; bit >= 0; bit--)
        {
            remainder <<= 1;
            remainder = new(remainder.High, remainder.Low | ((p0 >> bit) & 1UL));
            if (remainder.CompareTo(modulus) >= 0)
            {
                remainder.Sub(modulus);
            }
        }

        High = remainder.High;
        Low = remainder.Low;
    }

    public void ModPow(GpuUInt128 exponent, GpuUInt128 modulus)
    {
        GpuUInt128 result = new(0UL, 1UL);
        GpuUInt128 baseValue = this;

        while (!exponent.IsZero)
        {
            if ((exponent.Low & 1UL) != 0UL)
            {
                result.MulMod(baseValue, modulus);
            }

            exponent >>= 1;
            baseValue.MulMod(baseValue, modulus);
        }

        High = result.High;
        Low = result.Low;
    }

    public void ModPow(ulong exponent, GpuUInt128 modulus)
    {
        GpuUInt128 result = new(0UL, 1UL);
        GpuUInt128 baseValue = this;

        while (exponent != 0UL)
        {
            if ((exponent & 1UL) != 0UL)
            {
                result.MulMod(baseValue, modulus);
            }

            exponent >>= 1;
            baseValue.MulMod(baseValue, modulus);
        }

        High = result.High;
        Low = result.Low;
    }

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void ModInv(ulong modulus)
    {
        ModPow(modulus - 2UL, new GpuUInt128(0UL, modulus));
    }

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ModInv(GpuUInt128 modulus)
    {
        if (modulus.High == 0UL)
        {
            ModInv(modulus.Low);
            return;
        }

        var exponent = new GpuUInt128(modulus);
        exponent.Sub(2UL);
        ModPow(exponent, modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override readonly string ToString() =>
        ((UInt128)this).ToString();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void MultiplyFull(GpuUInt128 left, GpuUInt128 right, out ulong p3, out ulong p2, out ulong p1, out ulong p0)
    {
        var (h0, l0) = Mul64(left.Low, right.Low);
        var (h1, l1) = Mul64(left.Low, right.High);
        var (h2, l2) = Mul64(left.High, right.Low);
        var (h3, l3) = Mul64(left.High, right.High);

        p0 = l0;
        ulong carry = 0UL;
        ulong sum = h0;
        sum += l1;
        if (sum < l1)
        {
            carry++;
        }

        sum += l2;
        if (sum < l2)
        {
            carry++;
        }

        p1 = sum;
        sum = h1;
        sum += h2;
        ulong carry2 = sum < h2 ? 1UL : 0UL;
        sum += l3;
        if (sum < l3)
        {
            carry2++;
        }

        sum += carry;
        if (sum < carry)
        {
            carry2++;
        }

        p2 = sum;
        p3 = h3 + carry2;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Multiply(GpuUInt128 left, GpuUInt128 right, out ulong high, out ulong low)
    {
        var (h0, l0) = Mul64(left.Low, right.Low);
        var (_, l1) = Mul64(left.Low, right.High);
        var (_, l2) = Mul64(left.High, right.Low);

        low = l0;
        high = h0 + l1 + l2;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (ulong High, ulong Low) Mul64(ulong left, ulong right)
    {
        ulong a0 = (uint)left;
        ulong a1 = left >> 32;
        ulong b0 = (uint)right;
        ulong b1 = right >> 32;

        ulong lo = a0 * b0;
        ulong mid1 = a1 * b0;
        ulong mid2 = a0 * b1;
        ulong hi = a1 * b1;

        ulong carry = (lo >> 32) + (uint)mid1 + (uint)mid2;
        ulong low = (lo & 0xFFFFFFFFUL) | (carry << 32);
        hi += (mid1 >> 32) + (mid2 >> 32) + (carry >> 32);

        return (hi, low);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SquareFull(GpuUInt128 value, out ulong p3, out ulong p2, out ulong p1, out ulong p0)
    {
        // Compute (H*2^64 + L)^2 = H^2*2^128 + 2*H*L*2^64 + L^2
        ulong L = value.Low;
        ulong H = value.High;
        var (hLL, lLL) = Mul64(L, L);      // L^2 -> (hLL,lLL)
        var (hHH, lHH) = Mul64(H, H);      // H^2 -> (hHH,lHH)
        var (hLH, lLH) = Mul64(L, H);      // L*H -> (hLH,lLH)

        // double LH: (hLH,lLH) << 1
        ulong dLH_low = lLH << 1;
        ulong carry = (lLH >> 63) & 1UL;
        ulong dLH_high = (hLH << 1) | carry;

        // Assemble 256-bit result parts
        p0 = lLL;

        // p1 = hLL + dLH_low (with carry)
        ulong sum1 = hLL + dLH_low;
        ulong c1 = sum1 < hLL ? 1UL : 0UL;
        p1 = sum1;

        // p2 = lHH + dLH_high + c1
        ulong sum2 = lHH + dLH_high;
        ulong c2 = sum2 < lHH ? 1UL : 0UL;
        sum2 += c1;
        if (sum2 < c1)
        {
            c2++;
        }
        p2 = sum2;

        // p3 = hHH + c2
        p3 = hHH + c2;
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MulMod64(ulong a, ulong b, ulong modulus)
    {
        if (modulus <= 1UL)
        {
            return 0UL;
        }

        ulong aReduced = a % modulus;
        ulong bReduced = b % modulus;
        if (aReduced == 0UL || bReduced == 0UL)
        {
            return 0UL;
        }

        if (TryGetMersenneExponent(modulus, out int exponent))
        {
            return MulModMersenne(aReduced, bReduced, modulus, exponent);
        }

        // Fallback for general moduli using the branch-free shift-add reducer.
        ulong result = 0UL;
        ulong x = aReduced;
        ulong y = bReduced;
        while (y != 0UL)
        {
            if ((y & 1UL) != 0UL)
            {
                result += x;
                if (result >= modulus)
                {
                    result -= modulus;
                }
            }

            x <<= 1;
            if (x >= modulus)
            {
                x -= modulus;
            }

            y >>= 1;
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MulModMersenne(ulong a, ulong b, ulong modulus, int exponent)
    {
        Mul64Parts(a, b, out ulong productHigh, out ulong productLow);
        ulong mask = exponent == 64 ? ulong.MaxValue : modulus;

        ulong currentHigh = productHigh;
        ulong currentLow = productLow;

        do
        {
            ulong lower = currentLow & mask;
            ShiftRight128(currentHigh, currentLow, exponent, out ulong shiftedHigh, out ulong shiftedLow);
            currentLow = lower + shiftedLow;
            currentHigh = shiftedHigh;
            if (currentLow < lower)
            {
                currentHigh++;
            }
        }
        while (currentHigh != 0UL);

        ulong result = currentLow;
        while (result >= modulus)
        {
            result -= modulus;
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool TryGetMersenneExponent(ulong modulus, out int exponent)
    {
        if (modulus <= 1UL)
        {
            exponent = 0;
            return false;
        }

        ulong plusOne = modulus + 1UL;
        if (plusOne == 0UL)
        {
            exponent = 64;
            return true;
        }

        if ((plusOne & (plusOne - 1UL)) != 0UL)
        {
            exponent = 0;
            return false;
        }

        int bits = 0;
        while ((plusOne & 1UL) == 0UL)
        {
            plusOne >>= 1;
            bits++;
        }

        exponent = bits;
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ShiftRight128(ulong high, ulong low, int shift, out ulong newHigh, out ulong newLow)
    {
        if (shift == 0)
        {
            newHigh = high;
            newLow = low;
            return;
        }

        if (shift < 64)
        {
            newLow = (low >> shift) | (high << (64 - shift));
            newHigh = high >> shift;
            return;
        }

        if (shift == 64)
        {
            newLow = high;
            newHigh = 0UL;
            return;
        }

        int extra = shift - 64;
        newLow = high >> extra;
        newHigh = 0UL;
    }

    // Montgomery core for 64-bit operands. Returns a*b*R^{-1} mod modulus.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MontMul64(ulong aR, ulong bR, ulong modulus, ulong nPrime)
    {
        // t = aR * bR (128-bit)
        ulong tLow, tHigh;
        Mul64Parts(aR, bR, out tHigh, out tLow);
        // m = (tLow * nPrime) mod 2^64 (low 64 bits only)
        ulong mLow, mHigh;
        Mul64Parts(tLow, nPrime, out mHigh, out mLow);
        // u = (t + m * modulus) >> 64
        ulong mmLow, mmHigh;
        Mul64Parts(mLow, modulus, out mmHigh, out mmLow);
        ulong carry = 0UL;
        ulong low = tLow + mmLow;
        if (low < tLow)
        {
            carry = 1UL;
        }

        ulong high = tHigh + mmHigh + carry;
        ulong u = high; // (t + m*n) >> 64
        if (u >= modulus)
        {
            u -= modulus;
        }

        return u;
    }

    // 64x64 -> 128 multiply into (high, low)
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
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

    private static ulong[] CreateNativeModuloBitMasks()
    {
        ulong[] masks = new ulong[NativeModuloBitMaskTableSize];
        for (int bit = 0; bit < NativeModuloBitMaskTableSize; bit++)
        {
            masks[bit] = bit < 64 ? 1UL << bit : 0UL;
        }

        return masks;
    }
}

    // TODO(MOD-OPT): Montgomery/Barrett integration plan
    // - Introduce caches of modulus-dependent constants:
    //   * Montgomery: n' (-(n^{-1}) mod 2^64 or 2^128), R2 = (R^2 mod n)
    //   * Barrett: mu = floor(2^k / n) for k ∈ {128, 192, 256}
    // - Add fast-path for 64-bit NTT primes (modulus.High == 0UL) using pure 64-bit ops.
    // - Expose helpers to retrieve/calc constants once per modulus and reuse in kernels.
    // - Wire these into MulMod and SquareMod hot paths under feature toggles.
    // - Ensure ILGPU compatibility (no BigInteger, no % inside kernels).

