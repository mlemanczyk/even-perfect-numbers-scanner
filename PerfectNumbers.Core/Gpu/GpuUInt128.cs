using System.Numerics;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core.Gpu;

public readonly struct GpuUInt128 : IComparable<GpuUInt128>, IEquatable<GpuUInt128>
{
    public readonly ulong High;

    public readonly ulong Low;

    public static readonly GpuUInt128 Zero = new(0UL, 0UL);
    public static readonly GpuUInt128 One = new(0UL, 1UL);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuUInt128(ulong high, ulong low)
    {
        High = high;
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
    public static implicit operator GpuUInt128(ulong value) => new(0UL, value);

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
	public static GpuUInt128 operator ++(GpuUInt128 value) => value.Add(1UL);

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
    public GpuUInt128 Add(GpuUInt128 other)
    {
        ulong low = Low + other.Low;
        return new(High + other.High + (low < Low ? 1UL : 0UL), low);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuUInt128 Add(ulong value)
    {
        var low = Low + value;
        return new(High + (low < Low ? 1UL : 0UL), low);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuUInt128 AddMod(GpuUInt128 value, GpuUInt128 modulus)
    {
        if (modulus.High == 0UL)
        {
            return AddMod(value, modulus.Low);
        }

        value = Add(value);
        if (value.CompareTo(modulus) >= 0)
        {
            value = value.Sub(modulus);
        }

        return value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuUInt128 AddMod(ulong value, GpuUInt128 modulus)
    {
        if (modulus.High == 0UL)
        {
            return AddMod(value, modulus.Low);
        }

        var result = Add(value);
        if (result.CompareTo(modulus) >= 0)
        {
            result = result.Sub(modulus);
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuUInt128 AddMod(GpuUInt128 value, ulong modulus)
    {
        ulong a = Low % modulus;
        ulong b = value.Low % modulus;
        ulong sum = a + b;
        if (sum >= modulus || sum < a)
        {
            sum -= modulus;
        }

        return new(0UL, sum);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuUInt128 AddMod(ulong value, ulong modulus)
    {
        ulong a = Low % modulus;
        ulong b = value % modulus;
        ulong sum = a + b;
        if (sum >= modulus || sum < a)
        {
            sum -= modulus;
        }

        return new(0UL, sum);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuUInt128 Sub(GpuUInt128 other)
    {
        ulong borrow = Low < other.Low ? 1UL : 0UL;
        return new(High - other.High + borrow, Low - other.Low);
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
        ulong carry =
            (((xLow * yLow) >> 32) +
            (uint)w1 +
            (uint)w2) >> 32;
        return (xHigh * yHigh) + (w1 >> 32) + (w2 >> 32) + carry;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly GpuUInt128 Mul64(GpuUInt128 other)
    {
        // Multiply this.Low (assumed 64-bit value) by full 128-bit other
        ulong a = Low;
        ulong low = a * other.Low;
        ulong high = a * other.High + MulHigh(a, other.Low);
        return new(high, low);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuUInt128 AddMod(GpuUInt128 a, GpuUInt128 b, GpuUInt128 modulus)
    {
        a = a.Add(b);
        if (a >= modulus)
        {
            a = a.Sub(modulus);
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
                a = a.Sub(modulus);
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


    public GpuUInt128 Sub(ulong value)
    {
        ulong borrow = Low < value ? 1UL : 0UL;
        return new(High - borrow, Low - value);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuUInt128 SubMod(GpuUInt128 value, GpuUInt128 modulus)
    {
        if (CompareTo(value) >= 0)
        {
            return Sub(value);
        }
        else
        {
            return Add(modulus).Sub(value);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuUInt128 SubMod(ulong value, GpuUInt128 modulus)
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
        return new (high - borrow, low - value);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuUInt128 Mul(GpuUInt128 other)
    {
        Multiply(this, other, out var high, out var low);
        return new(high, low);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuUInt128 Mul(ulong value)
    {
        var (highPart, lowPart) = Mul64(Low, value);
        return new(highPart + High * value, lowPart);
    }

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public GpuUInt128 Xor(GpuUInt128 other) => new(High ^ other.High, Low ^ other.Low);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public GpuUInt128 Xor(ulong value) => new(High, Low ^ value);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public GpuUInt128 ShiftLeft(int shift) => this << shift;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public GpuUInt128 ShiftRight(int shift) => this >> shift;

	public GpuUInt128 MulMod(GpuUInt128 other, GpuUInt128 modulus)
    {
        if (modulus.High == 0UL)
        {
            return MulMod(other, modulus.Low);
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
                remainder = remainder.Sub(modulus);
            }
        }

        for (bit = 63; bit >= 0; bit--)
        {
            remainder <<= 1;
            remainder = new(remainder.High, remainder.Low | ((p2 >> bit) & 1UL));
            if (remainder.CompareTo(modulus) >= 0)
            {
                remainder = remainder.Sub(modulus);
            }
        }

        for (bit = 63; bit >= 0; bit--)
        {
            remainder <<= 1;
            remainder = new(remainder.High, remainder.Low | ((p1 >> bit) & 1UL));
            if (remainder.CompareTo(modulus) >= 0)
            {
                remainder = remainder.Sub(modulus);
            }
        }

        for (bit = 63; bit >= 0; bit--)
        {
            remainder <<= 1;
            remainder = new(remainder.High, remainder.Low | ((p0 >> bit) & 1UL));
            if (remainder.CompareTo(modulus) >= 0)
            {
                remainder = remainder.Sub(modulus);
            }
        }

        return remainder;
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
    internal GpuUInt128 MulModBigInteger(GpuUInt128 other, GpuUInt128 modulus)
    {
        var left = (BigInteger)(UInt128)this;
        var right = (BigInteger)(UInt128)other;
        var mod = (BigInteger)(UInt128)modulus;
        var reduced = (UInt128)((left * right) % mod);
        return new((ulong)(reduced >> 64), (ulong)reduced);
    }

    /// <summary>
    /// Experimental limb-based reduction. The current implementation performs
    /// repeated subtractions and becomes extremely slow for large remainders.
    /// Kept for future optimization work.
    /// </summary>
    internal GpuUInt128 MulModByLimb(GpuUInt128 other, GpuUInt128 modulus)
    {
        MultiplyFull(this, other, out var p3, out var p2, out var p1, out var p0);

        GpuUInt128 remainder = new(p3, p2);
        while (remainder.CompareTo(modulus) >= 0)
        {
            remainder = remainder.Sub(modulus);
        }

        ulong limb = p1;
        for (int i = 0; i < 2; i++)
        {
            remainder <<= 64;
            remainder = new(remainder.High, limb);
            while (remainder.CompareTo(modulus) >= 0)
            {
                remainder = remainder.Sub(modulus);
            }

            limb = p0;
        }

        return remainder;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuUInt128 MulMod(ulong value, GpuUInt128 modulus)
    {
        if (modulus.High == 0UL)
        {
            return MulMod(value, modulus.Low);
        }

        GpuUInt128 other = new(0UL, value);
        return MulMod(other, modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuUInt128 MulMod(GpuUInt128 other, ulong modulus)
    {
        // Fast-path: use single 64-bit multiply when it cannot overflow
        // (enabled by ILGPU support for 64-bit % in kernels). Otherwise fall
        // back to the generic branchless double-and-add implementation.
        ulong a = Low % modulus;
        ulong b = other.Low % modulus;
        ulong result;
        if (a == 0UL || b == 0UL)
        {
            result = 0UL;
        }
        else if (b <= ulong.MaxValue / a)
        {
            // Safe to multiply without overflow
            result = (a * b) % modulus;
        }
        else
        {
            result = MulMod64(a, b, modulus);
        }

        return new(0UL, result);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuUInt128 MulMod(ulong value, ulong modulus)
    {
        // Same fast-path as above for 64-bit operands
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

        return new(0UL, result);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuUInt128 MulModMontgomery64(GpuUInt128 other, ulong modulus, ulong nPrime, ulong r2)
    {
        // Montgomery-based 64-bit modular multiplication without division.
        // Computes (this * other) % modulus using:
        // aR = MontMul(a, R^2), bR = MontMul(b, R^2), cR = MontMul(aR, bR), c = MontMul(cR, 1)
        // where R = 2^64, nPrime = -mod^{-1} mod 2^64, r2 = R^2 mod mod.
        ulong a = Low % modulus;
        ulong b = other.Low % modulus;
        if (a == 0UL || b == 0UL)
        {
			return Zero;
        }

        ulong aR = MontMul64(a, r2, modulus, nPrime);
        ulong bR = MontMul64(b, r2, modulus, nPrime);
        ulong cR = MontMul64(aR, bR, modulus, nPrime);
        ulong c = MontMul64(cR, 1UL, modulus, nPrime);
        return new(0UL, c);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuUInt128 MulModMontgomery64(ulong other, ulong modulus, ulong nPrime, ulong r2)
    {
        ulong a = Low % modulus;
        ulong b = other % modulus;
        if (a == 0UL || b == 0UL)
        {
			return Zero; 
        }

        ulong aR = MontMul64(a, r2, modulus, nPrime);
        ulong bR = MontMul64(b, r2, modulus, nPrime);
        ulong cR = MontMul64(aR, bR, modulus, nPrime);
        ulong c = MontMul64(cR, 1UL, modulus, nPrime);
        return new(0UL, c);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuUInt128 SquareMod(GpuUInt128 modulus)
    {
        if (modulus.High == 0UL)
        {
            // Use fast 64-bit path.
            return MulMod(Low, modulus.Low);
        }

        // Specialized 128-bit squaring with bitwise reduction (faster than generic multiply).
        SquareFull(this, out var p3, out var p2, out var p1, out var p0);

        GpuUInt128 remainder = new(0UL, 0UL);
        int bit;
        for (bit = 63; bit >= 0; bit--)
        {
            remainder <<= 1;
            remainder = new(remainder.High, remainder.Low | ((p3 >> bit) & 1UL));
            if (remainder.CompareTo(modulus) >= 0)
            {
                remainder = remainder.Sub(modulus);
            }
        }

        for (bit = 63; bit >= 0; bit--)
        {
            remainder <<= 1;
            remainder = new(remainder.High, remainder.Low | ((p2 >> bit) & 1UL));
            if (remainder.CompareTo(modulus) >= 0)
            {
                remainder = remainder.Sub(modulus);
            }
        }

        for (bit = 63; bit >= 0; bit--)
        {
            remainder <<= 1;
            remainder = new(remainder.High, remainder.Low | ((p1 >> bit) & 1UL));
            if (remainder.CompareTo(modulus) >= 0)
            {
                remainder = remainder.Sub(modulus);
            }
        }

        for (bit = 63; bit >= 0; bit--)
        {
            remainder <<= 1;
            remainder = new(remainder.High, remainder.Low | ((p0 >> bit) & 1UL));
            if (remainder.CompareTo(modulus) >= 0)
            {
                remainder = remainder.Sub(modulus);
            }
        }

		return remainder;
    }

    public GpuUInt128 ModPow(GpuUInt128 exponent, GpuUInt128 modulus)
    {
        GpuUInt128 result = new(0UL, 1UL);
        GpuUInt128 baseValue = this;

        while (!exponent.IsZero)
        {
            if ((exponent.Low & 1UL) != 0UL)
            {
                result = result.MulMod(baseValue, modulus);
            }

            exponent >>= 1;
            baseValue = baseValue.MulMod(baseValue, modulus);
        }

		return result;
    }

    public GpuUInt128 ModPow(ulong exponent, GpuUInt128 modulus)
    {
        GpuUInt128 result = new(0UL, 1UL);
        GpuUInt128 baseValue = this;

        while (exponent != 0UL)
        {
            if ((exponent & 1UL) != 0UL)
            {
                result = result.MulMod(baseValue, modulus);
            }

            exponent >>= 1;
            baseValue = baseValue.MulMod(baseValue, modulus);
        }

        return result;
    }

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public GpuUInt128 ModInv(ulong modulus) => ModPow(modulus - 2UL, new GpuUInt128(0UL, modulus));

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuUInt128 ModInv(GpuUInt128 modulus)
    {
        if (modulus.High == 0UL)
        {
            return ModInv(modulus.Low);
        }

        var exponent = new GpuUInt128(modulus).Sub(2UL);
        return ModPow(exponent, modulus);
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
        // Note: 64-bit % is supported in ILGPU kernels, but the straightforward
        // (UInt128)a*b % modulus is not, because 128-bit multiply is not
        // supported in device code. We therefore implement a branch-free
        // shift-add reduction here based solely on 64-bit ops.
        ulong result = 0UL;
        ulong x = a % modulus;
        ulong y = b;
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
}

    // TODO(MOD-OPT): Montgomery/Barrett integration plan
    // - Introduce caches of modulus-dependent constants:
    //   * Montgomery: n' (-(n^{-1}) mod 2^64 or 2^128), R2 = (R^2 mod n)
    //   * Barrett: mu = floor(2^k / n) for k ∈ {128, 192, 256}
    // - Add fast-path for 64-bit NTT primes (modulus.High == 0UL) using pure 64-bit ops.
    // - Expose helpers to retrieve/calc constants once per modulus and reuse in kernels.
    // - Wire these into MulMod and SquareMod hot paths under feature toggles.
    // - Ensure ILGPU compatibility (no BigInteger, no % inside kernels).
