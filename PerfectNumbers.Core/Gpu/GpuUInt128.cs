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

    public static readonly GpuUInt128 Zero = new();
    public static readonly GpuUInt128 One = new(1UL);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuUInt128()
    {
    }

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
        left.Add(right);
        return left;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuUInt128 operator -(GpuUInt128 left, GpuUInt128 right)
    {
        left.Sub(right);
        return left;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuUInt128 operator ^(GpuUInt128 left, GpuUInt128 right) =>
        new(left.High ^ right.High, left.Low ^ right.Low);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuUInt128 operator *(GpuUInt128 left, GpuUInt128 right)
    {
        Multiply(left, right, out var high, out var low);
        left.High = high;
        left.Low = low;
        return left;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuUInt128 operator <<(GpuUInt128 value, int shift)
    {
        shift &= 127;
        if (shift == 0)
        {
            return value;
        }

        ulong high = value.High;
        ulong low = value.Low;

        if (shift >= 64)
        {
            int longShift = shift - 64;
            value.High = low << longShift;
            value.Low = 0UL;
            return value;
        }

        int inverseShift = 64 - shift;
        value.High = (high << shift) | (low >> inverseShift);
        value.Low = low << shift;
        return value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuUInt128 operator >>(GpuUInt128 value, int shift)
    {
        shift &= 127;
        if (shift == 0)
        {
            return value;
        }

        ulong high = value.High;
        ulong low = value.Low;

        if (shift >= 64)
        {
            int longShift = shift - 64;
            value.Low = longShift == 0 ? high : high >> longShift;
            value.High = 0UL;
            return value;
        }

        int inverseShift = 64 - shift;
        value.Low = (low >> shift) | (high << inverseShift);
        value.High = high >> shift;
        return value;
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
        ulong originalLow = Low;
        ulong low = originalLow + other.Low;
        ulong carry = low < originalLow ? 1UL : 0UL;
        High = High + other.High + carry;
        Low = low;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Add(ulong value)
    {
        ulong low = Low + value;
        High += (low < Low ? 1UL : 0UL);
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

        // TODO: Replace this single-bit ladder with the ProcessEightBitWindows strategy measured
        // in GpuPow2ModBenchmarks once the GPU kernels can share the windowed helper. The windowed
        // version held 21.5 µs for 33-bit moduli versus 51.0 µs here when scanning large divisors.
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

        // TODO: Mirror the eight-bit window batching once the scalar helper above switches; the
        // current bit-serial loop lags by ~3× on 8k–33-bit moduli compared to the windowed path.
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
        // TODO: Pre-reduce the operands via the Montgomery ladder used in MulMod64Benchmarks so the GPU
        // compatible shim stops paying for `%` on every call; the InlineUInt128 helper ran 6–82× faster on
        // large 64-bit workloads while preserving compatibility with the CPU scanner.
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
        // TODO: Fold these operands with the ImmediateModulo helper once the GPU shim exposes it, avoiding
        // repeated `%` reductions that the benchmarks showed are far slower than the Montgomery-based path
        // for dense operands.
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
        ulong yLow = (uint)y;
        ulong lowProduct = xLow * yLow;

        ulong yHigh = y >> 32;
        ulong cross = xLow * yHigh;

        ulong xHigh = x >> 32;
        ulong result = xHigh * yHigh;
        ulong temp = xHigh * yLow;

        // Keeping the wide partial sum in a dedicated local prevents the JIT from
        // materialising it on the stack before the final carry propagation. The
        // additional store looks redundant in C#, but it shortens the generated
        // instruction sequence by avoiding an extra temporary and results in a
        // measurable throughput win in the MulHigh benchmarks.
        result += cross >> 32;
        result += temp >> 32;

        lowProduct >>= 32;
        lowProduct += (uint)cross;
        lowProduct += (uint)temp;
        result += lowProduct >> 32;
        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Mul64(GpuUInt128 other)
    {
        // Multiply this.Low (assumed 64-bit value) by full 128-bit other
        ulong operand = Low;
        Low = operand * other.Low;
        ulong highProduct = operand * other.High;
        High = highProduct + MulHigh(operand, other.Low);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void MulMod(in GpuUInt128 value, in GpuUInt128 modulus)
    {
        GpuUInt128 multiplicand = this;
        GpuUInt128 multiplier = value;

        High = 0UL;
        Low = 0UL;

        while (!multiplier.IsZero)
        {
            if ((multiplier.Low & 1UL) != 0UL)
            {
                AddMod(multiplicand, modulus);
            }

            multiplicand.ShiftLeft(1);
            if (multiplicand.CompareTo(modulus) >= 0)
            {
                multiplicand.Sub(modulus);
            }

            multiplier.ShiftRight(1);
        }
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GpuUInt128 Pow2Mod(ulong exponent, GpuUInt128 modulus)
    {
        if (modulus.IsZero || modulus == One)
        {
            return new();
        }

        const int WindowSizeBits = 8;
        const ulong WindowMask = (1UL << WindowSizeBits) - 1UL;

        GpuUInt128 result = new(1UL);
        GpuUInt128 baseVal = new(2UL);

        ulong e = exponent;
        while (e != 0UL)
        {
            int remainingBits = BitOperations.Log2(e) + 1;
            int bitsToProcess = remainingBits >= WindowSizeBits ? WindowSizeBits : remainingBits;
            ulong chunk = e & (WindowMask >> (WindowSizeBits - bitsToProcess));

            for (int bitIndex = 0; bitIndex < bitsToProcess; bitIndex++)
            {
                if ((chunk & 1UL) != 0UL)
                {
                    result.MulMod(baseVal, modulus);
                }

                baseVal.MulMod(baseVal, modulus);
                chunk >>= 1;
            }

            e >>= bitsToProcess;
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
        // TODO: Replace this scalar binary GCD with the branchless reduction from
        // GpuUInt128BinaryGcdBenchmarks so CPU fallbacks stay aligned with the GPU kernel
        // performance when resolving large divisor residues.
        if (u.IsZero)
        {
            return v;
        }

        if (v.IsZero)
        {
            return u;
        }

        int shift = TrailingZeroCount(new GpuUInt128(u.High | v.High, u.Low | v.Low));
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
        ulong low = Low;
        ulong high = High;
        if (high == 0UL && low < value)
        {
            low += modulus.Low;
            high = modulus.High + (low < modulus.Low ? 1UL : 0UL);
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
        shift &= 127;
        if (shift == 0)
        {
            return;
        }

        ulong high = High;
        ulong low = Low;

        if (shift >= 64)
        {
            High = low << (shift - 64);
            Low = 0UL;
            return;
        }

        int inverseShift = 64 - shift;
        High = (high << shift) | (low >> inverseShift);
        Low = low << shift;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ShiftRight(int shift)
    {
        shift &= 127;
        if (shift == 0)
        {
            return;
        }

        ulong high = High;
        ulong low = Low;

        if (shift >= 64)
        {
            Low = shift == 64 ? high : high >> (shift - 64);
            High = 0UL;
            return;
        }

        int inverseShift = 64 - shift;
        Low = (low >> shift) | (high << inverseShift);
        High = high >> shift;
    }

    public void MulMod(GpuUInt128 other, GpuUInt128 modulus)
    {
        if (modulus.High == 0UL)
        {
            Low = MulMod(other, modulus.Low);
            High = 0UL;
            return;
        }

        MultiplyFull(this, other, out var p3, out var p2, out var p1, out var p0);
        ReduceProductBitwise(p3, p2, p1, p0, modulus);
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
        // TODO: Relocate this limb-based reducer to the benchmark project once the production
        // pipeline switches to the faster allocating legacy path demonstrated in the benchmarks.
        MultiplyFull(this, other, out var p3, out var p2, out var p1, out var p0);

        High = p3;
        Low = p2;
        while (CompareTo(modulus) >= 0)
        {
            Sub(modulus);
        }

        ulong limb = p1;
        for (int i = 0; i < 2; i++)
        {
            ShiftLeft(64);
            Low = limb;
            while (CompareTo(modulus) >= 0)
            {
                Sub(modulus);
            }

            limb = p0;
        }

    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void MulMod(ulong value, GpuUInt128 modulus)
    {
        if (modulus.High == 0UL)
        {
            Low = MulMod(value, modulus.Low);
            High = 0UL;
            return;
        }

        MultiplyFull(this, value, out var p3, out var p2, out var p1, out var p0);
        ReduceProductBitwise(p3, p2, p1, p0, modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly ulong MulMod(GpuUInt128 other, ulong modulus)
    {
        ulong modulusLocal = modulus;
        ulong a = Low % modulusLocal;
        ulong b = other.Low % modulusLocal;

        if (a == 0UL || b == 0UL)
        {
            return 0UL;
        }

        return b <= ulong.MaxValue / a
            ? (a * b) % modulusLocal
            : MulMod64(a, b, modulusLocal);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly ulong MulMod(ulong value, ulong modulus)
    {
        ulong a = Low % modulus;
        ulong b = value % modulus;

        if (a == 0UL || b == 0UL)
        {
            return 0UL;
        }

        ulong ulongRange = ulong.MaxValue / a;
        return b <= ulongRange
            ? (a * b) % modulus
            : MulMod64(a, b, modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly ulong MulModSimplified(ulong value, ulong modulus)
    {
        ulong modulusLocal = modulus;
        ulong a = Low % modulusLocal;
        ulong b = value % modulusLocal;

        if (a == 0UL || b == 0UL)
        {
            return 0UL;
        }

        return MulMod64(a, b, modulusLocal);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly ulong MulModWithNativeModulo(ulong value, ulong modulus)
    {
        // TODO: Drop this native-modulo path from production after migrating callers to MulMod,
        // which benchmarked 4-8× faster on dense operands and still wins on mixed workloads.
        ulong multiplicand = Low % modulus;
        var remainder = value % modulus;

        if (multiplicand == 0UL || remainder == 0UL)
        {
            return 0UL;
        }

        ulong result = 0UL;
        while (true)
        {
            ulong chunk = remainder & NativeModuloChunkMask;
            if (chunk != 0UL)
            {
                chunk = MultiplyChunkModulo(multiplicand, chunk, modulus);
                result = (result + chunk) % modulus;
            }

            remainder >>= NativeModuloChunkBits;
            if (remainder == 0UL)
            {
                break;
            }

            multiplicand = ShiftLeftByNativeChunk(multiplicand, modulus);
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ReduceProductBitwise(ulong p3, ulong p2, ulong p1, ulong p0, GpuUInt128 modulus)
    {
        GpuUInt128 remainder = new();
        int bit;
        // TODO: Can we modify these loops to process multiple bits at a time? E.g. 64-bit chunks.
        for (bit = 63; bit >= 0; bit--)
        {
            remainder.ShiftLeft(1);
            remainder.Low |= (p3 >> bit) & 1UL;
            if (remainder.CompareTo(modulus) >= 0)
            {
                remainder.Sub(modulus);
            }
        }

        for (bit = 63; bit >= 0; bit--)
        {
            remainder.ShiftLeft(1);
            remainder.Low |= (p2 >> bit) & 1UL;
            if (remainder.CompareTo(modulus) >= 0)
            {
                remainder.Sub(modulus);
            }
        }

        for (bit = 63; bit >= 0; bit--)
        {
            remainder.ShiftLeft(1);
            remainder.Low |= (p1 >> bit) & 1UL;
            if (remainder.CompareTo(modulus) >= 0)
            {
                remainder.Sub(modulus);
            }
        }

        for (bit = 63; bit >= 0; bit--)
        {
            remainder.ShiftLeft(1);
            remainder.Low |= (p0 >> bit) & 1UL;
            if (remainder.CompareTo(modulus) >= 0)
            {
                remainder.Sub(modulus);
            }
        }

        High = remainder.High;
        Low = remainder.Low;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong ShiftLeftByNativeChunk(ulong value, ulong modulus)
    {
        // TODO: Collapse this eight-step shift ladder into the ProcessEightBitWindows helper once it lands so
        // we reuse the precomputed window residues instead of emitting `% modulus` after every shift.
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
        ulong[] nativeModuloBitMasks = NativeModuloBitMasks;
        for (int bit = 0; bit < NativeModuloChunkBits; bit++)
        {
            if ((chunk & nativeModuloBitMasks[bit]) != 0UL)
            {
                result = (result + value) % modulus;
            }

            if (bit == NativeModuloChunkBitsMinusOne)
            {
                break;
            }

            value = (value << 1) % modulus;
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly ulong MulModMontgomery64(GpuUInt128 other, ulong modulus, ulong nPrime, ulong r2)
    {
        // TODO: Retire this struct-based Montgomery path from production after adopting the extension
        // helper, which benchmarks 6-7× faster across dense and near-modulus operands.
        ulong modulusLocal = modulus;
        ulong a = Low % modulusLocal;
        ulong b = other.Low % modulusLocal;

        if (a == 0UL || b == 0UL)
        {
            return 0UL;
        }

        ulong aR = MontMul64(a, r2, modulusLocal, nPrime);
        ulong bR = MontMul64(b, r2, modulusLocal, nPrime);
        ulong cR = MontMul64(aR, bR, modulusLocal, nPrime);
        return MontMul64(cR, 1UL, modulusLocal, nPrime);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly ulong MulModMontgomery64(ulong other, ulong modulus, ulong nPrime, ulong r2)
    {
        // TODO: Same as above—migrate callers to the scalar extension to avoid this 6-7× slowdown.
        ulong modulusLocal = modulus;
        ulong a = Low % modulusLocal;
        ulong b = other % modulusLocal;

        if (a == 0UL || b == 0UL)
        {
            return 0UL;
        }

        ulong aR = MontMul64(a, r2, modulusLocal, nPrime);
        ulong bR = MontMul64(b, r2, modulusLocal, nPrime);
        ulong cR = MontMul64(aR, bR, modulusLocal, nPrime);
        return MontMul64(cR, 1UL, modulusLocal, nPrime);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void SquareMod(GpuUInt128 modulus)
    {
        if (modulus.High == 0UL)
        {
            Low = MulMod(Low, modulus.Low);
            High = 0UL;
            return;
        }

        SquareFull(this, out var p3, out var p2, out var p1, out var p0);

        // TODO: This should operate on the instance itself, not on a copy. Avoid creating new instances anywhere.
        GpuUInt128 remainder = new();
        int bit;
        // TODO: Can we modify these loops to process multiple bits at a time? E.g. 64-bit chunks.
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
        // TODO: Swap this copy-heavy path for the pooled base/exponent ladder used in
        // GpuUInt128MulModBenchmarks so the GPU shim keeps reusing buffers instead of
        // allocating temporary structs during residue scans.
        GpuUInt128 result = new(1UL);
        GpuUInt128 baseValue = this;

        // TODO: Replace the single-bit square-and-multiply loop with the 64-bit windowed
        // ladder measured fastest in GpuUInt128MulModBenchmarks to align with the GPU kernel
        // implementation once the shared helper lands.
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
        // TODO: Share the pooled ladder state from GpuUInt128MulModBenchmarks here as well so
        // Lucas–Lehmer batches avoid constructing throwaway temporaries on every exponent.
        GpuUInt128 result = new(1UL);
        GpuUInt128 baseValue = this;

        // TODO: Upgrade this loop to the same 64-bit windowed ladder proven fastest in
        // GpuUInt128MulModBenchmarks so the scalar helper matches the GPU-optimized path.
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
        // TODO: Replace this Fermat inversion with the Montgomery ladder highlighted in
        // GpuUInt128Montgomery64Benchmarks so we avoid instantiating a temporary modulus and
        // reuse the pooled reduction constants.
        ModPow(modulus - 2UL, new GpuUInt128(modulus));
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
        // TODO: Reuse variables to reduce register pressure following the fused-limb layout
        // from GpuUInt128MulModByLimbBenchmarks so the multiply helper matches the fastest
        // GPU-compatible scalar routine.
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
    private static void MultiplyFull(GpuUInt128 left, ulong right, out ulong p3, out ulong p2, out ulong p1, out ulong p0)
    {
        var (highLow, lowLow) = Mul64(left.Low, right);
        var (highHigh, lowHigh) = Mul64(left.High, right);

        p0 = lowLow;

        ulong mid = highLow + lowHigh;
        ulong carry = mid < highLow ? 1UL : 0UL;
        p1 = mid;

        ulong upper = highHigh + carry;
        p3 = upper < highHigh ? 1UL : 0UL;
        p2 = upper;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Multiply(GpuUInt128 left, GpuUInt128 right, out ulong high, out ulong low)
    {
        // TODO: Reuse variables to reduce register pressure.
        var (h0, l0) = Mul64(left.Low, right.Low);
        low = l0;
        // TODO: Since we ignore the first result element, can we create a version of the function which calculates and returns only the second element?
        (_, l0) = Mul64(left.Low, right.High);
        var (_, l2) = Mul64(left.High, right.Low);

        // TODO: Why not just modify left instance directly instead using out parameters?
        high = h0 + l0 + l2;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (ulong High, ulong Low) Mul64(ulong left, ulong right)
    {
        // TODO: Reuse variables to reduce register pressure.
        ulong a0 = (uint)left;
        ulong a1 = left >> 32;
        ulong b0 = (uint)right;
        ulong b1 = right >> 32;

        ulong lo = a0 * b0;
        ulong mid1 = a1 * b0;
        b0 = a0 * b1;
        b1 *= a1;

        a0 = (lo >> 32) + (uint)mid1 + (uint)b0;
        a1 = (lo & 0xFFFFFFFFUL) | (a0 << 32);
        b1 += (mid1 >> 32) + (b0 >> 32) + (a0 >> 32);

        return (b1, a1);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SquareFull(GpuUInt128 value, out ulong p3, out ulong p2, out ulong p1, out ulong p0)
    {
        // TODO: Reuse variables to reduce register pressure.
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
        ulong aReduced = a % modulus;
        ulong bReduced = b % modulus;
        if (aReduced == 0UL || bReduced == 0UL)
        {
            return 0UL;
        }

        // TODO: Can we identify where we deal with Mersenne exponent in EvenPerfectBitScanner and directly use MulModMersenne there, removing this branch?
        if (TryGetMersenneExponent(modulus, out int exponent))
        {
            return MulModMersenne(aReduced, bReduced, modulus, exponent);
        }

        // Fallback for general moduli using the branch-free shift-add reducer.
        ulong result = 0UL;
        ulong x = aReduced;
        ulong y = bReduced;
        // TODO: Can we modify this loop to process multiple bits at a time? E.g. 64-bit chunks.
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
        // TODO: Reuse variables to reduce register pressure.
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
        // TODO: Can we modify this loop to process multiple bits at a time? E.g. 64-bit chunks.
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

        modulus++;
        if (modulus == 0UL)
        {
            exponent = 64;
            return true;
        }

        if ((modulus & (modulus - 1UL)) != 0UL)
        {
            exponent = 0;
            return false;
        }

        int bits = 0;
        // TODO: Can we modify this loop to process multiple bits at a time? E.g. 64-bit chunks.
        while ((modulus & 1UL) == 0UL)
        {
            modulus >>= 1;
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
        // TODO: Reuse variables to reduce register pressure.
        // t = aR * bR (128-bit)
        Mul64Parts(aR, bR, out ulong tHigh, out ulong tLow);
        // m = (tLow * nPrime) mod 2^64 (low 64 bits only)
        Mul64Parts(tLow, nPrime, out _, out ulong mLow);
        // u = (t + m * modulus) >> 64
        Mul64Parts(mLow, modulus, out ulong mmHigh, out ulong mmLow);
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
        // TODO: Reuse variables to reduce register pressure.
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

    // TODO: Check if the TODO below is still relevant.

    // TODO(MOD-OPT): Montgomery/Barrett integration plan
    // - Introduce caches of modulus-dependent constants:
    //   * Montgomery: n' (-(n^{-1}) mod 2^64 or 2^128), R2 = (R^2 mod n)
    //   * Barrett: mu = floor(2^k / n) for k ∈ {128, 192, 256}
    // - Add fast-path for 64-bit NTT primes (modulus.High == 0UL) using pure 64-bit ops.
    // - Expose helpers to retrieve/calc constants once per modulus and reuse in kernels.
    // - Wire these into MulMod and SquareMod hot paths under feature toggles.
    // - Ensure ILGPU compatibility (no BigInteger, no % inside kernels).

