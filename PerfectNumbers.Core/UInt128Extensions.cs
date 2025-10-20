using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

public static class UInt128Extensions
{
    private const int Pow2WindowSize = 8;
    private const ulong Pow2WindowFallbackThreshold = 32UL;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static UInt128 BinaryGcd(this UInt128 u, UInt128 v)
    {
        // TODO: Replace this hand-rolled binary GCD with the shared subtract-free ladder from
        // GpuUInt128BinaryGcdBenchmarks so wide operands reuse the optimized helper instead of
        // repeating the slower shift/subtract loop on both CPU and GPU paths.
        UInt128 zero = UInt128.Zero;
        if (u == zero)
        {
            return v;
        }

        if (v == zero)
        {
            return u;
        }

        int shift = CountTrailingZeros(u | v);
        u >>= CountTrailingZeros(u);

        do
        {
            v >>= CountTrailingZeros(v);
            if (u > v)
            {
                (u, v) = (v, u);
            }

            v -= u;
        }
        while (v != zero);

        return u << shift;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int GetBitLength(this UInt128 value)
    {
        ulong high = (ulong)(value >> 64);
        if (high != 0UL)
        {
            return 64 + (64 - BitOperations.LeadingZeroCount(high));
        }

        ulong low = (ulong)value;
        if (low == 0UL)
        {
            return 0;
        }

        return 64 - BitOperations.LeadingZeroCount(low);
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static UInt128 Pow2MontgomeryModWindowed(this UInt128 exponent, UInt128 modulus)
    {
        // TODO: Prototype a UInt128-native Pow2MontgomeryModWindowed path that matches the GPU implementation once a faster modular multiplication helper is available, so we can revisit removing the GPU struct dependency without regressing benchmarks.
        if (modulus == UInt128.One)
        {
            return UInt128.Zero;
        }

        if (exponent == UInt128.Zero)
        {
            return UInt128.One % modulus;
        }

        GpuUInt128 modulusGpu = new(modulus);
        GpuUInt128 baseValue = new(2UL);
        if (baseValue.CompareTo(modulusGpu) >= 0)
        {
            baseValue.Sub(modulusGpu);
        }

        if (ShouldUseSingleBit(exponent))
        {
            GpuUInt128 singleBitResult = Pow2MontgomeryModSingleBit(exponent, modulusGpu, baseValue);
            return (UInt128)singleBitResult;
        }

        GpuUInt128 exponentGpu = new(exponent);
        int bitLength = exponentGpu.GetBitLength();
        int windowSize = GetWindowSize(bitLength);
        int oddPowerCount = 1 << (windowSize - 1);

        Span<GpuUInt128> oddPowers = stackalloc GpuUInt128[oddPowerCount];
        InitializeOddPowers(baseValue, modulusGpu, oddPowers);

        GpuUInt128 result = GpuUInt128.One;
        int index = bitLength - 1;

        while (index >= 0)
        {
            if (!IsBitSet(exponentGpu, index))
            {
                result.MulMod(result, modulusGpu);
                index--;
                continue;
            }

            int windowStart = index - windowSize + 1;
            if (windowStart < 0)
            {
                windowStart = 0;
            }

            while (!IsBitSet(exponentGpu, windowStart))
            {
                windowStart++;
            }

            int windowBitCount = index - windowStart + 1;
            for (int square = 0; square < windowBitCount; square++)
            {
                result.MulMod(result, modulusGpu);
            }

            ulong windowValue = ExtractWindowValue(exponentGpu, windowStart, windowBitCount);
            int tableIndex = (int)((windowValue - 1UL) >> 1);
            GpuUInt128 factor = oddPowers[tableIndex];
            result.MulMod(factor, modulusGpu);

            index = windowStart - 1;
        }

        return (UInt128)result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool ShouldUseSingleBit(UInt128 exponent) => (exponent >> 64) == UInt128.Zero && (ulong)exponent <= Pow2WindowFallbackThreshold;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int GetWindowSize(int bitLength)
    {
        if (bitLength <= Pow2WindowSize)
        {
            return Math.Max(bitLength, 1);
        }

        if (bitLength <= 23)
        {
            return 4;
        }

        if (bitLength <= 79)
        {
            return 5;
        }

        if (bitLength <= 239)
        {
            return 6;
        }

        if (bitLength <= 671)
        {
            return 7;
        }

        return Pow2WindowSize;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void InitializeOddPowers(GpuUInt128 baseValue, GpuUInt128 modulus, Span<GpuUInt128> oddPowers)
    {
        oddPowers[0] = baseValue;
        if (oddPowers.Length == 1)
        {
            return;
        }

        // Reusing baseValue to hold base^2 for the shared odd-power ladder, just like the GPU helper.
        baseValue.MulMod(baseValue, modulus);
        for (int i = 1; i < oddPowers.Length; i++)
        {
            oddPowers[i] = oddPowers[i - 1];
            oddPowers[i].MulMod(baseValue, modulus);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static GpuUInt128 Pow2MontgomeryModSingleBit(UInt128 exponent, GpuUInt128 modulus, GpuUInt128 baseValue)
    {
        GpuUInt128 result = GpuUInt128.One;
        GpuUInt128 remainingExponent = new(exponent);

        while (!remainingExponent.IsZero)
        {
            if ((remainingExponent.Low & 1UL) != 0UL)
            {
                result.MulMod(baseValue, modulus);
            }

            remainingExponent.ShiftRight(1);
            if (remainingExponent.IsZero)
            {
                break;
            }

            // Reusing baseValue to hold the squared base in-place between iterations.
            baseValue.MulMod(baseValue, modulus);
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsBitSet(GpuUInt128 value, int bitIndex)
    {
        if (bitIndex >= 64)
        {
            return ((value.High >> (bitIndex - 64)) & 1UL) != 0UL;
        }

        return ((value.Low >> bitIndex) & 1UL) != 0UL;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong ExtractWindowValue(GpuUInt128 exponent, int windowStart, int windowBitCount)
    {
		ulong mask = (1UL << windowBitCount) - 1UL;
        if (windowStart != 0)
        {
            exponent.ShiftRight(windowStart);
            return exponent.Low & mask;
        }

        return exponent.Low & mask;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static UInt128 ReduceProductBitwise(ulong p3, ulong p2, ulong p1, ulong p0, UInt128 modulus)
    {
        UInt128 remainder = UInt128.Zero;
        for (int bit = 63; bit >= 0; bit--)
        {
            remainder <<= 1;
            remainder |= (UInt128)((p3 >> bit) & 1UL);
            if (remainder >= modulus)
            {
                remainder -= modulus;
            }
        }

        for (int bit = 63; bit >= 0; bit--)
        {
            remainder <<= 1;
            remainder |= (UInt128)((p2 >> bit) & 1UL);
            if (remainder >= modulus)
            {
                remainder -= modulus;
            }
        }

        for (int bit = 63; bit >= 0; bit--)
        {
            remainder <<= 1;
            remainder |= (UInt128)((p1 >> bit) & 1UL);
            if (remainder >= modulus)
            {
                remainder -= modulus;
            }
        }

        for (int bit = 63; bit >= 0; bit--)
        {
            remainder <<= 1;
            remainder |= (UInt128)((p0 >> bit) & 1UL);
            if (remainder >= modulus)
            {
                remainder -= modulus;
            }
        }

        return remainder;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void MultiplyFull(UInt128 left, UInt128 right, out ulong p3, out ulong p2, out ulong p1, out ulong p0)
    {
        ulong leftLow = (ulong)left;
        ulong leftHigh = (ulong)(left >> 64);
        ulong rightLow = (ulong)right;
        ulong rightHigh = (ulong)(right >> 64);

        var partial0 = Multiply64(leftLow, rightLow);
        var partial1 = Multiply64(leftLow, rightHigh);
        var partial2 = Multiply64(leftHigh, rightLow);
        var partial3 = Multiply64(leftHigh, rightHigh);

        p0 = partial0.Low;
        ulong carry = partial0.High;
        ulong sum = carry + partial1.Low;
        ulong carryMid = sum < partial1.Low ? 1UL : 0UL;
        sum += partial2.Low;
        if (sum < partial2.Low)
        {
            carryMid++;
        }

        p1 = sum;
        sum = partial1.High + partial2.High;
        ulong carryHigh = sum < partial2.High ? 1UL : 0UL;
        sum += partial3.Low;
        if (sum < partial3.Low)
        {
            carryHigh++;
        }

        sum += carryMid;
        if (sum < carryMid)
        {
            carryHigh++;
        }

        p2 = sum;
        p3 = partial3.High + carryHigh;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (ulong High, ulong Low) Multiply64(ulong left, ulong right)
    {
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

    public static ulong CalculateOrder(this UInt128 q)
    {
        if (q <= UInt128Numbers.Two)
        {
            return 0UL;
        }

        UInt128 one = UInt128.One;
        UInt128 phi = q - one;
        if (phi > ulong.MaxValue)
        {
            throw new NotImplementedException("Such big values are not yet supported");
        }

        ulong order = (ulong)phi;
        uint[] smallPrimes = PrimesGenerator.SmallPrimes;
        ulong[] smallPrimesPow2 = PrimesGenerator.SmallPrimesPow2;

        int i = 0, primesLength = smallPrimes.Length;
    UInt128 cycle = MersenneDivisorCycles.GetCycle(q);
    // TODO: If the cache lacks this cycle, immediately schedule the configured device
    // (GPU by default) to compute it on the fly and skip inserting it into the cache so
    // wide-order factoring can still leverage cycle stepping without breaching the
    // memory cap or introducing extra synchronization.
        ulong prime, temp;
        for (; i < primesLength; i++)
        {
            if (smallPrimesPow2[i] > order)
            {
                break;
            }

            prime = smallPrimes[i];
      while (order % prime == 0UL)
      {
        temp = order / prime;
        // TODO: Switch this divisor-order powmod to the ProcessEightBitWindows helper so the
        // cycle factoring loop benefits from the faster windowed pow2 ladder measured in CPU benchmarks.
        if (temp.PowModWithCycle(q, cycle) == one)
        {
          order = temp;
        }
                else
                {
                    break;
                }
            }
        }

        return order;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int CountTrailingZeros(this UInt128 value)
    {
        ulong valuePart = (ulong)value;
        if (valuePart != 0UL)
        {
            return BitOperations.TrailingZeroCount(valuePart);
        }

        return 64 + BitOperations.TrailingZeroCount((ulong)(value >> 64));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsPrimeCandidate(this UInt128 n)
    {
        UInt128 p, zero = UInt128.Zero;

        uint[] smallPrimes = PrimesGenerator.SmallPrimes;
        ulong[] smallPrimesPow2 = PrimesGenerator.SmallPrimesPow2;
        ulong i, smallPrimesCount = (ulong)smallPrimes.Length;
    for (i = 0UL; i < smallPrimesCount; i++)
    {
      if (smallPrimesPow2[i] > n)
      {
        break;
      }

      p = smallPrimes[i];
      // TODO: Replace this direct `%` test with the shared divisor-cycle filter once the
      // UInt128 path is wired into the cached cycle tables so wide candidates skip the slow
      // modulo checks during primality pre-filtering.
      if (n % p == zero)
      {
        return n == p;
      }
    }

        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Mod3(this UInt128 value)
    {
        // TODO: Fold these reductions into the multiply-high trick captured in Mod3BenchmarkResults so
        // the UInt128 modulo helpers avoid `%` altogether and align with the faster CPU/GPU residue filters.
        ulong remainder = ((ulong)value) % 3UL + ((ulong)(value >> 64)) % 3UL;
        return remainder >= 3UL ? remainder - 3UL : remainder;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Mod5(this UInt128 value)
    {
        // TODO: Replace the `% 5` operations with the precomputed multiply-high constants from the Mod5
        // benchmarks so the UInt128 path matches the 64-bit helpers without extra modulo instructions.
        ulong remainder = (((ulong)value) % 5UL) + (((ulong)(value >> 64)) % 5UL);
        return remainder >= 5UL ? remainder - 5UL : remainder;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Mod6(this UInt128 value) => ((value.Mod3() << 1) | ((ulong)value & 1UL)) switch
    {
        0UL => 0UL,
        1UL => 3UL,
        2UL => 4UL,
        3UL => 1UL,
        4UL => 2UL,
        _ => 5UL,
    };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Mod7(this UInt128 value)
    {
        // TODO: Inline the lookup-based Mod7 reducer validated in the residue benchmarks so this helper
        // stops relying on `% 7` in hot loops and mirrors the optimized CPU cycle filters.
        ulong low = (ulong)value % 7UL;
        ulong high = (ulong)(value >> 64) % 7UL;
        ulong remainder = low + (high * 2UL);
        if (remainder >= 7UL)
        {
      remainder -= 7UL;
      if (remainder >= 7UL)
      {
        remainder -= 7UL;
      }
    }

    return remainder;
  }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Mod8(this UInt128 value) => (ulong)value & 7UL;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Mod10(this UInt128 value128)
    {
        // Split and fold under mod 10: 2^64 ≡ 6 (mod 10)
        // TODO: Precompute the 2^64 ≡ 6 folding constants once so this path stops recomputing `% 10`
        // on each half and instead uses the span-based lookup captured in Mod10BenchmarkResults.
        ulong low = (ulong)value128;
        ulong high = (ulong)(value128 >> 64);
        return ((low % 10UL) + ((high % 10UL) * 6UL)) % 10UL;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Mod10_8_5_3(this UInt128 value, out ulong mod10, out ulong mod8, out ulong mod5, out ulong mod3)
    {
        byte current;
        uint byteSum = 0U;
        mod8 = (ulong)value & 7UL;
        UInt128 zero = UInt128.Zero;
        do
        {
            current = (byte)value;
            byteSum += current;
            value >>= 8;
        }
        while (value != zero);

        mod5 = byteSum.Mod5();
        // TODO: Collapse this Mod10 switch into the shared lookup table from the CLI benchmarks so we
        // avoid the nested switch expressions once the pooled residue tables become available.
        mod10 = (mod8 & 1UL) == 0UL
            ? mod5 switch
            {
                0U => 0UL,
                1U => 6UL,
                            2U => 2UL,
                            3U => 8UL,
                            _ => 4UL,
                        }
                        : mod5 switch
                        {
                            0U => 5UL,
                            1U => 1UL,
                            2U => 7UL,
                            3U => 3UL,
                            _ => 9UL,
                        };

        mod3 = byteSum % 3U;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Mod11(this UInt128 value)
    {
        ulong remainder = (((ulong)value) % 11UL) + (((ulong)(value >> 64)) % 11UL) * 5UL;
        while (remainder >= 11UL)
        {
            remainder -= 11UL;
        }

        return remainder;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Mod128(this UInt128 value) => (ulong)value & 127UL;


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static UInt128 ModPow(this UInt128 value, UInt128 exponent, UInt128 modulus)
    {
        UInt128 zero = UInt128.Zero;
        UInt128 one = UInt128.One;
        UInt128 result = UInt128.One;
        value %= modulus;

        while (exponent != zero)
        {
            if ((exponent & one) != zero)
            {
                result = MulMod(result, value, modulus);
            }

            value = MulMod(value, value, modulus);
            exponent >>= 1;
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public static UInt128 Mul64(this UInt128 a, UInt128 b)
  {
    ulong aLow = (ulong)a;
    ulong bLow = (ulong)b;
    ulong bHigh = (ulong)(b >> 64);

    ulong low = aLow * bLow;

    // Keep the high-word accumulation in locals so the JIT does not rebuild the
    // expression tree around the shift. Mirroring the MulHigh layout lets RyuJIT
    // keep the intermediate sum in registers instead of reloading the partial
    // product from the stack.
    ulong high = aLow.MulHigh(bLow);
    high += aLow * bHigh;

    return ((UInt128)high << 64) | low;
  }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static UInt128 MulMod(this UInt128 a, UInt128 b, UInt128 modulus)
    {
        if (modulus <= UInt128.One)
        {
            return UInt128.Zero;
        }

        UInt128 left = a % modulus;
        UInt128 right = b % modulus;
        if (left == UInt128.Zero || right == UInt128.Zero)
        {
            return UInt128.Zero;
        }

        if ((modulus >> 64) == UInt128.Zero)
        {
            // Reusing left to hold the reduced product; this mirrors the inline UInt128 multiply
            // that topped MulMod64Benchmarks for 64-bit moduli.
            ulong modulus64 = (ulong)modulus;
            left = (UInt128)(((UInt128)(ulong)left * (ulong)right) % modulus64);
            return left;
        }

        MultiplyFull(left, right, out var p3, out var p2, out var p1, out var p0);
        // Reusing left to capture the reduced 128-bit remainder before returning.
        left = ReduceProductBitwise(p3, p2, p1, p0, modulus);
        return left;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static UInt128 AddMod(this UInt128 value, UInt128 addend, UInt128 modulus)
    {
        // All current CPU callers (Pollard Rho and the Mersenne residue tracker) only supply moduli
        // of at least three, so leave this guard documented but disabled.
        // if (modulus <= UInt128.One)
        // {
        //     return UInt128.Zero;
        // }

        // Both operands arrive pre-reduced on these paths, letting us skip the redundant folds the old
        // helper performed.
        // value %= modulus;
        // addend %= modulus;
        // Pollard Rho keeps its polynomial constant inside [1, modulus - 1], and the Mersenne pow-delta
        // term never hits zero for odd moduli, leaving this branch inactive outside targeted tests.
        // if (addend == UInt128.Zero)
        // {
        //     return value;
        // }

        UInt128 threshold = modulus - addend;
        if (value >= threshold)
        {
            return value - threshold;
        }

        UInt128 sum = value + addend;
        // The sum stays below the modulus under the threshold guard above, so no wraparound occurs.
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static UInt128 SubtractOneMod(this UInt128 value, UInt128 modulus)
    {
        // Only the Mersenne residue tracker calls this helper, and it never observes moduli below three,
        // so leave this guard documented but inactive.
        // if (modulus <= UInt128.One)
        // {
        //     return UInt128.Zero;
        // }

        if (value == UInt128.Zero)
        {
            // Wrapping occurs when the prior AddMod call produced one, so step back to modulus - 1.
            return modulus - UInt128.One;
        }

        return value - UInt128.One;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static UInt128 Pow(this UInt128 value, ulong exponent)
    {
        UInt128 result = UInt128.One;
        while (exponent != 0UL)
        {
            if ((exponent & 1UL) != 0UL)
            {
                result *= value;
            }

            value *= value;
            exponent >>= 1;
        }

        return result;
    }
}
