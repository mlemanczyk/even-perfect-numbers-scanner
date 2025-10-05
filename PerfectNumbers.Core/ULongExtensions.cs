using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

public static class ULongExtensions
{
    private const int Pow2WindowSize = 8;
    private const ulong Pow2WindowFallbackThreshold = 32UL;
    private const int Pow2WindowOddPowerCount = 1 << (Pow2WindowSize - 1);

    private struct Pow2OddPowerTable
    {
        public ulong Element0;
        public ulong Element1;
        public ulong Element2;
        public ulong Element3;
        public ulong Element4;
        public ulong Element5;
        public ulong Element6;
        public ulong Element7;
        public ulong Element8;
        public ulong Element9;
        public ulong Element10;
        public ulong Element11;
        public ulong Element12;
        public ulong Element13;
        public ulong Element14;
        public ulong Element15;
        public ulong Element16;
        public ulong Element17;
        public ulong Element18;
        public ulong Element19;
        public ulong Element20;
        public ulong Element21;
        public ulong Element22;
        public ulong Element23;
        public ulong Element24;
        public ulong Element25;
        public ulong Element26;
        public ulong Element27;
        public ulong Element28;
        public ulong Element29;
        public ulong Element30;
        public ulong Element31;
        public ulong Element32;
        public ulong Element33;
        public ulong Element34;
        public ulong Element35;
        public ulong Element36;
        public ulong Element37;
        public ulong Element38;
        public ulong Element39;
        public ulong Element40;
        public ulong Element41;
        public ulong Element42;
        public ulong Element43;
        public ulong Element44;
        public ulong Element45;
        public ulong Element46;
        public ulong Element47;
        public ulong Element48;
        public ulong Element49;
        public ulong Element50;
        public ulong Element51;
        public ulong Element52;
        public ulong Element53;
        public ulong Element54;
        public ulong Element55;
        public ulong Element56;
        public ulong Element57;
        public ulong Element58;
        public ulong Element59;
        public ulong Element60;
        public ulong Element61;
        public ulong Element62;
        public ulong Element63;
        public ulong Element64;
        public ulong Element65;
        public ulong Element66;
        public ulong Element67;
        public ulong Element68;
        public ulong Element69;
        public ulong Element70;
        public ulong Element71;
        public ulong Element72;
        public ulong Element73;
        public ulong Element74;
        public ulong Element75;
        public ulong Element76;
        public ulong Element77;
        public ulong Element78;
        public ulong Element79;
        public ulong Element80;
        public ulong Element81;
        public ulong Element82;
        public ulong Element83;
        public ulong Element84;
        public ulong Element85;
        public ulong Element86;
        public ulong Element87;
        public ulong Element88;
        public ulong Element89;
        public ulong Element90;
        public ulong Element91;
        public ulong Element92;
        public ulong Element93;
        public ulong Element94;
        public ulong Element95;
        public ulong Element96;
        public ulong Element97;
        public ulong Element98;
        public ulong Element99;
        public ulong Element100;
        public ulong Element101;
        public ulong Element102;
        public ulong Element103;
        public ulong Element104;
        public ulong Element105;
        public ulong Element106;
        public ulong Element107;
        public ulong Element108;
        public ulong Element109;
        public ulong Element110;
        public ulong Element111;
        public ulong Element112;
        public ulong Element113;
        public ulong Element114;
        public ulong Element115;
        public ulong Element116;
        public ulong Element117;
        public ulong Element118;
        public ulong Element119;
        public ulong Element120;
        public ulong Element121;
        public ulong Element122;
        public ulong Element123;
        public ulong Element124;
        public ulong Element125;
        public ulong Element126;
        public ulong Element127;

        public ref ulong this[int index]
        {
            get
            {
                return ref GetElementRef(index);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private ref ulong GetElementRef(int index)
        {
            switch (index)
            {
                case 0:
                    return ref Element0;
                case 1:
                    return ref Element1;
                case 2:
                    return ref Element2;
                case 3:
                    return ref Element3;
                case 4:
                    return ref Element4;
                case 5:
                    return ref Element5;
                case 6:
                    return ref Element6;
                case 7:
                    return ref Element7;
                case 8:
                    return ref Element8;
                case 9:
                    return ref Element9;
                case 10:
                    return ref Element10;
                case 11:
                    return ref Element11;
                case 12:
                    return ref Element12;
                case 13:
                    return ref Element13;
                case 14:
                    return ref Element14;
                case 15:
                    return ref Element15;
                case 16:
                    return ref Element16;
                case 17:
                    return ref Element17;
                case 18:
                    return ref Element18;
                case 19:
                    return ref Element19;
                case 20:
                    return ref Element20;
                case 21:
                    return ref Element21;
                case 22:
                    return ref Element22;
                case 23:
                    return ref Element23;
                case 24:
                    return ref Element24;
                case 25:
                    return ref Element25;
                case 26:
                    return ref Element26;
                case 27:
                    return ref Element27;
                case 28:
                    return ref Element28;
                case 29:
                    return ref Element29;
                case 30:
                    return ref Element30;
                case 31:
                    return ref Element31;
                case 32:
                    return ref Element32;
                case 33:
                    return ref Element33;
                case 34:
                    return ref Element34;
                case 35:
                    return ref Element35;
                case 36:
                    return ref Element36;
                case 37:
                    return ref Element37;
                case 38:
                    return ref Element38;
                case 39:
                    return ref Element39;
                case 40:
                    return ref Element40;
                case 41:
                    return ref Element41;
                case 42:
                    return ref Element42;
                case 43:
                    return ref Element43;
                case 44:
                    return ref Element44;
                case 45:
                    return ref Element45;
                case 46:
                    return ref Element46;
                case 47:
                    return ref Element47;
                case 48:
                    return ref Element48;
                case 49:
                    return ref Element49;
                case 50:
                    return ref Element50;
                case 51:
                    return ref Element51;
                case 52:
                    return ref Element52;
                case 53:
                    return ref Element53;
                case 54:
                    return ref Element54;
                case 55:
                    return ref Element55;
                case 56:
                    return ref Element56;
                case 57:
                    return ref Element57;
                case 58:
                    return ref Element58;
                case 59:
                    return ref Element59;
                case 60:
                    return ref Element60;
                case 61:
                    return ref Element61;
                case 62:
                    return ref Element62;
                case 63:
                    return ref Element63;
                case 64:
                    return ref Element64;
                case 65:
                    return ref Element65;
                case 66:
                    return ref Element66;
                case 67:
                    return ref Element67;
                case 68:
                    return ref Element68;
                case 69:
                    return ref Element69;
                case 70:
                    return ref Element70;
                case 71:
                    return ref Element71;
                case 72:
                    return ref Element72;
                case 73:
                    return ref Element73;
                case 74:
                    return ref Element74;
                case 75:
                    return ref Element75;
                case 76:
                    return ref Element76;
                case 77:
                    return ref Element77;
                case 78:
                    return ref Element78;
                case 79:
                    return ref Element79;
                case 80:
                    return ref Element80;
                case 81:
                    return ref Element81;
                case 82:
                    return ref Element82;
                case 83:
                    return ref Element83;
                case 84:
                    return ref Element84;
                case 85:
                    return ref Element85;
                case 86:
                    return ref Element86;
                case 87:
                    return ref Element87;
                case 88:
                    return ref Element88;
                case 89:
                    return ref Element89;
                case 90:
                    return ref Element90;
                case 91:
                    return ref Element91;
                case 92:
                    return ref Element92;
                case 93:
                    return ref Element93;
                case 94:
                    return ref Element94;
                case 95:
                    return ref Element95;
                case 96:
                    return ref Element96;
                case 97:
                    return ref Element97;
                case 98:
                    return ref Element98;
                case 99:
                    return ref Element99;
                case 100:
                    return ref Element100;
                case 101:
                    return ref Element101;
                case 102:
                    return ref Element102;
                case 103:
                    return ref Element103;
                case 104:
                    return ref Element104;
                case 105:
                    return ref Element105;
                case 106:
                    return ref Element106;
                case 107:
                    return ref Element107;
                case 108:
                    return ref Element108;
                case 109:
                    return ref Element109;
                case 110:
                    return ref Element110;
                case 111:
                    return ref Element111;
                case 112:
                    return ref Element112;
                case 113:
                    return ref Element113;
                case 114:
                    return ref Element114;
                case 115:
                    return ref Element115;
                case 116:
                    return ref Element116;
                case 117:
                    return ref Element117;
                case 118:
                    return ref Element118;
                case 119:
                    return ref Element119;
                case 120:
                    return ref Element120;
                case 121:
                    return ref Element121;
                case 122:
                    return ref Element122;
                case 123:
                    return ref Element123;
                case 124:
                    return ref Element124;
                case 125:
                    return ref Element125;
                case 126:
                    return ref Element126;
                case 127:
                    return ref Element127;
                default:
                    return ref Element0;
            }
        }
    }

    public static ulong CalculateOrder(this ulong q)
    {
        if (q <= 2UL)
        {
            return 0UL;
        }

        ulong order = q - 1UL, prime, temp;
        uint[] smallPrimes = PrimesGenerator.SmallPrimes;
        ulong[] smallPrimesPow2 = PrimesGenerator.SmallPrimesPow2;

        int i = 0, primesLength = smallPrimes.Length;
        UInt128 q128 = q,
                cycle = MersenneDivisorCycles.GetCycle(q128);
        // TODO: When the shared cycle snapshot cannot serve this divisor, trigger an on-demand
        // GPU computation (respecting the configured device) without promoting the result into
        // the cache so the order calculator still benefits from cycle stepping while keeping the
        // single-block memory plan intact.

        for (; i < primesLength; i++)
        {
            if (smallPrimesPow2[i] > order)
            {
                break;
            }

            prime = smallPrimes[i];
            // TODO: Replace this `%` driven factor peeling with the divisor-cycle aware
            // factoring helper so large orders reuse the cached remainders highlighted in
            // the latest divisor-cycle benchmarks instead of recomputing slow modulo checks.
            while (order % prime == 0UL)
            {
                temp = order / prime;
                if (temp.PowModWithCycle(q128, cycle) == UInt128.One)
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
    public static int GetBitLength(this ulong value)
    {
        return 64 - BitOperations.LeadingZeroCount(value);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong MultiplyShiftRight(ulong value, ulong multiplier, int shift)
    {
        UInt128 product = (UInt128)value * multiplier;
        return (ulong)(product >> shift);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong MultiplyShiftRightShiftFirst(ulong value, ulong multiplier, int shift)
    {
        ulong high = value >> shift;
        ulong mask = (1UL << shift) - 1UL;
        ulong low = value & mask;

        UInt128 highContribution = (UInt128)high * multiplier;
        UInt128 lowContribution = (UInt128)low * multiplier;

        UInt128 combined = highContribution + (lowContribution >> shift);
        return (ulong)combined;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong FastDiv64(this ulong value, ulong divisor, ulong mul)
    {
        ulong quotient = (ulong)(((UInt128)value * mul) >> 64);
        UInt128 remainder = (UInt128)value - ((UInt128)quotient * divisor);
        if (remainder >= divisor)
        {
            quotient++;
        }

        return quotient;
    }

    public const ulong WordBitMask = 0xFFFFUL;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsPrimeCandidate(this ulong n)
    {
        int i = 0;
        uint[] smallPrimes = PrimesGenerator.SmallPrimes;
        ulong[] smallPrimesPow2 = PrimesGenerator.SmallPrimesPow2;
        int len = smallPrimes.Length;
        ulong p;
        for (; i < len; i++)
        {
            if (smallPrimesPow2[i] > n)
            {
                break;
            }

            p = smallPrimes[i];
            // TODO: Swap this modulo check for the shared small-prime cycle filter once the
            // divisor-cycle cache is mandatory, matching the PrimeTester improvements noted in
            // the CPU sieve benchmarks.
            if ((n % p) == 0UL)
            {
                return n == p;
            }
        }

        return true;
    }

    // Benchmarks (Mod5ULongBenchmarks) show the direct `% 5` is still cheaper (~0.26 ns vs 0.43 ns), so keep the modulo until a faster lookup is proven.
    // (Mod8/Mod10 stay masked because they win; Mod5 currently does not.)
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Mod10(this ulong value) => (value & 1UL) == 0UL
            ? (value % 5UL) switch
            {
                0UL => 0UL,
                1UL => 6UL,
                2UL => 2UL,
                3UL => 8UL,
                _ => 4UL,
            }
            : (value % 5UL) switch
            {
                0UL => 5UL,
                1UL => 1UL,
                2UL => 7UL,
                3UL => 3UL,
                _ => 9UL,
            };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Mod128(this ulong value) => value & 127UL;

    // Benchmarks confirm `%` beats our current Mod5/Mod3 helpers for 64-bit inputs, so leave these modulo operations in place until a superior lookup is available.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Mod10_8_5_3(this ulong value, out ulong mod10, out ulong mod8, out ulong mod5, out ulong mod3)
    {
        mod8 = value & 7UL;
        // Benchmarks show `%` remains faster for the Mod5/Mod3 pair on ulong, so we keep the modulo path here for now.
        mod5 = value % 5UL;
        mod3 = value % 3UL;

        mod10 = (mod8 & 1UL) == 0UL
            ? mod5 switch
            {
                0UL => 0UL,
                1UL => 6UL,
                2UL => 2UL,
                3UL => 8UL,
                _ => 4UL,
            }
            : mod5 switch
            {
                0UL => 5UL,
                1UL => 1UL,
                2UL => 7UL,
                3UL => 3UL,
                _ => 9UL,
            };
    }

    // Mod5/Mod3 lookup tables are currently slower on 64-bit operands; keep the direct modulo until benchmarks flip.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Mod10_8_5_3Steps(this ulong value, out ulong step10, out ulong step8, out ulong step5, out ulong step3)
    {
        ulong mod8 = value & 7UL;
        // Same rationale: `%` wins in Mod5/Mod3 benches today, so avoid swapping until a faster lookup exists.
        ulong mod5 = value % 5UL;
        ulong mod3 = value % 3UL;
        ulong mod10 = (mod8 & 1UL) == 0UL
            ? mod5 switch
            {
                0UL => 0UL,
                1UL => 6UL,
                2UL => 2UL,
                3UL => 8UL,
                _ => 4UL,
            }
            : mod5 switch
            {
                0UL => 5UL,
                1UL => 1UL,
                2UL => 7UL,
                3UL => 3UL,
                _ => 9UL,
            };

        step10 = mod10 + mod10;
        if (step10 >= 10UL)
        {
            step10 -= 10UL;
        }

        step8 = (mod8 + mod8) & 7UL;

        step5 = mod5 + mod5;
        if (step5 >= 5UL)
        {
            step5 -= 5UL;
        }

        step3 = mod3 + mod3;
        if (step3 >= 3UL)
        {
            step3 -= 3UL;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static UInt128 Mul64(this ulong a, ulong b) => ((UInt128)a.MulHigh(b) << 64) | (UInt128)(a * b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong MulHigh(this ulong x, ulong y)
    {
        // TODO: Investigate replacing this manual decomposition with the UInt128-based implementation
        // for CPU callers; the latest benchmarks show the intrinsic path is an order of magnitude
        // faster, while GPU code can keep using GpuUInt128.MulHigh.
        ulong xLow = (uint)x;
        ulong xHigh = x >> 32;
        ulong yLow = (uint)y;
        ulong yHigh = y >> 32;

        ulong w1 = xLow * yHigh;
        ulong w2 = xHigh * yLow;
        ulong w3 = xLow * yLow;

        // Matching the layout used in GpuUInt128.MulHigh: introducing the
        // intermediate result looks like one extra store, but it lets RyuJIT keep
        // the accumulated high word entirely in registers. Without this explicit
        // local the JIT spills the partial sum, which is where the performance
        // regression in the benchmarks came from.
        ulong result = (xHigh * yHigh) + (w1 >> 32) + (w2 >> 32);
        result += ((w3 >> 32) + (uint)w1 + (uint)w2) >> 32;
        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong MulHighGpuCompatible(this ulong x, ulong y)
    {
        GpuUInt128 product = new(x);
        product.Mul64(new GpuUInt128(y));
        return product.High;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong ModPow64(this ulong value, ulong exponent, ulong modulus)
    {
        ulong result = 1UL;
        // TODO: Replace this `%` with the Montgomery folding helper highlighted in MulMod64Benchmarks so the
        // modular exponentiation avoids the slow integer division before the ladder even starts.
        value %= modulus;

        while (exponent != 0UL)
        {
            if ((exponent & 1UL) != 0UL)
            {
                result = MulMod64(result, value, modulus);
            }

            value = MulMod64(value, value, modulus);
            exponent >>= 1;
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    // TODO: Replace this fallback with the UInt128 Montgomery helper measured fastest in
    // MulMod64Benchmarks so CPU callers stop paying for triple modulo operations.
    public static ulong MulMod64(this ulong a, ulong b, ulong modulus) => (ulong)(UInt128)(((a % modulus) * (b % modulus)) % modulus);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong MulMod64GpuCompatible(this ulong a, ulong b, ulong modulus)
    {
        // TODO: Remove this GPU-compatible shim from production once callers migrate to MulMod64,
        // which the benchmarks show is roughly 6-7× faster on dense 64-bit inputs.
        GpuUInt128 state = new(a % modulus);
        return state.MulMod(b, modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong MulMod64GpuCompatibleDeferred(this ulong a, ulong b, ulong modulus)
    {
        // TODO: Move this deferred helper to the benchmark suite; the baseline MulMod64 avoids the
        // 5-40× slowdown seen across real-world operand distributions.
        GpuUInt128 state = new(a);
        return state.MulModWithNativeModulo(b, modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong MontgomeryMultiply(this ulong a, ulong b, ulong modulus, ulong nPrime)
    {
        ulong tLow = unchecked(a * b);
        ulong m = unchecked(tLow * nPrime);
        ulong mTimesModulusLow = unchecked(m * modulus);

        ulong result = unchecked(a.MulHigh(b) + m.MulHigh(modulus) + (unchecked(tLow + mTimesModulusLow) < tLow ? 1UL : 0UL));
        if (result >= modulus)
        {
            result -= modulus;
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Pow2MontgomeryModWindowed(this ulong exponent, in MontgomeryDivisorData divisor, bool keepMontgomery)
    {
        ulong modulus = divisor.Modulus;
        if (exponent == 0UL)
        {
            return keepMontgomery ? divisor.MontgomeryOne : 1UL % modulus;
        }

        if (exponent <= Pow2WindowFallbackThreshold)
        {
            return Pow2MontgomeryModSingleBit(exponent, divisor, keepMontgomery);
        }

        int bitLength = BitOperations.Log2(exponent) + 1;
        int windowSize = GetWindowSize(bitLength);
        int oddPowerCount = 1 << (windowSize - 1);

        ulong result = divisor.MontgomeryOne;
        ulong nPrime = divisor.NPrime;
        Pow2OddPowerTable oddPowers = default;
        InitializeMontgomeryOddPowers(divisor, modulus, nPrime, ref oddPowers, oddPowerCount);

        int index = bitLength - 1;
        while (index >= 0)
        {
            if (((exponent >> index) & 1UL) == 0UL)
            {
                result = result.MontgomeryMultiply(result, modulus, nPrime);
                index--;
                continue;
            }

            int windowStart = index - windowSize + 1;
            if (windowStart < 0)
            {
                windowStart = 0;
            }

            while (((exponent >> windowStart) & 1UL) == 0UL)
            {
                windowStart++;
            }

            int windowLength = index - windowStart + 1;
            for (int square = 0; square < windowLength; square++)
            {
                result = result.MontgomeryMultiply(result, modulus, nPrime);
            }

            ulong mask = (1UL << windowLength) - 1UL;
            ulong windowValue = (exponent >> windowStart) & mask;
            int tableIndex = (int)((windowValue - 1UL) >> 1);
            ulong multiplier = oddPowers[tableIndex];
            result = result.MontgomeryMultiply(multiplier, modulus, nPrime);

            index = windowStart - 1;
        }

        if (keepMontgomery)
        {
            return result;
        }

        return result.MontgomeryMultiply(1UL, modulus, nPrime);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Pow2MontgomeryModSingleBit(ulong exponent, in MontgomeryDivisorData divisor, bool keepMontgomery)
    {
        ulong modulus = divisor.Modulus;
        ulong nPrime = divisor.NPrime;
        ulong result = divisor.MontgomeryOne;
        ulong baseVal = divisor.MontgomeryTwo;
        ulong remainingExponent = exponent;

        while (remainingExponent != 0UL)
        {
            if ((remainingExponent & 1UL) != 0UL)
            {
                result = result.MontgomeryMultiply(baseVal, modulus, nPrime);
            }

            remainingExponent >>= 1;
            if (remainingExponent == 0UL)
            {
                break;
            }

            baseVal = baseVal.MontgomeryMultiply(baseVal, modulus, nPrime);
        }

        return keepMontgomery ? result : result.MontgomeryMultiply(1UL, modulus, nPrime);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int GetWindowSize(int bitLength)
    {
        if (bitLength <= 6)
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
    private static void InitializeMontgomeryOddPowers(in MontgomeryDivisorData divisor, ulong modulus, ulong nPrime, ref Pow2OddPowerTable oddPowers, int oddPowerCount)
    {
        oddPowers[0] = divisor.MontgomeryTwo;
        if (oddPowerCount == 1)
        {
            return;
        }

        ulong square = divisor.MontgomeryTwoSquared;
        for (int i = 1; i < oddPowerCount; i++)
        {
            ulong previous = oddPowers[i - 1];
            oddPowers[i] = previous.MontgomeryMultiply(square, modulus, nPrime);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Pow2MontgomeryModWithCycle(this ulong exponent, ulong cycleLength, in MontgomeryDivisorData divisor)
    {
        ulong rotationCount = exponent % cycleLength;
        return Pow2MontgomeryModWindowed(rotationCount, divisor, keepMontgomery: false);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Pow2MontgomeryModFromCycleRemainder(this ulong reducedExponent, in MontgomeryDivisorData divisor)
    {
        return Pow2MontgomeryModWindowed(reducedExponent, divisor, keepMontgomery: false);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static UInt128 PowMod(this ulong exponent, UInt128 modulus)
    {
        UInt128 result = UInt128.One;
        ulong exponentLoopIndex = 0UL;

        // TODO: Port this scalar PowMod fallback to the ProcessEightBitWindows helper so CPU callers get the
        // eight-bit window wins measured against the classic square-and-subtract implementation.
        // Return 1 because 2^0 = 1
        if (exponent == 0UL)
            return result;

        // Any number mod 1 is 0
        if (modulus == UInt128.One)
            return UInt128.Zero;

        // For small exponents, do classic method
        if (exponent < 64 || modulus < 4)
        {
            for (; exponentLoopIndex < exponent; exponentLoopIndex++)
            {
                result <<= 1;
                if (result >= modulus)
                    result -= modulus;
            }

            return result;
        }

        // Special case: if modulus is a power of two, use bitmasking for efficiency and correctness
        if ((modulus & (modulus - 1)) == 0)
        {
            result = UInt128.One << (int)(exponent & 127);
            return result & (modulus - 1);
        }

        // Reusing exponentLoopIndex to iterate again over the exponent range for the general-case accumulation.
        exponentLoopIndex = 0UL;
        // Reusing result after resetting it for the general modulus accumulation phase.
        result = UInt128.One;
        for (; exponentLoopIndex < exponent; exponentLoopIndex++)
        {
            result <<= 1;
            if (result >= modulus)
                result -= modulus;
        }

        return result;
    }

    /// <summary>
    /// Computes 2^exponent mod modulus using a known cycle length.
    /// </summary>
    public static UInt128 PowModWithCycle(this ulong exponent, UInt128 modulus, ulong cycleLength)
    {
        UInt128 result = UInt128.One;
        ulong exponentLoopIndex = 0UL;

        // TODO: Wire this cycle-aware overload into the ProcessEightBitWindows helper so the reduced exponent path
        // inherits the faster windowed pow2 routine highlighted in the Pow2Montgomery benchmarks.
        // Return 1 because 2^0 = 1
        if (exponent == 0UL)
            return result;

        // Any number mod 1 is 0
        if (modulus == UInt128.One)
            return UInt128.Zero;

        // For small exponents, do classic method
        if (exponent < 64 || modulus < 4)
        {
            for (; exponentLoopIndex < exponent; exponentLoopIndex++)
            {
                result <<= 1;
                if (result >= modulus)
                    result -= modulus;
            }

            return result;
        }

        // Special case: if modulus is a power of two, use bitmasking for efficiency and correctness
        if ((modulus & (modulus - 1)) == 0)
        {
            result = UInt128.One << (int)(exponent & 127);
            return result & (modulus - 1);
        }

        // Reusing exponentLoopIndex to iterate over the rotation count for the cycle-aware path.
        exponentLoopIndex = 0UL;
        // Reusing result after resetting it for the rotation-accumulation pass.
        result = UInt128.One;
        // TODO: Replace this modulo with the cached cycle remainder produced by the divisor-cycle cache so PowModWithCycle avoids
        // repeated `%` work, matching the ProcessEightBitWindows wins captured in Pow2MontgomeryModCycleComputationBenchmarks.
        ulong rotationCount = exponent % cycleLength;
        for (; exponentLoopIndex < rotationCount; exponentLoopIndex++)
        {
            result <<= 1;
            if (result >= modulus)
                result -= modulus;
        }

        return result;
    }

    /// <summary>
    /// Computes 2^exponent mod modulus using a known cycle length.
    /// </summary>
    public static UInt128 PowModWithCycle(this ulong exponent, UInt128 modulus, UInt128 cycleLength)
    {
        UInt128 result = UInt128.One;
        ulong exponentLoopIndex = 0UL;

        // TODO: Replace this UInt128-cycle overload with the ProcessEightBitWindows helper so large-exponent CPU scans
        // reuse the faster windowed pow2 ladder instead of the manual rotation loop measured to lag behind in benchmarks.
        // Return 1 because 2^0 = 1
        if (exponent == UInt128.Zero)
            return result;

        // Any number mod 1 is 0
        if (modulus == UInt128.One)
            return UInt128.Zero;

        // For small exponents, do classic method
        if (exponent < UInt128Numbers.SixtyFour || modulus < UInt128Numbers.Four)
        {
            for (; exponentLoopIndex < exponent; exponentLoopIndex++)
            {
                result <<= 1;
                if (result >= modulus)
                    result -= modulus;
            }

            return result;
        }

        // Special case: if modulus is a power of two, use bitmasking for efficiency and correctness
        if ((modulus & (modulus - UInt128.One)) == UInt128.Zero)
        {
            result = UInt128.One << (int)(exponent & UInt128Numbers.OneHundredTwentySeven);
            return result & (modulus - 1);
        }

        // Reusing result after resetting it for the rotation-driven accumulation phase.
        result = UInt128.One;
        // TODO: Swap this modulo with the upcoming UInt128 cycle remainder helper so large-exponent scans reuse cached
        // reductions instead of recomputing `%` for every lookup, as demonstrated in Pow2MontgomeryModCycleComputationBenchmarks.
        UInt128 rotationCount = exponent % cycleLength;
        UInt128 rotationIndex = UInt128.Zero;
        while (rotationIndex < rotationCount)
        {
            result <<= 1;
            if (result >= modulus)
                result -= modulus;

            rotationIndex += UInt128.One;
        }

        return result;
    }

    /// <summary>
    /// Computes 2^exponent mod modulus using a known cycle length.
    /// </summary>
    public static UInt128 PowModWithCycle(this UInt128 exponent, UInt128 modulus, UInt128 cycleLength)
    {
        UInt128 one = UInt128.One,
                result = one,
                zero = UInt128.Zero;
        ulong exponentLoopIndex = 0UL;

        // TODO: Migrate this UInt128 exponent overload to ProcessEightBitWindows so the large-cycle reductions drop the
        // slow manual loop that underperforms the windowed pow2 helper in the Pow2 benchmark suite.
        // Return 1 because 2^0 = 1
        if (exponent == zero)
            return result;

        // Any number mod 1 is 0
        if (modulus == one)
            return zero;

        // For small exponents, do classic method
        if (exponent < UInt128Numbers.SixtyFour || modulus < UInt128Numbers.Four)
        {
            for (; exponentLoopIndex < exponent; exponentLoopIndex++)
            {
                result <<= 1;
                if (result >= modulus)
                    result -= modulus;
            }

            return result;
        }

        // Special case: if modulus is a power of two, use bitmasking for efficiency and correctness
        if ((modulus & (modulus - one)) == zero)
        {
            result = one << (int)(exponent & UInt128Numbers.OneHundredTwentySeven);
            return result & (modulus - 1);
        }

        // Reusing result after resetting it for the rotation-driven accumulation phase.
        result = one;
        // TODO: Swap this modulo with the shared UInt128 cycle remainder helper once available so CRT powmods reuse cached
        // reductions in the windowed ladder, avoiding the `%` cost highlighted in Pow2MontgomeryModCycleComputationBenchmarks.
        UInt128 rotationCount = exponent % cycleLength;

        // We're reusing "zero" as rotation index for just a little better performance
        while (zero < rotationCount)
        {
            result <<= 1;
            if (result >= modulus)
                result -= modulus;

            zero += one;
        }

        return result;
    }

    /// <summary>
    /// Computes 2^exponent mod modulus using iterative CRT composition from mod 10 up to modulus.
    /// Only for modulus >= 10 and reasonable size.
    /// </summary>
    public static UInt128 PowModCrt(this ulong exponent, UInt128 modulus, MersenneDivisorCycles cycles)
    {
        if (modulus < 10)
            return PowMod(exponent, modulus); // fallback to classic

        // Use cycle length 4 for mod 10
        UInt128 currentModulus = 10,
                cycle,
                modulusCandidate = 11,
                remainderForCandidate,
                result = PowModWithCycle(exponent, 10, 4),
                zero = UInt128.Zero;

        for (; modulusCandidate <= modulus; modulusCandidate++)
        {
            cycle = MersenneDivisorCycles.GetCycle(modulusCandidate);
            remainderForCandidate = cycle > zero
                    ? PowModWithCycle(exponent, modulusCandidate, cycle)
                    : PowMod(exponent, modulusCandidate);

            // Solve x ≡ result mod currentModulus
            //      x ≡ remM   mod m
            // Find x mod (currentModulus * m)
            // Since currentModulus and m are coprime, use CRT:
            // x = result + currentModulus * t, where t ≡ (remM - result) * inv(currentModulus, m) mod m

            // TODO: Replace this `% modulusCandidate` with the cached residue helper derived from Mod10_8_5_3Benchmarks so CRT
            // composition avoids repeated modulo divisions when combining residues for large divisor sets.
            result += currentModulus * ((remainderForCandidate + modulusCandidate - (result % modulusCandidate)) * ModInverse(currentModulus, modulusCandidate) % modulusCandidate);
            currentModulus *= modulusCandidate;

            if (currentModulus >= modulus)
                break;
        }

        // TODO: Swap this final `% modulus` with the pooled remainder cache so the CRT result write-back avoids one more division,
        // aligning with the optimizations captured in Mod10_8_5_3Benchmarks.
        return result % modulus;
    }

    // Helper: modular inverse (extended Euclidean algorithm)
    private static UInt128 ModInverse(UInt128 a, UInt128 m)
    {
        UInt128 m0 = m,
                originalA,
                originalM,
                temp,
                x0 = 0,
                x1 = 1;

        if (m == 1)
        {
            return 0;
        }

        while (a > 1)
        {
            originalA = a;
            originalM = m;
            m = originalA % originalM;
            a = originalM;
            temp = x0;
            x0 = x1 - (originalA / originalM) * x0;
            x1 = temp;
        }

        if (x1 < 0)
        {
            x1 += m0;
        }

        return x1;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool SharesFactorWithExponentMinusOne(this ulong exponent)
    {
        ulong prime, value = exponent - 1UL;
        value >>= BitOperations.TrailingZeroCount(value);
        uint[] smallPrimes = PrimesGenerator.SmallPrimes;
        ulong[] smallPrimesPow2 = PrimesGenerator.SmallPrimesPow2;
        int i = 1, smallPrimesLength = smallPrimes.Length;

        for (; i < smallPrimesLength && smallPrimesPow2[i] <= value; i++)
        {
            prime = smallPrimes[i];
            if (value % prime != 0UL)
            {
                continue;
            }

            if (exponent % prime.CalculateOrder() == 0UL)
            {
                return true;
            }

            do
            {
                value /= prime;
            }
            while (value % prime == 0UL);
        }

        if (value > 1UL && exponent % value.CalculateOrder() == 0UL)
        {
            return true;
        }

        return false;
    }
}
