using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using ILGPU.Algorithms;
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

        public ulong this[int index]
        {
            get
            {
                return GetElement(index);
            }

            set
            {
                SetElement(index, value);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private ulong GetElement(int index) => index switch
        {
            0 => Element0,
            1 => Element1,
            2 => Element2,
            3 => Element3,
            4 => Element4,
            5 => Element5,
            6 => Element6,
            7 => Element7,
            8 => Element8,
            9 => Element9,
            10 => Element10,
            11 => Element11,
            12 => Element12,
            13 => Element13,
            14 => Element14,
            15 => Element15,
            16 => Element16,
            17 => Element17,
            18 => Element18,
            19 => Element19,
            20 => Element20,
            21 => Element21,
            22 => Element22,
            23 => Element23,
            24 => Element24,
            25 => Element25,
            26 => Element26,
            27 => Element27,
            28 => Element28,
            29 => Element29,
            30 => Element30,
            31 => Element31,
            32 => Element32,
            33 => Element33,
            34 => Element34,
            35 => Element35,
            36 => Element36,
            37 => Element37,
            38 => Element38,
            39 => Element39,
            40 => Element40,
            41 => Element41,
            42 => Element42,
            43 => Element43,
            44 => Element44,
            45 => Element45,
            46 => Element46,
            47 => Element47,
            48 => Element48,
            49 => Element49,
            50 => Element50,
            51 => Element51,
            52 => Element52,
            53 => Element53,
            54 => Element54,
            55 => Element55,
            56 => Element56,
            57 => Element57,
            58 => Element58,
            59 => Element59,
            60 => Element60,
            61 => Element61,
            62 => Element62,
            63 => Element63,
            64 => Element64,
            65 => Element65,
            66 => Element66,
            67 => Element67,
            68 => Element68,
            69 => Element69,
            70 => Element70,
            71 => Element71,
            72 => Element72,
            73 => Element73,
            74 => Element74,
            75 => Element75,
            76 => Element76,
            77 => Element77,
            78 => Element78,
            79 => Element79,
            80 => Element80,
            81 => Element81,
            82 => Element82,
            83 => Element83,
            84 => Element84,
            85 => Element85,
            86 => Element86,
            87 => Element87,
            88 => Element88,
            89 => Element89,
            90 => Element90,
            91 => Element91,
            92 => Element92,
            93 => Element93,
            94 => Element94,
            95 => Element95,
            96 => Element96,
            97 => Element97,
            98 => Element98,
            99 => Element99,
            100 => Element100,
            101 => Element101,
            102 => Element102,
            103 => Element103,
            104 => Element104,
            105 => Element105,
            106 => Element106,
            107 => Element107,
            108 => Element108,
            109 => Element109,
            110 => Element110,
            111 => Element111,
            112 => Element112,
            113 => Element113,
            114 => Element114,
            115 => Element115,
            116 => Element116,
            117 => Element117,
            118 => Element118,
            119 => Element119,
            120 => Element120,
            121 => Element121,
            122 => Element122,
            123 => Element123,
            124 => Element124,
            125 => Element125,
            126 => Element126,
            127 => Element127,
            _ => Element0,
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void SetElement(int index, ulong value)
        {
            switch (index)
            {
            case 0:
                Element0 = value;
                return;
            case 1:
                Element1 = value;
                return;
            case 2:
                Element2 = value;
                return;
            case 3:
                Element3 = value;
                return;
            case 4:
                Element4 = value;
                return;
            case 5:
                Element5 = value;
                return;
            case 6:
                Element6 = value;
                return;
            case 7:
                Element7 = value;
                return;
            case 8:
                Element8 = value;
                return;
            case 9:
                Element9 = value;
                return;
            case 10:
                Element10 = value;
                return;
            case 11:
                Element11 = value;
                return;
            case 12:
                Element12 = value;
                return;
            case 13:
                Element13 = value;
                return;
            case 14:
                Element14 = value;
                return;
            case 15:
                Element15 = value;
                return;
            case 16:
                Element16 = value;
                return;
            case 17:
                Element17 = value;
                return;
            case 18:
                Element18 = value;
                return;
            case 19:
                Element19 = value;
                return;
            case 20:
                Element20 = value;
                return;
            case 21:
                Element21 = value;
                return;
            case 22:
                Element22 = value;
                return;
            case 23:
                Element23 = value;
                return;
            case 24:
                Element24 = value;
                return;
            case 25:
                Element25 = value;
                return;
            case 26:
                Element26 = value;
                return;
            case 27:
                Element27 = value;
                return;
            case 28:
                Element28 = value;
                return;
            case 29:
                Element29 = value;
                return;
            case 30:
                Element30 = value;
                return;
            case 31:
                Element31 = value;
                return;
            case 32:
                Element32 = value;
                return;
            case 33:
                Element33 = value;
                return;
            case 34:
                Element34 = value;
                return;
            case 35:
                Element35 = value;
                return;
            case 36:
                Element36 = value;
                return;
            case 37:
                Element37 = value;
                return;
            case 38:
                Element38 = value;
                return;
            case 39:
                Element39 = value;
                return;
            case 40:
                Element40 = value;
                return;
            case 41:
                Element41 = value;
                return;
            case 42:
                Element42 = value;
                return;
            case 43:
                Element43 = value;
                return;
            case 44:
                Element44 = value;
                return;
            case 45:
                Element45 = value;
                return;
            case 46:
                Element46 = value;
                return;
            case 47:
                Element47 = value;
                return;
            case 48:
                Element48 = value;
                return;
            case 49:
                Element49 = value;
                return;
            case 50:
                Element50 = value;
                return;
            case 51:
                Element51 = value;
                return;
            case 52:
                Element52 = value;
                return;
            case 53:
                Element53 = value;
                return;
            case 54:
                Element54 = value;
                return;
            case 55:
                Element55 = value;
                return;
            case 56:
                Element56 = value;
                return;
            case 57:
                Element57 = value;
                return;
            case 58:
                Element58 = value;
                return;
            case 59:
                Element59 = value;
                return;
            case 60:
                Element60 = value;
                return;
            case 61:
                Element61 = value;
                return;
            case 62:
                Element62 = value;
                return;
            case 63:
                Element63 = value;
                return;
            case 64:
                Element64 = value;
                return;
            case 65:
                Element65 = value;
                return;
            case 66:
                Element66 = value;
                return;
            case 67:
                Element67 = value;
                return;
            case 68:
                Element68 = value;
                return;
            case 69:
                Element69 = value;
                return;
            case 70:
                Element70 = value;
                return;
            case 71:
                Element71 = value;
                return;
            case 72:
                Element72 = value;
                return;
            case 73:
                Element73 = value;
                return;
            case 74:
                Element74 = value;
                return;
            case 75:
                Element75 = value;
                return;
            case 76:
                Element76 = value;
                return;
            case 77:
                Element77 = value;
                return;
            case 78:
                Element78 = value;
                return;
            case 79:
                Element79 = value;
                return;
            case 80:
                Element80 = value;
                return;
            case 81:
                Element81 = value;
                return;
            case 82:
                Element82 = value;
                return;
            case 83:
                Element83 = value;
                return;
            case 84:
                Element84 = value;
                return;
            case 85:
                Element85 = value;
                return;
            case 86:
                Element86 = value;
                return;
            case 87:
                Element87 = value;
                return;
            case 88:
                Element88 = value;
                return;
            case 89:
                Element89 = value;
                return;
            case 90:
                Element90 = value;
                return;
            case 91:
                Element91 = value;
                return;
            case 92:
                Element92 = value;
                return;
            case 93:
                Element93 = value;
                return;
            case 94:
                Element94 = value;
                return;
            case 95:
                Element95 = value;
                return;
            case 96:
                Element96 = value;
                return;
            case 97:
                Element97 = value;
                return;
            case 98:
                Element98 = value;
                return;
            case 99:
                Element99 = value;
                return;
            case 100:
                Element100 = value;
                return;
            case 101:
                Element101 = value;
                return;
            case 102:
                Element102 = value;
                return;
            case 103:
                Element103 = value;
                return;
            case 104:
                Element104 = value;
                return;
            case 105:
                Element105 = value;
                return;
            case 106:
                Element106 = value;
                return;
            case 107:
                Element107 = value;
                return;
            case 108:
                Element108 = value;
                return;
            case 109:
                Element109 = value;
                return;
            case 110:
                Element110 = value;
                return;
            case 111:
                Element111 = value;
                return;
            case 112:
                Element112 = value;
                return;
            case 113:
                Element113 = value;
                return;
            case 114:
                Element114 = value;
                return;
            case 115:
                Element115 = value;
                return;
            case 116:
                Element116 = value;
                return;
            case 117:
                Element117 = value;
                return;
            case 118:
                Element118 = value;
                return;
            case 119:
                Element119 = value;
                return;
            case 120:
                Element120 = value;
                return;
            case 121:
                Element121 = value;
                return;
            case 122:
                Element122 = value;
                return;
            case 123:
                Element123 = value;
                return;
            case 124:
                Element124 = value;
                return;
            case 125:
                Element125 = value;
                return;
            case 126:
                Element126 = value;
                return;
            case 127:
                Element127 = value;
                return;
            default:
                // ILGPU kernels cannot throw exceptions, and callers guarantee the index range.
                return;
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

        int bitLength = GetPortableBitLength(exponent);
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
    private static int GetPortableBitLength(ulong value)
    {
        if (value == 0UL)
        {
            return 0;
        }

        return 64 - XMath.LeadingZeroCount(value);
    }

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
