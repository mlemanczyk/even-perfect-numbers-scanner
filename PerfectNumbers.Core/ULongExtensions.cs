using System;
using System.Buffers;
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
    private const int Pow2WindowSizeGpu = 5;
    private const int Pow2WindowOddPowerCountGpu = 1 << (Pow2WindowSizeGpu - 1);

    public static ulong CalculateOrder(this ulong q)
    {
        // Order computations only run on admissible divisors, so the small-prime guard remains commented out to keep the
        // path branch-free.
        // if (q <= 2UL)
        // {
        //     return 0UL;
        // }

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
                if (temp.Pow2ModWindowed(q128, cycle) == UInt128.One)
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
    public static ulong MulMod64(this ulong a, ulong b, ulong modulus)
    {
        // Scanner inputs always provide odd prime moduli, so the legacy modulus guard stays commented out to avoid extra
        // branching.
        // if (modulus <= 1UL)
        // {
        //     return 0UL;
        // }

        ulong reducedA = a >= modulus ? a % modulus : a;
        ulong reducedB = b >= modulus ? b % modulus : b;
        GpuUInt128 state = new(reducedA);
        return state.MulMod(reducedB, modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Pow2ModWindowedCpu(this ulong exponent, ulong modulus)
    {
        // The by-divisor pipeline only schedules odd prime moduli, so the defensive check below
        // would never trigger on that path. Leaving it commented documents the invariant without
        // forcing a branch in the hot loop.
        // if (modulus <= 1UL)
        // {
        //     return 0UL;
        // }

        // Exponents flow from the strictly positive candidate list on the by-divisor scanner, so
        // this guard would be redundant there.
        // if (exponent == 0UL)
        // {
        //     return 1UL % modulus;
        // }

        // Even though the by-divisor pipeline only feeds very large exponents, the legacy diagnostic callers
        // still hit this small-exponent fast path, so keep the fallback in place.
        if (exponent <= Pow2WindowFallbackThreshold)
        {
            return Pow2ModBinaryFallback(exponent, modulus);
        }

        int bitLength = GetPortableBitLength(exponent);
        int windowSize = GetWindowSize(bitLength);
        int oddPowerCount = 1 << (windowSize - 1);

        Span<ulong> oddPowerStorage = oddPowerCount <= PerfectNumberConstants.MaxOddPowersCount
            ? stackalloc ulong[PerfectNumberConstants.MaxOddPowersCount]
            : new ulong[oddPowerCount];
        oddPowerStorage = oddPowerStorage[..oddPowerCount];
        InitializePlainOddPowers(modulus, oddPowerStorage);

        // With modulus guaranteed to exceed one on the scanning path, the modulo operation would
        // always yield one, so keep the literal to highlight the intended base value.
        ulong result = 1UL;
        int index = bitLength - 1;

        while (index >= 0)
        {
            if (((exponent >> index) & 1UL) == 0UL)
            {
                result = result.MulMod64(result, modulus);
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
                result = result.MulMod64(result, modulus);
            }

            ulong windowValue = (exponent >> windowStart) & ((1UL << windowLength) - 1UL);
            int tableIndex = (int)((windowValue - 1UL) >> 1);
            result = result.MulMod64(oddPowerStorage[tableIndex], modulus);

            index = windowStart - 1;
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Pow2ModBinaryFallback(ulong exponent, ulong modulus)
    {
        // The scanning path keeps modulus strictly greater than one, but other utility callers still
        // rely on this fallback for tiny exponents, so leave the modulo in place for safety.
        ulong remainingExponent = exponent;
        // Production moduli are always odd primes greater than two, so normalize the seed values directly.
        ulong result = 1UL;
        ulong baseValue = 2UL;

        while (remainingExponent != 0UL)
        {
            if ((remainingExponent & 1UL) != 0UL)
            {
                result = result.MulMod64(baseValue, modulus);
            }

            remainingExponent >>= 1;
            if (remainingExponent == 0UL)
            {
                break;
            }

            baseValue = baseValue.MulMod64(baseValue, modulus);
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Pow2ModWindowedGpu(this ulong exponent, ulong modulus)
    {
        // The GPU pipeline feeds only odd prime moduli, so the removed guard would never fire for
        // EvenPerfectBitScanner workloads.
        // if (modulus <= 1UL)
        // {
        //     return 0UL;
        // }

        // The GPU pipeline only emits positive exponents on production workloads, so skip the zero guard to avoid an extra branch.
        // Diagnostic harnesses must normalize their inputs before invoking this helper.
        // if (exponent == 0UL)
        // {
        //     return 1UL;
        // }

        // The GPU kernels mostly process large exponents as well, but the binary fallback still serves
        // diagnostic harnesses that probe edge cases around the threshold.
        if (exponent <= Pow2WindowFallbackThreshold)
        {
            return Pow2ModBinaryGpuFallback(exponent, modulus);
        }

        int bitLength = GetPortableBitLength(exponent);
        int windowSize = GetGpuWindowSize(bitLength);
        int oddPowerCount = 1 << (windowSize - 1);

        PlainOddPowerTableGpu oddPowers = new PlainOddPowerTableGpu(modulus, oddPowerCount);

        ulong result = 1UL;
        int index = bitLength - 1;

        while (index >= 0)
        {
            if (((exponent >> index) & 1UL) == 0UL)
            {
                result = result.MulMod64(result, modulus);
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
                result = result.MulMod64(result, modulus);
            }

            ulong windowValue = (exponent >> windowStart) & ((1UL << windowLength) - 1UL);
            int tableIndex = (int)((windowValue - 1UL) >> 1);
            result = result.MulMod64(oddPowers.Get(tableIndex), modulus);

            index = windowStart - 1;
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Pow2ModBinaryGpuFallback(ulong exponent, ulong modulus)
    {
        ulong remainingExponent = exponent;
        // Production moduli are always odd primes greater than two, so normalize the seed values directly.
        ulong result = 1UL;
        ulong baseValue = 2UL;

        while (remainingExponent != 0UL)
        {
            if ((remainingExponent & 1UL) != 0UL)
            {
                result = result.MulMod64(baseValue, modulus);
            }

            remainingExponent >>= 1;
            if (remainingExponent == 0UL)
            {
                break;
            }

            baseValue = baseValue.MulMod64(baseValue, modulus);
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Pow2ModWindowed(this ulong exponent, ulong modulus)
    {
        return Pow2ModWindowed(exponent, modulus, 0UL);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Pow2ModWindowed(this ulong exponent, ulong modulus, ulong cycleLength)
    {
        ulong rotation = exponent;
        if (cycleLength != 0UL)
        {
            rotation %= cycleLength;
        }

        // Reusing rotation to carry the cycle-reduced exponent avoids introducing another local.
        return Pow2ModWindowedInternal(rotation, modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Pow2ModWindowed(this UInt128 exponent, ulong modulus)
    {
        return exponent.Pow2ModWindowed(modulus, 0UL);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Pow2ModWindowed(this UInt128 exponent, ulong modulus, ulong cycleLength)
    {
        UInt128 rotation = exponent;
        if (cycleLength != 0UL)
        {
            rotation %= cycleLength;
        }

        // Reusing rotation to carry the cycle-reduced exponent avoids introducing another local.
        UInt128 reduced = rotation.Pow2ModWindowed((UInt128)modulus, UInt128.Zero);
        return (ulong)reduced;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Pow2ModWindowedInternal(ulong rotation, ulong modulus)
    {
        if (rotation == 0UL)
        {
            // Scanning moduli stay above one, so returning the literal keeps the path branch-free.
            return 1UL;
        }

        if (GpuContextPool.ForceCpu)
        {
            return Pow2ModWindowedCpu(rotation, modulus);
        }

        return Pow2ModWindowedGpu(rotation, modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static UInt128 Pow2ModWindowed(this ulong exponent, UInt128 modulus, UInt128 cycleLength)
    {
        UInt128 rotation = exponent;
        if (cycleLength > UInt128.Zero)
        {
            // Production divisor lookups supply non-zero cycle lengths during scanning, so this hot path keeps the reduced rotation cached.
            // TODO: Replace this modulo with the cached cycle remainder produced by the divisor-cycle cache so Pow2ModWindowed avoids repeated '%'.
            rotation %= cycleLength;
        }

        return Pow2ModWindowedInternal(rotation, modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static UInt128 Pow2ModWindowed(this UInt128 exponent, UInt128 modulus, UInt128 cycleLength)
    {
        UInt128 rotation = exponent;
        if (cycleLength > UInt128.Zero)
        {
            // CRT reductions for very large exponents also ship precomputed cycles, so this matches the CPU and GPU scanning flow.
            // TODO: Replace this modulo with the cached cycle remainder produced by the divisor-cycle cache so Pow2ModWindowed avoids repeated '%'.
            rotation %= cycleLength;
        }

        return Pow2ModWindowedInternal(rotation, modulus);
    }

    private static UInt128 Pow2ModWindowedInternal(UInt128 rotation, UInt128 modulus)
    {
        // Scanning divisors are always odd primes greater than two, so the modulus == 1 guard never triggers
        // on production paths. Leave the branch commented out to document the invariant while keeping the hot
        // path branch-free; validation tests and benchmarks can rely on the general implementation instead.
        // if (modulus == UInt128.One)
        // {
        //     return UInt128.Zero;
        // }

        if (rotation == UInt128.Zero)
        {
            // A zero rotation appears when the exponent shares the divisor cycle length, which happens intermittently during real scans.
            return UInt128.One;
        }

        if (rotation <= (UInt128)Pow2WindowFallbackThreshold)
        {
            if ((modulus >> 64) == UInt128.Zero)
            {
                // Small exponents over 64-bit moduli reuse the CPU ladder to mirror the historical fallback without incurring GPU dispatch.
                return Pow2ModWindowedCpu(rotation, modulus);
            }

            if (GpuContextPool.ForceCpu)
            {
                // CLI callers forcing CPU pow2 computations still bypass the GPU ladder even for tiny rotations.
                return Pow2ModWindowedCpu(rotation, modulus);
            }

            // Large moduli benefit from the GPU ladder even for tiny exponents; routing them back to the device avoids
            // the expensive UInt128 multiplications that were stalling production scans and inflating host memory usage.
            return Pow2ModWindowedGpu(rotation, modulus);
        }

        // Production scans never feed power-of-two moduli, so the specialized mask-and-shift shortcut is
        // commented out. Validation harnesses that still exercise those inputs should call a dedicated helper
        // if they need the micro-optimized path.
        // if ((modulus & (modulus - UInt128.One)) == UInt128.Zero)
        // {
        //     int shift = (int)(rotation & UInt128Numbers.OneHundredTwentySeven);
        //     UInt128 mask = modulus - UInt128.One;
        //     UInt128 result = UInt128.One << shift;
        //     return result & mask;
        // }

        if (GpuContextPool.ForceCpu)
        {
            // Respect --primes-device=cpu outside GPU kernels by routing the windowed pow2 helper through the CPU-only ladder.
            return Pow2ModWindowedCpu(rotation, modulus);
        }

        return Pow2ModWindowedGpu(rotation, modulus);
    }

    private static UInt128 Pow2ModWindowedCpu(UInt128 rotation, UInt128 modulus)
    {
        if ((rotation >> 64) == UInt128.Zero && modulus <= ulong.MaxValue)
        {
            ulong rotation64 = (ulong)rotation;
            ulong modulus64 = (ulong)modulus;
            ulong remainder = rotation64.Pow2ModWindowedCpu(modulus64);
            return remainder;
        }

        return Pow2ModWindowedCpuLargeRotation(rotation, modulus);
    }

    private static UInt128 Pow2ModWindowedGpu(UInt128 rotation, UInt128 modulus)
    {
        if ((rotation >> 64) == UInt128.Zero)
        {
            ulong rotation64 = (ulong)rotation;
            ReadOnlyGpuUInt128 readOnlyModulus = new ReadOnlyGpuUInt128(modulus);
            GpuUInt128 pow = GpuUInt128.Pow2ModWindowed(rotation64, in readOnlyModulus);
            return (UInt128)pow;
        }

        return Pow2ModWindowedGpuLargeRotation(rotation, modulus);
    }

    private static UInt128 Pow2ModWindowedCpuLargeRotation(UInt128 rotation, UInt128 modulus)
    {
        int bitLength = rotation.GetBitLength();
        int windowSize = GetWindowSize(bitLength);
        int oddPowerCount = 1 << (windowSize - 1);

        Span<UInt128> oddPowerStorage = oddPowerCount <= PerfectNumberConstants.MaxOddPowersCount
            ? stackalloc UInt128[PerfectNumberConstants.MaxOddPowersCount]
            : new UInt128[oddPowerCount];
        Span<UInt128> oddPowers = oddPowerStorage[..oddPowerCount];
        InitializePow2OddPowersCpu(modulus, oddPowers);

        UInt128 result = UInt128.One;
        int index = bitLength - 1;

        while (index >= 0)
        {
            if (!IsBitSet(rotation, index))
            {
                result = result.MulMod(result, modulus);
                index--;
                continue;
            }

            int windowStart = index - windowSize + 1;
            if (windowStart < 0)
            {
                windowStart = 0;
            }

            while (!IsBitSet(rotation, windowStart))
            {
                windowStart++;
            }

            int windowBitCount = index - windowStart + 1;
            for (int square = 0; square < windowBitCount; square++)
            {
                result = result.MulMod(result, modulus);
            }

            ulong windowValue = ExtractWindowValue(rotation, windowStart, windowBitCount);
            int tableIndex = (int)((windowValue - 1UL) >> 1);
            result = result.MulMod(oddPowers[tableIndex], modulus);

            index = windowStart - 1;
        }

        return result;
    }

    private static UInt128 Pow2ModWindowedGpuLargeRotation(UInt128 rotation, UInt128 modulus)
    {
        ReadOnlyGpuUInt128 modulusReadOnly = new ReadOnlyGpuUInt128(modulus);
        int bitLength = rotation.GetBitLength();
        int windowSize = GetWindowSize(bitLength);
        int oddPowerCount = 1 << (windowSize - 1);

        Span<GpuUInt128> oddPowerStorage = oddPowerCount <= PerfectNumberConstants.MaxOddPowersCount
            ? stackalloc GpuUInt128[PerfectNumberConstants.MaxOddPowersCount]
            : new GpuUInt128[oddPowerCount];
        Span<GpuUInt128> oddPowers = oddPowerStorage[..oddPowerCount];
        InitializePow2OddPowersGpu(modulusReadOnly, oddPowers);

        GpuUInt128 result = GpuUInt128.One;
        int index = bitLength - 1;

        while (index >= 0)
        {
            if (!IsBitSet(rotation, index))
            {
                result.SquareMod(modulusReadOnly);
                index--;
                continue;
            }

            int windowStart = index - windowSize + 1;
            if (windowStart < 0)
            {
                windowStart = 0;
            }

            while (!IsBitSet(rotation, windowStart))
            {
                windowStart++;
            }

            int windowBitCount = index - windowStart + 1;
            for (int square = 0; square < windowBitCount; square++)
            {
                result.SquareMod(modulusReadOnly);
            }

            ulong windowValue = ExtractWindowValue(rotation, windowStart, windowBitCount);
            int tableIndex = (int)((windowValue - 1UL) >> 1);
            result.MulMod(oddPowers[tableIndex], modulusReadOnly);

            index = windowStart - 1;
        }

        return (UInt128)result;
    }

    private static void InitializePow2OddPowersCpu(UInt128 modulus, Span<UInt128> oddPowers)
    {
        UInt128 baseValue = UInt128.One << 1;
        int oddPowersLength = oddPowers.Length;
        oddPowers[0] = baseValue;
        if (oddPowersLength == 1)
        {
            return;
        }

        UInt128 baseSquare = baseValue;
        baseSquare = baseSquare.MulMod(baseValue, modulus);
        // Reusing baseSquare to carry base^2 so the CPU ladder mirrors the GPU helper without introducing extra temporaries.
        for (int i = 1; i < oddPowersLength; i++)
        {
            oddPowers[i] = oddPowers[i - 1];
            oddPowers[i] = oddPowers[i].MulMod(baseSquare, modulus);
        }
    }

    private static void InitializePow2OddPowersGpu(in ReadOnlyGpuUInt128 modulus, Span<GpuUInt128> oddPowers)
    {
        GpuUInt128 baseValue = GpuUInt128.Two;
        // Production moduli always exceed two, so the baseValue >= modulus guard never fires outside test
        // scaffolding. Keep the branch commented out to document the constraint without injecting extra
        // instructions into the scanner path.
        // if (baseValue.CompareTo(modulus) >= 0)
        // {
        //     baseValue.Sub(modulus);
        // }

        int oddPowersLength = oddPowers.Length;
        oddPowers[0] = baseValue;
        if (oddPowersLength == 1)
        {
            return;
        }

        GpuUInt128 baseSquare = baseValue;
        baseSquare.MulMod(baseValue, modulus);
        for (int i = 1; i < oddPowersLength; i++)
        {
            oddPowers[i] = oddPowers[i - 1];
            oddPowers[i].MulMod(baseSquare, modulus);
        }
    }

    private static bool IsBitSet(UInt128 value, int bitIndex)
    {
        if (bitIndex >= 64)
        {
            ulong high = (ulong)(value >> 64);
            return ((high >> (bitIndex - 64)) & 1UL) != 0UL;
        }

        ulong low = (ulong)value;
        return ((low >> bitIndex) & 1UL) != 0UL;
    }

    private static ulong ExtractWindowValue(UInt128 value, int windowStart, int windowBitCount)
    {
        UInt128 shifted = value >> windowStart;
        ulong mask = (1UL << windowBitCount) - 1UL;
        return (ulong)shifted & mask;
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
    public static ulong Pow2MontgomeryModWindowedCpu(this ulong exponent, in MontgomeryDivisorData divisor, bool keepMontgomery)
    {
        ulong modulus = divisor.Modulus;
        if (exponent == 0UL)
        {
            return keepMontgomery
                ? divisor.MontgomeryOne
                : 1UL; // Montgomery moduli stay above one on the scanning path, so the modulo would always yield one.
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

        Span<ulong> oddPowerStorage = oddPowerCount <= PerfectNumberConstants.MaxOddPowersCount
            ? stackalloc ulong[PerfectNumberConstants.MaxOddPowersCount]
            : default;
        ulong[]? pooledOddPowers = null;

        if (oddPowerCount <= PerfectNumberConstants.MaxOddPowersCount)
        {
            oddPowerStorage = oddPowerStorage[..oddPowerCount];
        }
        else if (oddPowerCount < PerfectNumberConstants.PooledArrayThreshold)
        {
            oddPowerStorage = new ulong[oddPowerCount];
        }
        else
        {
            pooledOddPowers = ArrayPool<ulong>.Shared.Rent(oddPowerCount);
            oddPowerStorage = pooledOddPowers;
        }

        Span<ulong> oddPowers = oddPowerStorage[..oddPowerCount];
        InitializeMontgomeryOddPowersCpu(divisor, modulus, nPrime, oddPowers);

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

        if (pooledOddPowers is not null)
        {
            ArrayPool<ulong>.Shared.Return(pooledOddPowers);
        }

        if (keepMontgomery)
        {
            return result;
        }

        return result.MontgomeryMultiply(1UL, modulus, nPrime);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Pow2MontgomeryModWindowedGpu(this ulong exponent, in MontgomeryDivisorData divisor, bool keepMontgomery)
    {
        ulong modulus = divisor.Modulus;
        if (exponent == 0UL)
        {
            return keepMontgomery
                ? divisor.MontgomeryOne
                : 1UL; // Montgomery moduli stay above one on the scanning path, so the modulo would always yield one.
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
        ulong[] oddPowers = InitializeMontgomeryOddPowersGpu(divisor, modulus, nPrime, oddPowerCount);

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
    private static int GetGpuWindowSize(int bitLength)
    {
        int window = GetWindowSize(bitLength);
        return window > Pow2WindowSizeGpu ? Pow2WindowSizeGpu : window;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void InitializePlainOddPowers(ulong modulus, Span<ulong> destination)
    {
        int oddPowerCount = destination.Length;
        // The by-divisor kernels always request at least one odd power, but other helpers reuse this initializer
        // for smaller workloads, so the early exit stays in place.
        if (oddPowerCount == 0)
        {
            return;
        }

        ulong baseValue = 2UL; // Moduli arrive as odd primes greater than two, so the modulo would always return two.
        destination[0] = baseValue;
        if (oddPowerCount == 1)
        {
            return;
        }

        ulong square = baseValue.MulMod64(baseValue, modulus);
        for (int i = 1; i < oddPowerCount; i++)
        {
            // Reuse baseValue to accumulate successive odd powers and avoid allocating another local.
            baseValue = baseValue.MulMod64(square, modulus);
            destination[i] = baseValue;
        }
    }

    private struct PlainOddPowerTableGpu
    {
        private readonly int _count;
        private ulong _value0;
        private ulong _value1;
        private ulong _value2;
        private ulong _value3;
        private ulong _value4;
        private ulong _value5;
        private ulong _value6;
        private ulong _value7;
        private ulong _value8;
        private ulong _value9;
        private ulong _value10;
        private ulong _value11;
        private ulong _value12;
        private ulong _value13;
        private ulong _value14;
        private ulong _value15;

        public PlainOddPowerTableGpu(ulong modulus, int requestedCount)
        {
            _count = requestedCount > Pow2WindowOddPowerCountGpu ? Pow2WindowOddPowerCountGpu : requestedCount;
            _value0 = 0UL;
            _value1 = 0UL;
            _value2 = 0UL;
            _value3 = 0UL;
            _value4 = 0UL;
            _value5 = 0UL;
            _value6 = 0UL;
            _value7 = 0UL;
            _value8 = 0UL;
            _value9 = 0UL;
            _value10 = 0UL;
            _value11 = 0UL;
            _value12 = 0UL;
            _value13 = 0UL;
            _value14 = 0UL;
            _value15 = 0UL;

            // Window planning always requests at least one odd power, so keep the zero-count check commented out to avoid
            // branching here.
            // if (_count == 0)
            // {
            //     return;
            // }

            ulong baseValue = 2UL;
            _value0 = baseValue;
            if (_count == 1)
            {
                return;
            }

            ulong square = baseValue.MulMod64(baseValue, modulus);
            ulong current = baseValue;
            for (int i = 1; i < _count; i++)
            {
                current = current.MulMod64(square, modulus);
                SetValue(i, current);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ulong Get(int index)
        {
            // Windowed lookups never exceed the populated table bounds on production workloads, so leave the check commented
            // out to remove the branch.
            // if ((uint)index >= (uint)_count)
            // {
            //     return 0UL;
            // }

            return index switch
            {
                0 => _value0,
                1 => _value1,
                2 => _value2,
                3 => _value3,
                4 => _value4,
                5 => _value5,
                6 => _value6,
                7 => _value7,
                8 => _value8,
                9 => _value9,
                10 => _value10,
                11 => _value11,
                12 => _value12,
                13 => _value13,
                14 => _value14,
                _ => _value15,
            };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void SetValue(int index, ulong value)
        {
            switch (index)
            {
                case 1:
                    _value1 = value;
                    break;
                case 2:
                    _value2 = value;
                    break;
                case 3:
                    _value3 = value;
                    break;
                case 4:
                    _value4 = value;
                    break;
                case 5:
                    _value5 = value;
                    break;
                case 6:
                    _value6 = value;
                    break;
                case 7:
                    _value7 = value;
                    break;
                case 8:
                    _value8 = value;
                    break;
                case 9:
                    _value9 = value;
                    break;
                case 10:
                    _value10 = value;
                    break;
                case 11:
                    _value11 = value;
                    break;
                case 12:
                    _value12 = value;
                    break;
                case 13:
                    _value13 = value;
                    break;
                case 14:
                    _value14 = value;
                    break;
                case 15:
                    _value15 = value;
                    break;
            }
        }
    }

    private static void InitializeMontgomeryOddPowersCpu(in MontgomeryDivisorData divisor, ulong modulus, ulong nPrime, Span<ulong> destination)
    {
        int oddPowerCount = destination.Length;
        if (oddPowerCount == 0)
        {
            return;
        }

        bool computedOnGpu = oddPowerCount <= PerfectNumberConstants.MaxOddPowersCount
            && MontgomeryOddPowerGpu.TryCompute(divisor, oddPowerCount, destination);
        if (computedOnGpu)
        {
            return;
        }

        destination[0] = divisor.MontgomeryTwo;
        if (oddPowerCount == 1)
        {
            return;
        }

        ulong square = divisor.MontgomeryTwoSquared;
        ulong current = destination[0];
        for (int i = 1; i < oddPowerCount; i++)
        {
            current = current.MontgomeryMultiply(square, modulus, nPrime);
            destination[i] = current;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong[] InitializeMontgomeryOddPowersGpu(in MontgomeryDivisorData divisor, ulong modulus, ulong nPrime, int oddPowerCount)
    {
        ulong[] oddPowers = new ulong[PerfectNumberConstants.MaxOddPowersCount];
        oddPowers[0] = divisor.MontgomeryTwo;
        if (oddPowerCount == 1)
        {
            return oddPowers;
        }

        ulong square = divisor.MontgomeryTwoSquared;
        for (int i = 1; i < oddPowerCount; i++)
        {
            ulong previous = oddPowers[i - 1];
            oddPowers[i] = previous.MontgomeryMultiply(square, modulus, nPrime);
        }

        return oddPowers;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Pow2MontgomeryModWithCycleCpu(this ulong exponent, ulong cycleLength, in MontgomeryDivisorData divisor)
    {
        ulong rotationCount = exponent % cycleLength;
        return Pow2MontgomeryModWindowedCpu(rotationCount, divisor, keepMontgomery: false);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Pow2MontgomeryModFromCycleRemainderCpu(this ulong reducedExponent, in MontgomeryDivisorData divisor)
    {
        return Pow2MontgomeryModWindowedCpu(reducedExponent, divisor, keepMontgomery: false);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Pow2MontgomeryModWithCycleGpu(this ulong exponent, ulong cycleLength, in MontgomeryDivisorData divisor)
    {
        ulong rotationCount = exponent % cycleLength;
        return Pow2MontgomeryModWindowedGpu(rotationCount, divisor, keepMontgomery: false);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Pow2MontgomeryModFromCycleRemainderGpu(this ulong reducedExponent, in MontgomeryDivisorData divisor)
    {
        return Pow2MontgomeryModWindowedGpu(reducedExponent, divisor, keepMontgomery: false);
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
                result = ((UInt128)exponent).Pow2ModWindowed(10, 4),
                zero = UInt128.Zero;

        for (; modulusCandidate <= modulus; modulusCandidate++)
        {
            cycle = MersenneDivisorCycles.GetCycle(modulusCandidate);
            remainderForCandidate = cycle > zero
                    ? ((UInt128)exponent).Pow2ModWindowed(modulusCandidate, cycle)
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
