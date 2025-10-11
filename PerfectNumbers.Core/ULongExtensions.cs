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
    public static ulong Pow2MontgomeryModWindowedCpu(this ulong exponent, in MontgomeryDivisorData divisor, bool keepMontgomery)
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

        Span<ulong> oddPowerStorage;
        ulong[]? pooledOddPowers = null;

        if (oddPowerCount <= PerfectNumberConstants.MaxOddPowersCount)
        {
            oddPowerStorage = stackalloc ulong[PerfectNumberConstants.MaxOddPowersCount];
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
