using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using System.Numerics;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
public class Pow2MontgomeryModBenchmarks
{
    private const int SampleCount = 256;

    private readonly MontgomeryDivisorData[] _smallDivisors = new MontgomeryDivisorData[SampleCount];
    private readonly MontgomeryDivisorData[] _largeDivisors = new MontgomeryDivisorData[SampleCount];
    private readonly ulong[] _smallExponents = new ulong[SampleCount];
    private readonly ulong[] _largeExponents = new ulong[SampleCount];
    private readonly ulong[] _smallCycles = new ulong[SampleCount];
    private readonly ulong[] _largeCycles = new ulong[SampleCount];

    private readonly Random _random = new(13);

    public enum InputScale
    {
        Small,
        Large
    }

    [ParamsAllValues]
    public InputScale Scale { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        for (int i = 0; i < SampleCount; i++)
        {
            ulong smallModulus = NextSmallOddModulus();
            _smallDivisors[i] = CreateMontgomeryDivisorData(smallModulus);
            _smallExponents[i] = NextSmallExponent();
            _smallCycles[i] = MersenneDivisorCycles.CalculateCycleLength(smallModulus);

            (ulong largeModulus, ulong largeCycle) = NextLargeModulusAndCycle();
            _largeDivisors[i] = CreateMontgomeryDivisorData(largeModulus);
            _largeExponents[i] = NextLargeExponent();
            _largeCycles[i] = largeCycle;
        }
    }

    /// <summary>
    /// Baseline right-to-left Montgomery ladder that averages ~60 μs on the small set and ~205 μs on the large one.
    /// </summary>
    [Benchmark(Baseline = true)]
    public ulong Baseline()
    {
        GetData(out var exponents, out var divisors, out _);
        ulong checksum = 0UL;

        for (int i = 0; i < SampleCount; i++)
        {
            checksum ^= exponents[i].Pow2MontgomeryMod(divisors[i]);
        }

        return checksum;
    }

    /// <summary>
    /// Uses a known divisor cycle to fold the exponent first, cutting runtime to ~0.69× on small inputs (~41.7 μs) and ~0.08× on large ones (~15.9 μs).
    /// </summary>
    [Benchmark]
    public ulong BaselineWithKnownCycle()
    {
        GetData(out var exponents, out var divisors, out var cycles);
        ulong checksum = 0UL;

        for (int i = 0; i < SampleCount; i++)
        {
            checksum ^= exponents[i].Pow2MontgomeryModWithCycle(cycles[i], divisors[i]);
        }

        return checksum;
    }

    /// <summary>
    /// Left-to-right Montgomery scan that trailed the baseline by ~15% on small inputs (~69.7 μs) and ~13% on large ones (~231 μs).
    /// </summary>
    [Benchmark]
    public ulong LeftToRight()
    {
        GetData(out var exponents, out var divisors, out _);
        ulong checksum = 0UL;

        for (int i = 0; i < SampleCount; i++)
        {
            checksum ^= Pow2MontgomeryModLeftToRight(exponents[i], divisors[i]);
        }

        return checksum;
    }

    /// <summary>
    /// Processes four exponent bits per loop, ending up ~8% slower on small cases (~65.2 μs) and ~10% slower on large ones (~225 μs).
    /// </summary>
    [Benchmark]
    public ulong Batched4()
    {
        GetData(out var exponents, out var divisors, out _);
        ulong checksum = 0UL;

        for (int i = 0; i < SampleCount; i++)
        {
            checksum ^= Pow2MontgomeryModBatched4(exponents[i], divisors[i]);
        }

        return checksum;
    }

    /// <summary>
    /// Windowed Montgomery ladder (width 5) that lagged by ~72% on small samples (~104 μs) and ~16% on large samples (~238 μs).
    /// </summary>
    [Benchmark]
    public ulong SlidingWindow()
    {
        GetData(out var exponents, out var divisors, out _);
        ulong checksum = 0UL;

        for (int i = 0; i < SampleCount; i++)
        {
            checksum ^= Pow2MontgomeryModSlidingWindow(exponents[i], divisors[i]);
        }

        return checksum;
    }

    /// <summary>
    /// Classic binary ladder without Montgomery reduction; it was ~3.6× slower on both small (~217 μs) and large (~733 μs) datasets.
    /// </summary>
    [Benchmark]
    public ulong BinaryMod()
    {
        GetData(out var exponents, out var divisors, out _);
        ulong checksum = 0UL;

        for (int i = 0; i < SampleCount; i++)
        {
            checksum ^= Pow2ModBinary(exponents[i], divisors[i].Modulus);
        }

        return checksum;
    }

    /// <summary>
    /// Precomputes all squarings once, landing ~44% slower on small inputs (~87.0 μs) and ~4.9× slower on large ones (~998 μs).
    /// </summary>
    [Benchmark]
    public ulong PrecomputedTableMod()
    {
        GetData(out var exponents, out var divisors, out _);
        ulong checksum = 0UL;

        for (int i = 0; i < SampleCount; i++)
        {
            checksum ^= Pow2ModPrecomputedTable(exponents[i], divisors[i].Modulus);
        }

        return checksum;
    }

    /// <summary>
    /// Relies on <see cref="BigInteger.ModPow"/>, outperforming Montgomery on small inputs (~39.6 μs, 0.66×) but ~4.7× slower on large ones (~954 μs).
    /// </summary>
    [Benchmark]
    public ulong BigIntegerMod()
    {
        GetData(out var exponents, out var divisors, out _);
        ulong checksum = 0UL;

        for (int i = 0; i < SampleCount; i++)
        {
            checksum ^= Pow2ModBigInteger(exponents[i], divisors[i].Modulus);
        }

        return checksum;
    }

    private void GetData(out ulong[] exponents, out MontgomeryDivisorData[] divisors, out ulong[] cycles)
    {
        if (Scale == InputScale.Small)
        {
            exponents = _smallExponents;
            divisors = _smallDivisors;
            cycles = _smallCycles;
        }
        else
        {
            exponents = _largeExponents;
            divisors = _largeDivisors;
            cycles = _largeCycles;
        }
    }

    private ulong NextSmallExponent() => (ulong)_random.NextInt64(1L, 1_000_000L);

    private ulong NextLargeExponent() => (ulong)_random.NextInt64(1L << 60, long.MaxValue);

    private ulong NextSmallOddModulus()
    {
        ulong value = (ulong)_random.NextInt64(3L, 1_000_000L);
        return value | 1UL;
    }

    private (ulong modulus, ulong cycleLength) NextLargeModulusAndCycle()
    {
        int bitLength = _random.Next(48, 64);
        ulong modulus = (1UL << bitLength) - 1UL;
        return (modulus, (ulong)bitLength);
    }

    private static MontgomeryDivisorData CreateMontgomeryDivisorData(ulong modulus)
    {
        if (modulus <= 1UL || (modulus & 1UL) == 0UL)
        {
            return new MontgomeryDivisorData(modulus, 0UL, 0UL, 0UL);
        }

        return new MontgomeryDivisorData(
            modulus,
            ComputeMontgomeryNPrime(modulus),
            ComputeMontgomeryResidue(1UL, modulus),
            ComputeMontgomeryResidue(2UL, modulus));
    }

    private static ulong ComputeMontgomeryResidue(ulong value, ulong modulus) => (ulong)((UInt128)value * (UInt128.One << 64) % modulus);

    private static ulong ComputeMontgomeryNPrime(ulong modulus)
    {
        ulong inv = modulus;
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        return unchecked(0UL - inv);
    }

    private static ulong Pow2MontgomeryModLeftToRight(ulong exponent, in MontgomeryDivisorData divisor)
    {
        ulong modulus = divisor.Modulus;
        if (modulus <= 1UL || (modulus & 1UL) == 0UL)
        {
            return 0UL;
        }

        ulong result = divisor.MontgomeryOne;
        ulong baseVal = divisor.MontgomeryTwo;
        ulong nPrime = divisor.NPrime;

        if (exponent == 0UL)
        {
            return result.MontgomeryMultiply(1UL, modulus, nPrime);
        }

        int msbIndex = 63 - BitOperations.LeadingZeroCount(exponent);

        for (int bitIndex = msbIndex; bitIndex >= 0; bitIndex--)
        {
            result = result.MontgomeryMultiply(result, modulus, nPrime);

            if (((exponent >> bitIndex) & 1UL) != 0UL)
            {
                result = result.MontgomeryMultiply(baseVal, modulus, nPrime);
            }
        }

        return result.MontgomeryMultiply(1UL, modulus, nPrime);
    }

    private static ulong Pow2MontgomeryModBatched4(ulong exponent, in MontgomeryDivisorData divisor)
    {
        ulong modulus = divisor.Modulus;
        if (modulus <= 1UL || (modulus & 1UL) == 0UL)
        {
            return 0UL;
        }

        ulong result = divisor.MontgomeryOne;
        ulong baseVal = divisor.MontgomeryTwo;
        ulong nPrime = divisor.NPrime;
        ulong remainingExponent = exponent;

        while (remainingExponent != 0UL)
        {
            ulong nibble = remainingExponent & 0xFUL;

            ulong base0 = baseVal;
            ulong base1 = base0.MontgomeryMultiply(base0, modulus, nPrime);
            ulong base2 = base1.MontgomeryMultiply(base1, modulus, nPrime);
            ulong base3 = base2.MontgomeryMultiply(base2, modulus, nPrime);

            if ((nibble & 1UL) != 0UL)
            {
                result = result.MontgomeryMultiply(base0, modulus, nPrime);
            }

            if ((nibble & 2UL) != 0UL)
            {
                result = result.MontgomeryMultiply(base1, modulus, nPrime);
            }

            if ((nibble & 4UL) != 0UL)
            {
                result = result.MontgomeryMultiply(base2, modulus, nPrime);
            }

            if ((nibble & 8UL) != 0UL)
            {
                result = result.MontgomeryMultiply(base3, modulus, nPrime);
            }

            baseVal = base3.MontgomeryMultiply(base3, modulus, nPrime);
            remainingExponent >>= 4;
        }

        return result.MontgomeryMultiply(1UL, modulus, nPrime);
    }

    private static ulong Pow2MontgomeryModSlidingWindow(ulong exponent, in MontgomeryDivisorData divisor)
    {
        ulong modulus = divisor.Modulus;
        if (modulus <= 1UL || (modulus & 1UL) == 0UL)
        {
            return 0UL;
        }

        ulong nPrime = divisor.NPrime;
        ulong result = divisor.MontgomeryOne;
        ulong baseVal = divisor.MontgomeryTwo;

        if (exponent == 0UL)
        {
            return result.MontgomeryMultiply(1UL, modulus, nPrime);
        }

        const int WindowSize = 5;
        Span<ulong> oddPowers = stackalloc ulong[1 << (WindowSize - 1)];
        oddPowers[0] = baseVal;
        ulong baseSquared = baseVal.MontgomeryMultiply(baseVal, modulus, nPrime);
        for (int i = 1; i < oddPowers.Length; i++)
        {
            oddPowers[i] = oddPowers[i - 1].MontgomeryMultiply(baseSquared, modulus, nPrime);
        }

        int bitIndex = 63 - BitOperations.LeadingZeroCount(exponent);

        while (bitIndex >= 0)
        {
            if (((exponent >> bitIndex) & 1UL) == 0UL)
            {
                result = result.MontgomeryMultiply(result, modulus, nPrime);
                bitIndex--;
                continue;
            }

            int windowStart = Math.Max(bitIndex - WindowSize + 1, 0);
            ulong windowValue = (exponent >> windowStart) & ((1UL << (bitIndex - windowStart + 1)) - 1UL);

            while ((windowValue & 1UL) == 0UL)
            {
                windowValue >>= 1;
                windowStart++;
            }

            int squares = bitIndex - windowStart + 1;
            for (int i = 0; i < squares; i++)
            {
                result = result.MontgomeryMultiply(result, modulus, nPrime);
            }

            result = result.MontgomeryMultiply(oddPowers[(int)(windowValue >> 1)], modulus, nPrime);
            bitIndex = windowStart - 1;
        }

        return result.MontgomeryMultiply(1UL, modulus, nPrime);
    }

    private static ulong Pow2ModBinary(ulong exponent, ulong modulus)
    {
        if (modulus <= 1UL || (modulus & 1UL) == 0UL)
        {
            return 0UL;
        }

        ulong result = 1UL % modulus;
        ulong baseVal = 2UL % modulus;
        ulong remainingExponent = exponent;

        while (remainingExponent > 0UL)
        {
            if ((remainingExponent & 1UL) != 0UL)
            {
                result = MultiplyMod(result, baseVal, modulus);
            }

            remainingExponent >>= 1;
            if (remainingExponent == 0UL)
            {
                break;
            }

            baseVal = MultiplyMod(baseVal, baseVal, modulus);
        }

        return result;
    }

    private static ulong Pow2ModPrecomputedTable(ulong exponent, ulong modulus)
    {
        if (modulus <= 1UL || (modulus & 1UL) == 0UL)
        {
            return 0UL;
        }

        if (exponent == 0UL)
        {
            return 1UL % modulus;
        }

        Span<ulong> powers = stackalloc ulong[64];
        powers[0] = 2UL % modulus;
        int highestBit = 63 - BitOperations.LeadingZeroCount(exponent);

        for (int index = 1; index <= highestBit; index++)
        {
            powers[index] = MultiplyMod(powers[index - 1], powers[index - 1], modulus);
        }

        ulong result = 1UL % modulus;
        ulong remainingExponent = exponent;
        int bitIndex = 0;

        while (remainingExponent > 0UL)
        {
            if ((remainingExponent & 1UL) != 0UL)
            {
                result = MultiplyMod(result, powers[bitIndex], modulus);
            }

            remainingExponent >>= 1;
            bitIndex++;
        }

        return result;
    }

    private static ulong Pow2ModBigInteger(ulong exponent, ulong modulus)
    {
        if (modulus <= 1UL || (modulus & 1UL) == 0UL)
        {
            return 0UL;
        }

        BigInteger mod = modulus;
        BigInteger result = BigInteger.ModPow(2, exponent, mod);
        return (ulong)result;
    }

    private static ulong MultiplyMod(ulong x, ulong y, ulong modulus)
    {
        UInt128 product = (UInt128)x * y;
        return (ulong)(product % modulus);
    }
}
