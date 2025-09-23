using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
public class GpuModularArithmeticBenchmarks
{
    private const int MaxBatchSize = 512;
    private readonly ulong[] _exponents = new ulong[MaxBatchSize];
    private readonly ulong[] _moduli = new ulong[MaxBatchSize];
    private readonly ulong[] _mersenneModuli = new ulong[MaxBatchSize];
    private readonly int[] _mersenneBitWidths = new int[MaxBatchSize];

    private readonly Random _random = new(7);

    [Params(32, 256)]
    public int BatchSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        for (int i = 0; i < MaxBatchSize; i++)
        {
            _exponents[i] = NextExponent();
            _moduli[i] = NextOddModulus();

            int bits = _random.Next(8, 62);
            _mersenneBitWidths[i] = bits;
            _mersenneModuli[i] = (1UL << bits) - 1UL;
        }
    }

    [Benchmark(Baseline = true)]
    public ulong BaselineMulMod()
    {
        ulong checksum = 0UL;
        for (int i = 0; i < BatchSize; i++)
        {
            checksum ^= Pow2ModBaseline(_exponents[i], _moduli[i]);
        }

        return checksum;
    }

    [Benchmark]
    public ulong MultiplyHighMulMod()
    {
        ulong checksum = 0UL;
        for (int i = 0; i < BatchSize; i++)
        {
            checksum ^= Pow2ModMultiplyHigh(_exponents[i], _moduli[i]);
        }

        return checksum;
    }

    [Benchmark]
    public ulong MersenneFoldingMulMod()
    {
        ulong checksum = 0UL;
        for (int i = 0; i < BatchSize; i++)
        {
            checksum ^= Pow2ModMersenne(_exponents[i], _mersenneBitWidths[i], _mersenneModuli[i]);
        }

        return checksum;
    }

    private ulong NextExponent()
    {
        return (ulong)_random.Next(24, 63);
    }

    private ulong NextOddModulus()
    {
        ulong value = (ulong)_random.NextInt64(5L, long.MaxValue);
        if ((value & 1UL) == 0UL)
        {
            value++;
        }

        return value;
    }

    private static ulong Pow2ModBaseline(ulong exponent, ulong modulus)
    {
        if (modulus <= 1UL)
        {
            return 0UL;
        }

        ulong result = 1UL % modulus;
        ulong baseVal = 2UL % modulus;
        ulong exp = exponent;
        while (exp > 0UL)
        {
            if ((exp & 1UL) != 0UL)
            {
                result = MulModBaseline(result, baseVal, modulus);
            }

            exp >>= 1;
            if (exp == 0UL)
            {
                break;
            }

            baseVal = MulModBaseline(baseVal, baseVal, modulus);
        }

        return result;
    }

    private static ulong Pow2ModMultiplyHigh(ulong exponent, ulong modulus)
    {
        if (modulus <= 1UL)
        {
            return 0UL;
        }

        ulong result = 1UL % modulus;
        ulong baseVal = 2UL % modulus;
        ulong exp = exponent;
        while (exp > 0UL)
        {
            if ((exp & 1UL) != 0UL)
            {
                result = MulModMultiplyHigh(result, baseVal, modulus);
            }

            exp >>= 1;
            if (exp == 0UL)
            {
                break;
            }

            baseVal = MulModMultiplyHigh(baseVal, baseVal, modulus);
        }

        return result;
    }

    private static ulong Pow2ModMersenne(ulong exponent, int bitWidth, ulong modulus)
    {
        if (modulus <= 1UL)
        {
            return 0UL;
        }

        ulong result = 1UL & modulus;
        ulong baseVal = 2UL & modulus;
        ulong exp = exponent;
        while (exp > 0UL)
        {
            if ((exp & 1UL) != 0UL)
            {
                result = MulModMersenne(result, baseVal, modulus, bitWidth);
            }

            exp >>= 1;
            if (exp == 0UL)
            {
                break;
            }

            baseVal = MulModMersenne(baseVal, baseVal, modulus, bitWidth);
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MulModBaseline(ulong a, ulong b, ulong modulus)
    {
        if (modulus == 0UL)
        {
            return 0UL;
        }

        ulong result = 0UL;
        ulong x = a % modulus;
        ulong y = b;
        while (y > 0UL)
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
    private static ulong MulModMultiplyHigh(ulong a, ulong b, ulong modulus)
    {
        if (modulus == 0UL)
        {
            return 0UL;
        }

        UInt128 product = (UInt128)a * b;
        return (ulong)(product % modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MulModMersenne(ulong a, ulong b, ulong modulus, int bitWidth)
    {
        UInt128 product = (UInt128)a * b;
        ulong mask = modulus;
        ulong low = (ulong)product & mask;
        ulong high = (ulong)(product >> bitWidth);
        ulong result = low + high;

        while (result > mask)
        {
            result = (result & mask) + (result >> bitWidth);
        }

        if (result >= modulus)
        {
            result -= modulus;
        }

        return result;
    }
}
