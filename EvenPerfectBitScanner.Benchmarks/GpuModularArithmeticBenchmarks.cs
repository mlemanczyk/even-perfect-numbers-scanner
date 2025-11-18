using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
public class GpuModularArithmeticBenchmarks
{
    private const int MaxBatchSize = 512;
    private ulong[] _exponents = Array.Empty<ulong>();
    private ulong[] _moduli = Array.Empty<ulong>();
    private ulong[] _mersenneModuli = Array.Empty<ulong>();
    private int[] _mersenneBitWidths = Array.Empty<int>();

    [Params(32, 256)]
    public int BatchSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        ModularArithmeticSampleData data = ModularArithmeticSampleCache.Instance;
        _exponents = data.Exponents;
        _moduli = data.Moduli;
        _mersenneModuli = data.MersenneModuli;
        _mersenneBitWidths = data.MersenneBitWidths;
    }

    private sealed class ModularArithmeticSampleData
    {
        public ModularArithmeticSampleData(ulong[] exponents, ulong[] moduli, ulong[] mersenneModuli, int[] mersenneBitWidths)
        {
            Exponents = exponents;
            Moduli = moduli;
            MersenneModuli = mersenneModuli;
            MersenneBitWidths = mersenneBitWidths;
        }

        public ulong[] Exponents { get; }
        public ulong[] Moduli { get; }
        public ulong[] MersenneModuli { get; }
        public int[] MersenneBitWidths { get; }
    }

    private static class ModularArithmeticSampleCache
    {
        private static readonly Lazy<ModularArithmeticSampleData> Cache = new(Create);

        public static ModularArithmeticSampleData Instance => Cache.Value;

        private static ModularArithmeticSampleData Create()
        {
            Console.WriteLine("Preparing modular arithmetic benchmark sample data...");

            Random random = new(7);
            var exponents = new ulong[MaxBatchSize];
            var moduli = new ulong[MaxBatchSize];
            var mersenneModuli = new ulong[MaxBatchSize];
            var mersenneBitWidths = new int[MaxBatchSize];

            for (int i = 0; i < MaxBatchSize; i++)
            {
                exponents[i] = NextExponent(random);
                moduli[i] = NextOddModulus(random);
                int bits = random.Next(8, 62);
                mersenneBitWidths[i] = bits;
                mersenneModuli[i] = (1UL << bits) - 1UL;
            }

            ModularArithmeticSampleData data = new ModularArithmeticSampleData(exponents, moduli, mersenneModuli, mersenneBitWidths);

            Console.WriteLine("Finished preparing modular arithmetic benchmark sample data.");

            return data;
        }

        private static ulong NextExponent(Random random)
        {
            return (ulong)random.Next(24, 63);
        }

        private static ulong NextOddModulus(Random random)
        {
            ulong value = (ulong)random.NextInt64(5L, long.MaxValue);
            if ((value & 1UL) == 0UL)
            {
                value++;
            }

            return value;
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
