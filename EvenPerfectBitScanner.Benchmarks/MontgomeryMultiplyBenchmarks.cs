using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 10)]
public class MontgomeryMultiplyBenchmarks
{
    private const int MaxBatchSize = 1024;

    private readonly ulong[] _operandsA = new ulong[MaxBatchSize];
    private readonly ulong[] _operandsB = new ulong[MaxBatchSize];
    private readonly ulong[] _moduli = new ulong[MaxBatchSize];
    private readonly ulong[] _nPrimes = new ulong[MaxBatchSize];

    private readonly Random _random = new(1979);

    [Params(64, 256, 1024)]
    public int BatchSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        for (int i = 0; i < MaxBatchSize; i++)
        {
            ulong modulus = NextOddModulus();
            _moduli[i] = modulus;
            _nPrimes[i] = ComputeMontgomeryNPrime(modulus);
            _operandsA[i] = (ulong)_random.NextInt64(0, (long)modulus);
            _operandsB[i] = (ulong)_random.NextInt64(0, (long)modulus);
        }
    }

    [Benchmark(Baseline = true)]
    public ulong OriginalImplementation()
    {
        ulong checksum = 0UL;
        for (int i = 0; i < BatchSize; i++)
        {
            checksum ^= OriginalMontgomeryMultiply(_operandsA[i], _operandsB[i], _moduli[i], _nPrimes[i]);
        }

        return checksum;
    }

    [Benchmark]
    public ulong OptimizedImplementation()
    {
        ulong checksum = 0UL;
        for (int i = 0; i < BatchSize; i++)
        {
            checksum ^= OptimizedMontgomeryMultiply(_operandsA[i], _operandsB[i], _moduli[i], _nPrimes[i]);
        }

        return checksum;
    }

    private ulong NextOddModulus()
    {
        ulong modulus = (ulong)_random.NextInt64(3, long.MaxValue);
        if ((modulus & 1UL) == 0UL)
        {
            modulus++;
        }

        return modulus;
    }

    private static ulong OriginalMontgomeryMultiply(ulong a, ulong b, ulong modulus, ulong nPrime)
    {
        ulong tLow = unchecked(a * b);
        ulong tHigh = MultiplyHigh(a, b);
        ulong m = unchecked(tLow * nPrime);
        ulong mnLow = unchecked(m * modulus);
        ulong mnHigh = MultiplyHigh(m, modulus);

        ulong sumLow = unchecked(tLow + mnLow);
        ulong carry = sumLow < tLow ? 1UL : 0UL;
        ulong sumHigh = unchecked(tHigh + mnHigh + carry);

        ulong result = sumHigh;
        if (result >= modulus)
        {
            result -= modulus;
        }

        return result;
    }

    private static ulong OptimizedMontgomeryMultiply(ulong a, ulong b, ulong modulus, ulong nPrime)
    {
        ulong tLow = unchecked(a * b);
        ulong m = unchecked(tLow * nPrime);
        ulong sumLow = unchecked(tLow + m * modulus);
        ulong carry = sumLow < tLow ? 1UL : 0UL;
        ulong sumHigh = unchecked(MultiplyHigh(a, b) + MultiplyHigh(m, modulus) + carry);

        ulong result = sumHigh;
        if (result >= modulus)
        {
            result -= modulus;
        }

        return result;
    }

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

    private static ulong MultiplyHigh(ulong x, ulong y)
    {
        ulong xLow = (uint)x;
        ulong xHigh = x >> 32;
        ulong yLow = (uint)y;
        ulong yHigh = y >> 32;

        ulong lowProduct = xLow * yLow;
        ulong mid = xHigh * yLow + (lowProduct >> 32);
        ulong midLow = mid & 0xFFFFFFFFUL;
        ulong midHigh = mid >> 32;
        midLow += xLow * yHigh;

        ulong resultHigh = xHigh * yHigh + midHigh + (midLow >> 32);
        return resultHigh;
    }
}

