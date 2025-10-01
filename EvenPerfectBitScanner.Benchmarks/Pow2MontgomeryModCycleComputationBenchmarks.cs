using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

/// <remarks>
/// Benchmark results collected on AMD Ryzen 5 5625U (.NET 8.0, launchCount: 1, warmupCount: 1, iterationCount: 5):
/// Small sample (~1e6 random odd moduli)
/// - MontgomeryWithoutCycle: 35.41 µs (baseline)
/// - MontgomeryWithPrecomputedCycle: 25.81 µs (0.73× baseline)
/// - MontgomeryWithGpuCycleComputation: 54.09 ms (1,527× baseline, includes GPU cycle calculation)
/// Large sample (Mersenne-like moduli from 2^48-1 to 2^63-1)
/// - MontgomeryWithoutCycle: 117.33 µs (baseline)
/// - MontgomeryWithPrecomputedCycle: 9.12 µs (0.08× baseline)
/// - MontgomeryWithGpuCycleComputation: 12.91 µs (0.11× baseline)
/// </remarks>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
public class Pow2MontgomeryModCycleComputationBenchmarks
{
    private const int SampleCount = 256;

    private readonly MontgomeryDivisorData[] _smallDivisors = new MontgomeryDivisorData[SampleCount];
    private readonly MontgomeryDivisorData[] _largeDivisors = new MontgomeryDivisorData[SampleCount];
    private readonly ulong[] _smallExponents = new ulong[SampleCount];
    private readonly ulong[] _largeExponents = new ulong[SampleCount];
    private readonly ulong[] _smallCycles = new ulong[SampleCount];
    private readonly ulong[] _largeCycles = new ulong[SampleCount];

    private readonly Random _random = new(113);

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
    /// Baseline Montgomery reduction without precomputed cycle data; measured 35.41 μs on the small sample and 117.33 μs on the
    /// large set.
    /// </summary>
    [Benchmark(Baseline = true)]
    public ulong MontgomeryWithoutCycle()
    {
        GetData(out ulong[] exponents, out MontgomeryDivisorData[] divisors, out _);
        ulong checksum = 0UL;

        for (int i = 0; i < SampleCount; i++)
        {
            checksum ^= exponents[i].Pow2MontgomeryMod(divisors[i]);
        }

        return checksum;
    }

    /// <summary>
    /// Montgomery reduction with a known cycle length (cycle lookup only); 25.81 μs on the small set (0.73× baseline) and 9.12 μs
    /// on the large one (0.08×).
    /// </summary>
    [Benchmark]
    public ulong MontgomeryWithPrecomputedCycle()
    {
        GetData(out ulong[] exponents, out MontgomeryDivisorData[] divisors, out ulong[] cycles);
        ulong checksum = 0UL;

        for (int i = 0; i < SampleCount; i++)
        {
            checksum ^= exponents[i].Pow2MontgomeryModWithCycle(cycles[i], divisors[i]);
        }

        return checksum;
    }

    /// <summary>
    /// Montgomery reduction with the cycle computed on the fly using the GPU helper; costs 54.09 ms on the small benchmark due to
    /// cycle discovery, but drops to 12.91 μs on the large set (0.11× baseline).
    /// </summary>
    [Benchmark]
    public ulong MontgomeryWithGpuCycleComputation()
    {
        GetData(out ulong[] exponents, out MontgomeryDivisorData[] divisors, out _);
        ulong checksum = 0UL;

        for (int i = 0; i < SampleCount; i++)
        {
            ulong cycle = MersenneDivisorCycles.CalculateCycleLengthGpu(divisors[i].Modulus);
            checksum ^= exponents[i].Pow2MontgomeryModWithCycle(cycle, divisors[i]);
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
}

