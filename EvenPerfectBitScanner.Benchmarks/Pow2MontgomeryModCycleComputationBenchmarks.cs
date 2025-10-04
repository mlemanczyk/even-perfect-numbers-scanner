using System;
using System.Buffers.Binary;
using System.Threading;
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
    private const ulong OneShiftedLeft63 = 1UL << 63;
    private const long OneShiftedLeft60 = 1L << 60;
    private readonly MontgomeryDivisorData[] _smallDivisors = new MontgomeryDivisorData[SampleCount];
    private readonly MontgomeryDivisorData[] _largeDivisors = new MontgomeryDivisorData[SampleCount];
    private readonly MontgomeryDivisorData[] _veryLargeDivisors = new MontgomeryDivisorData[SampleCount];
    private readonly UInt128[] _wideModuli = new UInt128[SampleCount];
    private readonly ulong[] _smallExponents = new ulong[SampleCount];
    private readonly ulong[] _largeExponents = new ulong[SampleCount];
    private readonly ulong[] _veryLargeExponents = new ulong[SampleCount];
    private readonly ulong[] _smallCycles = new ulong[SampleCount];
    private readonly ulong[] _largeCycles = new ulong[SampleCount];
    private readonly ulong[] _veryLargeCycles = new ulong[SampleCount];
    private readonly UInt128[] _wideCycles = new UInt128[SampleCount];

    private readonly Random _random = new(113);
    private ulong? _previousPrimeOrder;
    private UInt128? _previousWidePrimeOrder;

    public enum InputScale
    {
        Small,
        Large,
        VeryLarge
    }

    [ParamsAllValues]
    public InputScale Scale { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        for (int i = 0; i < SampleCount; i++)
        {
#if DEBUG
            Console.WriteLine($"Generating sample {i}");
#endif
            ulong smallModulus = NextSmallOddModulus();
            _smallDivisors[i] = CreateMontgomeryDivisorData(smallModulus);
            _smallExponents[i] = NextSmallExponent();
            _smallCycles[i] = CalculateCycleLengthWithHeuristics(smallModulus);

            (ulong largeModulus, ulong largeCycle) = NextLargeModulusAndCycle();
            _largeDivisors[i] = CreateMontgomeryDivisorData(largeModulus);
            _largeExponents[i] = NextLargeExponent();
            _largeCycles[i] = CalculateCycleLengthWithHeuristics(largeModulus, largeCycle);

            ulong veryLargeModulus = NextVeryLargeOddModulus();
            _veryLargeDivisors[i] = CreateMontgomeryDivisorData(veryLargeModulus);
            _veryLargeExponents[i] = NextVeryLargeExponent();
#if DEBUG
            Console.WriteLine($"Calculating cycle length {veryLargeModulus}");
#endif
            _veryLargeCycles[i] = CalculateCycleLengthWithHeuristics(veryLargeModulus);

            UInt128 wideModulus = NextWideOddModulus();
            _wideModuli[i] = wideModulus;
            _wideCycles[i] = CalculateCycleLengthWithHeuristics(wideModulus);
        }
    }

    /// <summary>
    /// Baseline Montgomery reduction without precomputed cycle data; measured 35.41 μs on the small sample and 117.33 μs on the
    /// large set.
    /// </summary>
    // [Benchmark(Baseline = true)]
    public ulong MontgomeryWithoutCycle()
    {
        GetData(out ulong[] exponents, out MontgomeryDivisorData[] divisors, out _);
        ulong checksum = 0UL;

        for (int i = 0; i < SampleCount; i++)
        {
            checksum ^= exponents[i].Pow2MontgomeryModWindowed(divisors[i], keepMontgomery: false);
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
    // [Benchmark]
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


    [Benchmark]
    public ulong MontgomeryWithHeuristicCycleComputation()
    {
        GetData(out ulong[] exponents, out MontgomeryDivisorData[] divisors, out _);
        ulong checksum = 0UL;

        for (int i = 0; i < SampleCount; i++)
        {
            ulong cycle = CalculateCycleLengthWithHeuristics(divisors[i].Modulus);
            checksum ^= exponents[i].Pow2MontgomeryModWithCycle(cycle, divisors[i]);
        }

        return checksum;
    }

    [Benchmark]
    public UInt128 HeuristicWideCycleComputation()
    {
        UInt128 checksum = UInt128.Zero;

        for (int i = 0; i < SampleCount; i++)
        {
            UInt128 cycle = CalculateCycleLengthWithHeuristics(_wideModuli[i]);
            _wideCycles[i] = cycle;
            checksum ^= cycle;
        }

        return checksum;
    }

    private void GetData(out ulong[] exponents, out MontgomeryDivisorData[] divisors, out ulong[] cycles)
    {
        switch (Scale)
        {
            case InputScale.Small:
                exponents = _smallExponents;
                divisors = _smallDivisors;
                cycles = _smallCycles;
                break;
            case InputScale.Large:
                exponents = _largeExponents;
                divisors = _largeDivisors;
                cycles = _largeCycles;
                break;
            case InputScale.VeryLarge:
                exponents = _veryLargeExponents;
                divisors = _veryLargeDivisors;
                cycles = _veryLargeCycles;
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(Scale), Scale, null);
        }
    }

    private ulong NextSmallExponent() => (ulong)_random.NextInt64(1L, 1_000_000L);

    private ulong NextLargeExponent() => (ulong)_random.NextInt64(OneShiftedLeft60, long.MaxValue);

    private ulong NextVeryLargeExponent()
    {
        Span<byte> buffer = stackalloc byte[8];
        _random.NextBytes(buffer);
        ulong value = checked(BinaryPrimitives.ReadUInt64LittleEndian(buffer) | OneShiftedLeft63);
        return value;
    }

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

    private ulong NextVeryLargeOddModulus()
    {
        Span<byte> buffer = stackalloc byte[8];
        _random.NextBytes(buffer);
        ulong value = BinaryPrimitives.ReadUInt64LittleEndian(buffer) | OneShiftedLeft63 | 1UL;
        return value;
    }

    private UInt128 NextWideOddModulus()
    {
        Span<byte> buffer = stackalloc byte[16];
        _random.NextBytes(buffer);
        ulong low = BinaryPrimitives.ReadUInt64LittleEndian(buffer);
        Span<byte> highSlice = buffer.Slice(8, 8);
        ulong high = BinaryPrimitives.ReadUInt64LittleEndian(highSlice) | (1UL << 63);
        UInt128 value = ((UInt128)high << 64) | low;
        return value | UInt128.One;
    }

    private static MontgomeryDivisorData CreateMontgomeryDivisorData(ulong modulus)
    {
        if (modulus <= 1UL || (modulus & 1UL) == 0UL)
        {
            return new MontgomeryDivisorData(modulus, 0UL, 0UL, 0UL, 0UL);
        }

        ulong nPrime = ComputeMontgomeryNPrime(modulus);
        ulong montgomeryOne = ComputeMontgomeryResidue(1UL, modulus);
        ulong montgomeryTwo = ComputeMontgomeryResidue(2UL, modulus);
        ulong montgomeryTwoSquared = ULongExtensions.MontgomeryMultiply(montgomeryTwo, montgomeryTwo, modulus, nPrime);

        return new MontgomeryDivisorData(
            modulus,
            nPrime,
            montgomeryOne,
            montgomeryTwo,
            montgomeryTwoSquared);
    }


    private ulong CalculateCycleLengthWithHeuristics(ulong modulus, ulong fallbackOrder = 0UL)
    {
        if (modulus <= 1UL || (modulus & 1UL) == 0UL)
        {
            return fallbackOrder != 0UL ? fallbackOrder : MersenneDivisorCycles.CalculateCycleLength(modulus);
        }

        // When the below is uncommented, the benchmark never completes due to the primality test cost on very large random set.

        // bool isPrime = Open.Numeric.Primes.Prime.Numbers.IsPrime(modulus);
        // // bool isPrime = PrimeTester.IsPrimeInternal(modulus, CancellationToken.None);
        // if (!isPrime)
        // {
        //     return fallbackOrder != 0UL ? fallbackOrder : MersenneDivisorCycles.CalculateCycleLength(modulus);
        // }

#if DEBUG
        Console.WriteLine("Trying heuristic. Prime order calculation");
#endif
        PrimeOrderCalculator.PrimeOrderResult orderResult = PrimeOrderCalculator.Calculate(
            modulus,
            _previousPrimeOrder,
            PrimeOrderCalculator.PrimeOrderSearchConfig.HeuristicDefault);

        if (orderResult.Order != 0UL)
        {
            _previousPrimeOrder = orderResult.Order;
            return orderResult.Order;
        }

        _previousPrimeOrder = null;

        Console.WriteLine($"Heuristic failed for {modulus}, falling back to full cycle calculation");
        return fallbackOrder != 0UL ? fallbackOrder : MersenneDivisorCycles.CalculateCycleLength(modulus);
    }

    private UInt128 CalculateCycleLengthWithHeuristics(UInt128 modulus)
    {
        if (modulus <= UInt128.One || (modulus & UInt128.One) == UInt128.Zero)
        {
            return MersenneDivisorCycles.GetCycle(modulus);
        }

#if DEBUG
        Console.WriteLine("Trying heuristic. Wide prime order calculation");
#endif
        PrimeOrderCalculator.PrimeOrderResultWide orderResult = PrimeOrderCalculator.Calculate(
            modulus,
            _previousWidePrimeOrder,
            PrimeOrderCalculator.PrimeOrderSearchConfig.HeuristicDefault);

        if (orderResult.Order != UInt128.Zero)
        {
            _previousWidePrimeOrder = orderResult.Order;
            return orderResult.Order;
        }

        _previousWidePrimeOrder = null;

        Console.WriteLine($"Wide heuristic failed for {modulus}, falling back to full cycle calculation");
        return MersenneDivisorCycles.GetCycle(modulus);
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

