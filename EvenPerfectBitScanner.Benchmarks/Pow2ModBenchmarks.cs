using System;
using System.Buffers.Binary;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using System.Numerics;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;

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
public class Pow2ModBenchmarks
{
    private const int SampleCount = 256;
    private const ulong OneShiftedLeft63 = 1UL << 63;
    private const long OneShiftedLeft60 = 1L << 60;
    private const ulong Pow2WindowFallbackThreshold = 32UL;

    private MontgomeryDivisorData[] _smallDivisors = Array.Empty<MontgomeryDivisorData>();
    private MontgomeryDivisorData[] _largeDivisors = Array.Empty<MontgomeryDivisorData>();
    private MontgomeryDivisorData[] _veryLargeDivisors = Array.Empty<MontgomeryDivisorData>();
    private UInt128[] _wideModuli = Array.Empty<UInt128>();
    private UInt128[] _wideExponents = Array.Empty<UInt128>();
    private ulong[] _smallExponents = Array.Empty<ulong>();
    private ulong[] _largeExponents = Array.Empty<ulong>();
    private ulong[] _veryLargeExponents = Array.Empty<ulong>();
    private ulong[] _smallCycles = Array.Empty<ulong>();
    private ulong[] _largeCycles = Array.Empty<ulong>();
    private ulong[] _veryLargeCycles = Array.Empty<ulong>();
    private UInt128[] _wideCycles = Array.Empty<UInt128>();
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
        SampleCollections cache = SampleCollectionsProvider.Instance;
        _smallDivisors = cache.SmallDivisors;
        _largeDivisors = cache.LargeDivisors;
        _veryLargeDivisors = cache.VeryLargeDivisors;
        _wideModuli = cache.WideModuli;
        _wideExponents = cache.WideExponents;
        _smallExponents = cache.SmallExponents;
        _largeExponents = cache.LargeExponents;
        _veryLargeExponents = cache.VeryLargeExponents;
        _smallCycles = cache.SmallCycles;
        _largeCycles = cache.LargeCycles;
        _veryLargeCycles = cache.VeryLargeCycles;
        _wideCycles = cache.WideCycles;
    }

    /// <summary>
    /// Baseline right-to-left Montgomery ladder; measured 36.36 μs on the small sample set and 117.32 μs on the large one.
    /// </summary>
    [Benchmark]
    public ulong MontgomeryWindowedCpu()
    {
        GetData(out var exponents, out var divisors, out _);
        ulong checksum = 0UL;

        for (int i = 0; i < SampleCount; i++)
        {
            checksum ^= exponents[i].Pow2MontgomeryModWindowedCpu(divisors[i], keepMontgomery: false);
        }

        return checksum;
    }

    /// <summary>
    /// Baseline right-to-left Montgomery ladder; measured ??.?? μs on the small sample set and ??.?? μs on the large one.
	/// 
	/// GPU benchmark failed running, hanging up early before reaching workload testing.
    /// </summary>
    // [Benchmark]
    public ulong MontgomeryWindowedGpu()
    {
        GetData(out var exponents, out var divisors, out _);
        ulong checksum = 0UL;

        for (int i = 0; i < SampleCount; i++)
        {
            checksum ^= exponents[i].Pow2MontgomeryModWindowedGpu(divisors[i], keepMontgomery: false);
        }

        return checksum;
    }

    /// <summary>
    /// Uses a known divisor cycle to fold the exponent first, dropping runtimes to 26.04 μs on small inputs (0.72× baseline) and 9.53 μs on large inputs (0.08× baseline).
    /// </summary>
    [Benchmark]
    public ulong MontgomeryWindowedWithCycleCpu()
    {
        GetData(out var exponents, out var divisors, out var cycles);
        ulong checksum = 0UL;

        for (int i = 0; i < SampleCount; i++)
        {
            checksum ^= exponents[i].Pow2MontgomeryModWithCycleCpu(cycles[i], divisors[i]);
        }

        return checksum;
    }

    /// <summary>
    /// Uses a known divisor cycle to fold the exponent first, dropping runtimes to 26.04 μs on small inputs (0.72× baseline) and 9.53 μs on large inputs (0.08× baseline).
	/// 
	/// GPU benchmark failed running, hanging up early before reaching workload testing.
    /// </summary>
    // [Benchmark]
    public ulong MontgomeryWindowedWithCycleGpu()
    {
        GetData(out var exponents, out var divisors, out var cycles);
        ulong checksum = 0UL;

        for (int i = 0; i < SampleCount; i++)
        {
            checksum ^= exponents[i].Pow2MontgomeryModWithCycleGpu(cycles[i], divisors[i]);
        }

        return checksum;
    }

    /// <summary>
    /// Montgomery reduction with the cycle computed on the fly using the GPU helper; costs 54.09 ms on the small benchmark due to
    /// cycle discovery, but drops to 12.91 μs on the large set (0.11× baseline).
	/// 
	/// GPU benchmark failed running, hanging up early before reaching workload testing.
    /// </summary>
    // [Benchmark]
    public ulong MontgomeryWithGpuCycleComputationCpu()
    {
        GetData(out var exponents, out var divisors, out _);
        ulong checksum = 0UL;

        for (int i = 0; i < SampleCount; i++)
        {
            ulong cycle = MersenneDivisorCycles.CalculateCycleLengthGpu(divisors[i].Modulus);
            checksum ^= exponents[i].Pow2MontgomeryModWithCycleCpu(cycle, divisors[i]);
        }

        return checksum;
    }

    /// <summary>
    /// Montgomery reduction with the cycle computed on the fly using the GPU helper; costs 54.09 ms on the small benchmark due to
    /// cycle discovery, but drops to 12.91 μs on the large set (0.11× baseline).
	/// 
	/// GPU benchmark failed running, hanging up early before reaching workload testing.
    /// </summary>
    // [Benchmark]
    public ulong MontgomeryWithGpuCycleComputationGpu()
    {
        GetData(out var exponents, out var divisors, out _);
        ulong checksum = 0UL;

        for (int i = 0; i < SampleCount; i++)
        {
            ulong cycle = MersenneDivisorCycles.CalculateCycleLengthGpu(divisors[i].Modulus);
            checksum ^= exponents[i].Pow2MontgomeryModWithCycleGpu(cycle, divisors[i]);
        }

        return checksum;
    }

    [Benchmark]
    public ulong MontgomeryWithHeuristicCycleComputationCpu()
    {
        GetData(out var exponents, out var divisors, out _);
        ulong checksum = 0UL;

        for (int i = 0; i < SampleCount; i++)
        {
            ulong cycle = CalculateCycleLengthWithHeuristics(divisors[i].Modulus);
            checksum ^= exponents[i].Pow2MontgomeryModWithCycleCpu(cycle, divisors[i]);
        }

        return checksum;
    }

	/// <summary>
	/// GPU benchmark failed running, hanging up early before reaching workload testing.
	/// </summary>
	/// <returns></returns>
	/// // [Benchmark]
	public ulong MontgomeryWithHeuristicCycleComputationGpu()
	{
		GetData(out var exponents, out var divisors, out _);
		ulong checksum = 0UL;

		for (int i = 0; i < SampleCount; i++)
		{
			ulong cycle = CalculateCycleLengthWithHeuristics(divisors[i].Modulus);
			checksum ^= exponents[i].Pow2MontgomeryModWithCycleGpu(cycle, divisors[i]);
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
            checksum ^= cycle;
        }

        return checksum;
    }

    /// <summary>
    /// Left-to-right Montgomery scan that trails the baseline: 39.11 μs on small inputs (1.08×) and 126.38 μs on large ones (1.08×).
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
    /// Processes four exponent bits per loop, finishing at 38.74 μs on small cases (1.07× baseline) and 120.11 μs on large ones (1.02×).
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
    /// Windowed Montgomery ladder (width 5) that lags behind the baseline at 52.82 μs on small samples (1.45×) and 124.13 μs on large samples (1.06×).
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
    /// Classic binary ladder without Montgomery reduction; runs 24.75 μs on small inputs (0.68× baseline) but collapses to 295.29 μs on large ones (2.52×).
    /// </summary>
    [Benchmark(Baseline = true)]
    public ulong StandardPow2Mod()
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
    /// Precomputes all squarings once; 30.18 μs on small inputs (0.83× baseline) but 289.80 μs on large ones (2.47×).
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
    /// Relies on <see cref="BigInteger.ModPow"/>; fastest on small inputs at 17.21 μs (0.47× baseline) but slowest on large ones at 562.78 μs (4.80×).
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

    /// <summary>
    /// Current UInt128-native windowed Montgomery ladder that avoids GpuUInt128 on CPU paths.
    /// </summary>
    [Benchmark]
    public UInt128 MontgomeryWindowedUInt128()
    {
        UInt128 checksum = UInt128.Zero;

        for (int i = 0; i < SampleCount; i++)
        {
            checksum ^= _wideExponents[i].Pow2MontgomeryModWindowed(_wideModuli[i]);
        }

        return checksum;
    }

    /// <summary>
    /// Previous CPU shim that reused the GPU-oriented GpuUInt128 implementation.
    /// </summary>
    [Benchmark]
    public UInt128 MontgomeryWindowedGpuShimUInt128()
    {
        UInt128 checksum = UInt128.Zero;

        for (int i = 0; i < SampleCount; i++)
        {
            checksum ^= Pow2MontgomeryModWindowedGpuShim(_wideExponents[i], _wideModuli[i]);
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

    private sealed class SampleCollections
    {
        public SampleCollections(
            MontgomeryDivisorData[] smallDivisors,
            MontgomeryDivisorData[] largeDivisors,
            MontgomeryDivisorData[] veryLargeDivisors,
            UInt128[] wideModuli,
            UInt128[] wideExponents,
            ulong[] smallExponents,
            ulong[] largeExponents,
            ulong[] veryLargeExponents,
            ulong[] smallCycles,
            ulong[] largeCycles,
            ulong[] veryLargeCycles,
            UInt128[] wideCycles)
        {
            SmallDivisors = smallDivisors;
            LargeDivisors = largeDivisors;
            VeryLargeDivisors = veryLargeDivisors;
            WideModuli = wideModuli;
            WideExponents = wideExponents;
            SmallExponents = smallExponents;
            LargeExponents = largeExponents;
            VeryLargeExponents = veryLargeExponents;
            SmallCycles = smallCycles;
            LargeCycles = largeCycles;
            VeryLargeCycles = veryLargeCycles;
            WideCycles = wideCycles;
        }

        public MontgomeryDivisorData[] SmallDivisors { get; }
        public MontgomeryDivisorData[] LargeDivisors { get; }
        public MontgomeryDivisorData[] VeryLargeDivisors { get; }
        public UInt128[] WideModuli { get; }
        public UInt128[] WideExponents { get; }
        public ulong[] SmallExponents { get; }
        public ulong[] LargeExponents { get; }
        public ulong[] VeryLargeExponents { get; }
        public ulong[] SmallCycles { get; }
        public ulong[] LargeCycles { get; }
        public ulong[] VeryLargeCycles { get; }
        public UInt128[] WideCycles { get; }
    }

    private static class SampleCollectionsProvider
    {
        private static readonly Lazy<SampleCollections> Cache = new(Create);

        public static SampleCollections Instance => Cache.Value;

        private static SampleCollections Create()
        {
            Console.WriteLine("Preparing power-of-two modulus benchmark sample collections...");

            Random random = new(13);
            Random heuristicRandom = new(113);
            Random wideRandom = new(19);

            var smallDivisors = new MontgomeryDivisorData[SampleCount];
            var largeDivisors = new MontgomeryDivisorData[SampleCount];
            var veryLargeDivisors = new MontgomeryDivisorData[SampleCount];
            var wideModuli = new UInt128[SampleCount];
            var wideExponents = new UInt128[SampleCount];
            var smallExponents = new ulong[SampleCount];
            var largeExponents = new ulong[SampleCount];
            var veryLargeExponents = new ulong[SampleCount];
            var smallCycles = new ulong[SampleCount];
            var largeCycles = new ulong[SampleCount];
            var veryLargeCycles = new ulong[SampleCount];
            var wideCycles = new UInt128[SampleCount];

            ulong? previousPrimeOrder = null;
            UInt128? previousWidePrimeOrder = null;

            for (int i = 0; i < SampleCount; i++)
            {
                ulong smallModulus = NextSmallOddModulus(random);
                smallDivisors[i] = CreateMontgomeryDivisorData(smallModulus);
                smallExponents[i] = NextSmallExponent(random);
                smallCycles[i] = MersenneDivisorCycles.CalculateCycleLength(smallModulus, MontgomeryDivisorData.FromModulus(smallModulus));

                (ulong largeModulus, ulong largeCycle) = NextLargeModulusAndCycle(random);
                largeDivisors[i] = CreateMontgomeryDivisorData(largeModulus);
                largeExponents[i] = NextLargeExponent(random);
                largeCycles[i] = largeCycle;

                ulong veryLargeModulus = NextVeryLargeOddModulus(heuristicRandom);
                veryLargeDivisors[i] = CreateMontgomeryDivisorData(veryLargeModulus);
                veryLargeExponents[i] = NextVeryLargeExponent(heuristicRandom);
                veryLargeCycles[i] = CalculateCycleLengthWithHeuristics(veryLargeModulus, ref previousPrimeOrder);

                UInt128 wideModulus = NextWideOddModulus(wideRandom);
                wideModuli[i] = wideModulus;
                wideExponents[i] = NextWideExponent(wideRandom);
                wideCycles[i] = CalculateCycleLengthWithHeuristics(wideModulus, ref previousWidePrimeOrder);
            }

            SampleCollections collections = new SampleCollections(
                smallDivisors,
                largeDivisors,
                veryLargeDivisors,
                wideModuli,
                wideExponents,
                smallExponents,
                largeExponents,
                veryLargeExponents,
                smallCycles,
                largeCycles,
                veryLargeCycles,
                wideCycles);

            Console.WriteLine("Finished preparing power-of-two modulus benchmark sample collections.");

            return collections;
        }

        private static ulong NextSmallExponent(Random random) => (ulong)random.NextInt64(1L, 1_000_000L);

        private static ulong NextLargeExponent(Random random) => (ulong)random.NextInt64(OneShiftedLeft60, long.MaxValue);

        private static ulong NextVeryLargeExponent(Random random)
        {
            Span<byte> buffer = stackalloc byte[8];
            random.NextBytes(buffer);
            ulong value = checked(BinaryPrimitives.ReadUInt64LittleEndian(buffer) | OneShiftedLeft63);
            return value;
        }

        private static ulong NextSmallOddModulus(Random random)
        {
            ulong value = (ulong)random.NextInt64(3L, 1_000_000L);
            return value | 1UL;
        }

        private static (ulong Modulus, ulong CycleLength) NextLargeModulusAndCycle(Random random)
        {
            int bitLength = random.Next(48, 64);
            ulong modulus = (1UL << bitLength) - 1UL;
            return (modulus, (ulong)bitLength);
        }

        private static ulong NextVeryLargeOddModulus(Random random)
        {
            Span<byte> buffer = stackalloc byte[8];
            random.NextBytes(buffer);
            ulong value = BinaryPrimitives.ReadUInt64LittleEndian(buffer) | OneShiftedLeft63 | 1UL;
            return value;
        }

        private static UInt128 NextWideOddModulus(Random random)
        {
            Span<byte> buffer = stackalloc byte[16];
            random.NextBytes(buffer);

            ulong low = BinaryPrimitives.ReadUInt64LittleEndian(buffer);
            ulong high = BinaryPrimitives.ReadUInt64LittleEndian(buffer[8..]);
            if (high == 0UL)
            {
                high = 1UL;
            }

            UInt128 modulus = ((UInt128)high << 64) | low;
            modulus |= UInt128.One;

            if (modulus <= UInt128.One)
            {
                modulus += 2UL;
            }

            return modulus;
        }

        private static UInt128 NextWideExponent(Random random)
        {
            Span<byte> buffer = stackalloc byte[16];
            random.NextBytes(buffer);
            ulong low = BinaryPrimitives.ReadUInt64LittleEndian(buffer);
            ulong high = BinaryPrimitives.ReadUInt64LittleEndian(buffer[8..]);
            if (high == 0UL)
            {
                high = 1UL;
            }

            return ((UInt128)high << 64) | low;
        }

        private static ulong CalculateCycleLengthWithHeuristics(ulong modulus, ref ulong? previousPrimeOrder)
        {
            MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(modulus);
            if (modulus <= 1UL || (modulus & 1UL) == 0UL)
            {
                return MersenneDivisorCycles.CalculateCycleLength(modulus, divisorData);
            }

            ulong order = PrimeOrderCalculator.Calculate(
                modulus,
                previousPrimeOrder,
                divisorData,
                PrimeOrderCalculator.PrimeOrderSearchConfig.HeuristicDefault,
                PrimeOrderCalculator.PrimeOrderHeuristicDevice.Cpu);

            if (order != 0UL)
            {
                previousPrimeOrder = order;
                return order;
            }

            previousPrimeOrder = null;
            return MersenneDivisorCycles.CalculateCycleLength(modulus, divisorData);
        }

        private static UInt128 CalculateCycleLengthWithHeuristics(UInt128 modulus, ref UInt128? previousWidePrimeOrder)
        {
            if (modulus <= UInt128.One || (modulus & UInt128.One) == UInt128.Zero)
            {
                return MersenneDivisorCycles.GetCycle(modulus);
            }

            UInt128 order = PrimeOrderCalculator.Calculate(
                modulus,
                previousWidePrimeOrder,
                PrimeOrderCalculator.PrimeOrderSearchConfig.HeuristicDefault,
                PrimeOrderCalculator.PrimeOrderHeuristicDevice.Cpu);

            if (order != UInt128.Zero)
            {
                previousWidePrimeOrder = order;
                return order;
            }

            previousWidePrimeOrder = null;
            return MersenneDivisorCycles.GetCycle(modulus);
        }
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
        MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(modulus);
        if (modulus <= 1UL || (modulus & 1UL) == 0UL)
        {
            return fallbackOrder != 0UL ? fallbackOrder : MersenneDivisorCycles.CalculateCycleLength(modulus, divisorData);
        }

#if DEBUG
        Console.WriteLine("Trying heuristic. Prime order calculation");
#endif
        ulong order = PrimeOrderCalculator.Calculate(
            modulus,
            _previousPrimeOrder,
            divisorData,
            PrimeOrderCalculator.PrimeOrderSearchConfig.HeuristicDefault,
            PrimeOrderCalculator.PrimeOrderHeuristicDevice.Cpu);

        if (order != 0UL)
        {
            _previousPrimeOrder = order;
            return order;
        }

        _previousPrimeOrder = null;

        Console.WriteLine($"Heuristic failed for {modulus}, falling back to full cycle calculation");
        return fallbackOrder != 0UL ? fallbackOrder : MersenneDivisorCycles.CalculateCycleLength(modulus, divisorData);
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
        UInt128 order = PrimeOrderCalculator.Calculate(
            modulus,
            _previousWidePrimeOrder,
            PrimeOrderCalculator.PrimeOrderSearchConfig.HeuristicDefault,
            PrimeOrderCalculator.PrimeOrderHeuristicDevice.Cpu);

        if (order != UInt128.Zero)
        {
            _previousWidePrimeOrder = order;
            return order;
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

    private static UInt128 Pow2MontgomeryModWindowedGpuShim(UInt128 exponent, UInt128 modulus)
    {
        if (modulus == UInt128.One)
        {
            return UInt128.Zero;
        }

        if (exponent == UInt128.Zero)
        {
            return UInt128.One % modulus;
        }

        GpuUInt128 modulusGpu = new(modulus);
        GpuUInt128 exponentGpu = new(exponent);
        GpuUInt128 baseValue = new(2UL);
        if (baseValue.CompareTo(modulusGpu) >= 0)
        {
            baseValue.Sub(modulusGpu);
        }

        if (ShouldUseSingleBit(exponentGpu))
        {
            GpuUInt128 singleBitResult = Pow2MontgomeryModSingleBit(exponentGpu, modulusGpu, baseValue);
            return (UInt128)singleBitResult;
        }

        int bitLength = exponentGpu.GetBitLength();
        int windowSize = GetWindowSize(bitLength);
        int oddPowerCount = 1 << (windowSize - 1);

        Span<GpuUInt128> oddPowers = stackalloc GpuUInt128[oddPowerCount];
        InitializeOddPowers(baseValue, modulusGpu, oddPowers);

        GpuUInt128 result = GpuUInt128.One;
        int index = bitLength - 1;

        while (index >= 0)
        {
            if (!IsBitSet(exponentGpu, index))
            {
                result.MulMod(result, modulusGpu);
                index--;
                continue;
            }

            int windowStart = index - windowSize + 1;
            if (windowStart < 0)
            {
                windowStart = 0;
            }

            while (!IsBitSet(exponentGpu, windowStart))
            {
                windowStart++;
            }

            int windowBitCount = index - windowStart + 1;
            for (int square = 0; square < windowBitCount; square++)
            {
                result.MulMod(result, modulusGpu);
            }

            ulong windowValue = ExtractWindowValue(exponentGpu, windowStart, windowBitCount);
            int tableIndex = (int)((windowValue - 1UL) >> 1);
            GpuUInt128 factor = oddPowers[tableIndex];
            result.MulMod(factor, modulusGpu);

            index = windowStart - 1;
        }

        return (UInt128)result;
    }

    private static bool ShouldUseSingleBit(GpuUInt128 exponent)
    {
        if (exponent.High == 0UL && exponent.Low <= Pow2WindowFallbackThreshold)
        {
            return true;
        }

        return false;
    }

    private static int GetWindowSize(int bitLength)
    {
        if (bitLength <= 8)
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

        return 8;
    }

    private static void InitializeOddPowers(GpuUInt128 baseValue, GpuUInt128 modulus, Span<GpuUInt128> oddPowers)
    {
        oddPowers[0] = baseValue;
        if (oddPowers.Length == 1)
        {
            return;
        }

        baseValue.MulMod(baseValue, modulus);
        for (int i = 1; i < oddPowers.Length; i++)
        {
            oddPowers[i] = oddPowers[i - 1];
            oddPowers[i].MulMod(baseValue, modulus);
        }
    }

    private static GpuUInt128 Pow2MontgomeryModSingleBit(GpuUInt128 exponent, GpuUInt128 modulus, GpuUInt128 baseValue)
    {
        GpuUInt128 result = GpuUInt128.One;

        while (!exponent.IsZero)
        {
            if ((exponent.Low & 1UL) != 0UL)
            {
                result.MulMod(baseValue, modulus);
            }

            exponent.ShiftRight(1);
            if (exponent.IsZero)
            {
                break;
            }

            baseValue.MulMod(baseValue, modulus);
        }

        return result;
    }

    private static bool IsBitSet(GpuUInt128 value, int bitIndex)
    {
        if (bitIndex >= 64)
        {
            return ((value.High >> (bitIndex - 64)) & 1UL) != 0UL;
        }

        return ((value.Low >> bitIndex) & 1UL) != 0UL;
    }

    private static ulong ExtractWindowValue(GpuUInt128 exponent, int windowStart, int windowBitCount)
    {
        if (windowStart != 0)
        {
            GpuUInt128 shifted = exponent;
            shifted.ShiftRight(windowStart);
            ulong mask = (1UL << windowBitCount) - 1UL;
            return shifted.Low & mask;
        }

        ulong directMask = (1UL << windowBitCount) - 1UL;
        return exponent.Low & directMask;
    }

    private static ulong MultiplyMod(ulong x, ulong y, ulong modulus)
    {
        UInt128 product = (UInt128)x * y;
        return (ulong)(product % modulus);
    }
}
