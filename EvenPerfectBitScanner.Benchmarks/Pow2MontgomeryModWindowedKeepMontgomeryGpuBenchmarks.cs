using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

/// <summary>
/// Benchmarks GPU Montgomery exponentiation helpers for exponents at or above 138 million.
/// The sample set is generated once and reused across all benchmark methods so every variant
/// processes the same data without incurring repeated initialization costs.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
public class Pow2MontgomeryModWindowedKeepMontgomeryGpuBenchmarks
{
    private BenchmarkInputs _inputs = null!;

    [GlobalSetup]
    public void Setup()
    {
        _inputs = BenchmarkInputsProvider.Instance;
    }

    /// <summary>
    /// Baseline GPU windowed Montgomery exponentiation that converts back to the standard residue.
    /// </summary>
    [Benchmark(Baseline = true)]
    public ulong WindowedMontgomeryConvert()
    {
        ulong checksum = 0UL;
        BenchmarkCase[] cases = _inputs.Cases;

        for (int i = 0; i < cases.Length; i++)
        {
            ref readonly BenchmarkCase current = ref cases[i];
            checksum ^= current.Exponent.Pow2MontgomeryModWindowedGpu(current.Divisor, keepMontgomery: false);
        }

        return checksum;
    }

    /// <summary>
    /// GPU windowed Montgomery exponentiation that keeps the result in the Montgomery domain.
    /// </summary>
    [Benchmark]
    public ulong WindowedMontgomeryKeep()
    {
        ulong checksum = 0UL;
        BenchmarkCase[] cases = _inputs.Cases;

        for (int i = 0; i < cases.Length; i++)
        {
            ref readonly BenchmarkCase current = ref cases[i];
            checksum ^= current.Exponent.Pow2MontgomeryModWindowedGpu(current.Divisor, keepMontgomery: true);
        }

        return checksum;
    }

    /// <summary>
    /// GPU windowed modular exponentiation without Montgomery reduction for comparison.
    /// </summary>
    [Benchmark]
    public ulong WindowedStandardMod()
    {
        ulong checksum = 0UL;
        BenchmarkCase[] cases = _inputs.Cases;

        for (int i = 0; i < cases.Length; i++)
        {
            ref readonly BenchmarkCase current = ref cases[i];
            checksum ^= current.Exponent.Pow2ModWindowedGpu(current.Modulus);
        }

        return checksum;
    }

    private readonly struct BenchmarkCase
    {
        public BenchmarkCase(ulong exponent, ulong modulus, in MontgomeryDivisorData divisor)
        {
            Exponent = exponent;
            Modulus = modulus;
            Divisor = divisor;
        }

        public ulong Exponent { get; }

        public ulong Modulus { get; }

        public MontgomeryDivisorData Divisor { get; }
    }

    private sealed class BenchmarkInputs
    {
        public BenchmarkInputs(BenchmarkCase[] cases)
        {
            Cases = cases;
        }

        public BenchmarkCase[] Cases { get; }
    }

    private readonly struct BenchmarkSeed
    {
        public BenchmarkSeed(ulong exponent, ulong multiplier)
        {
            Exponent = exponent;
            Multiplier = multiplier;
        }

        public ulong Exponent { get; }

        public ulong Multiplier { get; }
    }

    private static class BenchmarkInputsProvider
    {
        private static readonly Lazy<BenchmarkInputs> Cache = new(Create);

        public static BenchmarkInputs Instance => Cache.Value;

        private static BenchmarkInputs Create()
        {
            BenchmarkSeed[] seeds =
            {
                new BenchmarkSeed(138_000_001UL, 1UL),
                new BenchmarkSeed(150_000_013UL, 2UL),
                new BenchmarkSeed(175_000_019UL, 3UL),
                new BenchmarkSeed(190_000_151UL, 4UL),
                new BenchmarkSeed(210_000_089UL, 5UL),
                new BenchmarkSeed(230_000_039UL, 6UL),
                new BenchmarkSeed(260_000_111UL, 7UL),
                new BenchmarkSeed(300_000_007UL, 8UL)
            };

            var cases = new BenchmarkCase[seeds.Length];
            for (int i = 0; i < seeds.Length; i++)
            {
                BenchmarkSeed seed = seeds[i];
                ulong modulus = 2UL * seed.Multiplier * seed.Exponent + 1UL;
                MontgomeryDivisorData divisor = MontgomeryDivisorData.FromModulus(modulus);
                cases[i] = new BenchmarkCase(seed.Exponent, modulus, divisor);
            }

            return new BenchmarkInputs(cases);
        }
    }
}
