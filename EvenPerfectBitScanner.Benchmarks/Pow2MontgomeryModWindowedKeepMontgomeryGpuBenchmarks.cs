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
    private Pow2MontgomeryModWindowedBenchmarkInputs _inputs = null!;

    [GlobalSetup]
    public void Setup()
    {
        _inputs = Pow2MontgomeryModWindowedBenchmarkInputsProvider.Instance;
    }

    /// <summary>
    /// Baseline GPU windowed Montgomery exponentiation that converts back to the standard residue.
    /// </summary>
    [Benchmark(Baseline = true)]
    public ulong WindowedMontgomeryConvert()
    {
        ulong checksum = 0UL;
        Pow2MontgomeryModWindowedBenchmarkCase[] cases = _inputs.Cases;

        for (int i = 0; i < cases.Length; i++)
        {
            ref readonly Pow2MontgomeryModWindowedBenchmarkCase current = ref cases[i];
            checksum ^= current.Exponent.Pow2MontgomeryModWindowedConvertGpu(current.Divisor);
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
        Pow2MontgomeryModWindowedBenchmarkCase[] cases = _inputs.Cases;

        for (int i = 0; i < cases.Length; i++)
        {
            ref readonly Pow2MontgomeryModWindowedBenchmarkCase current = ref cases[i];
            checksum ^= current.Exponent.Pow2MontgomeryModWindowedKeepGpu(current.Divisor);
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
        Pow2MontgomeryModWindowedBenchmarkCase[] cases = _inputs.Cases;

        for (int i = 0; i < cases.Length; i++)
        {
            ref readonly Pow2MontgomeryModWindowedBenchmarkCase current = ref cases[i];
            checksum ^= current.Exponent.Pow2ModWindowedGpu(current.Modulus);
        }

        return checksum;
    }
}
