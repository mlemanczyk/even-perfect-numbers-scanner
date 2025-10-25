using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

/// <summary>
/// Benchmarks CPU Montgomery exponentiation helpers for exponents at or above 138 million.
/// Reuses the shared benchmark input cache so every method processes identical samples.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
public class Pow2MontgomeryModWindowedKeepMontgomeryCpuBenchmarks
{
    private Pow2MontgomeryModWindowedBenchmarkInputs _inputs = null!;

    [GlobalSetup]
    public void Setup()
    {
        _inputs = Pow2MontgomeryModWindowedBenchmarkInputsProvider.Instance;
    }

    /// <summary>
    /// Baseline CPU windowed Montgomery exponentiation that converts the residue back to standard form.
    /// </summary>
    [Benchmark(Baseline = true)]
    public ulong WindowedMontgomeryConvert()
    {
        ulong checksum = 0UL;
        Pow2MontgomeryModWindowedBenchmarkCase[] cases = _inputs.Cases;

        for (int i = 0; i < cases.Length; i++)
        {
            ref readonly Pow2MontgomeryModWindowedBenchmarkCase current = ref cases[i];
            checksum ^= current.Exponent.Pow2MontgomeryModWindowedCpu(current.Divisor, keepMontgomery: false);
        }

        return checksum;
    }

    /// <summary>
    /// CPU windowed Montgomery exponentiation that keeps the residue in the Montgomery domain.
    /// </summary>
    [Benchmark]
    public ulong WindowedMontgomeryKeep()
    {
        ulong checksum = 0UL;
        Pow2MontgomeryModWindowedBenchmarkCase[] cases = _inputs.Cases;

        for (int i = 0; i < cases.Length; i++)
        {
            ref readonly Pow2MontgomeryModWindowedBenchmarkCase current = ref cases[i];
            checksum ^= current.Exponent.Pow2MontgomeryModWindowedCpu(current.Divisor, keepMontgomery: true);
        }

        return checksum;
    }

    /// <summary>
    /// Uses the known cycle length to fold the exponent before running the windowed Montgomery ladder.
    /// </summary>
    [Benchmark]
    public ulong WindowedMontgomeryWithCycle()
    {
        ulong checksum = 0UL;
        Pow2MontgomeryModWindowedBenchmarkCase[] cases = _inputs.Cases;

        for (int i = 0; i < cases.Length; i++)
        {
            ref readonly Pow2MontgomeryModWindowedBenchmarkCase current = ref cases[i];
            checksum ^= current.Exponent.Pow2MontgomeryModWithCycleCpu(current.CycleLength, current.Divisor);
        }

        return checksum;
    }

    /// <summary>
    /// Runs the windowed Montgomery ladder on the reduced exponent that the divisor cycle exposes.
    /// </summary>
    [Benchmark]
    public ulong WindowedMontgomeryFromCycleRemainder()
    {
        ulong checksum = 0UL;
        Pow2MontgomeryModWindowedBenchmarkCase[] cases = _inputs.Cases;

        for (int i = 0; i < cases.Length; i++)
        {
            ref readonly Pow2MontgomeryModWindowedBenchmarkCase current = ref cases[i];
            checksum ^= current.ReducedExponent.Pow2MontgomeryModFromCycleRemainderCpu(current.Divisor);
        }

        return checksum;
    }
}
