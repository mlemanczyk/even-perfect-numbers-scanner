using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace EvenPerfectBitScanner.Benchmarks;

internal sealed class Pow2MontgomeryModWindowedBenchmarkInputs
{
    public Pow2MontgomeryModWindowedBenchmarkInputs(Pow2MontgomeryModWindowedBenchmarkCase[] cases)
    {
        Cases = cases;
    }

    public Pow2MontgomeryModWindowedBenchmarkCase[] Cases { get; }
}

internal readonly struct Pow2MontgomeryModWindowedBenchmarkCase
{
    public Pow2MontgomeryModWindowedBenchmarkCase(
        ulong exponent,
        ulong modulus,
        in MontgomeryDivisorData divisor,
        ulong cycleLength,
        ulong reducedExponent)
    {
        Exponent = exponent;
        Modulus = modulus;
        Divisor = divisor;
        CycleLength = cycleLength;
        ReducedExponent = reducedExponent;
    }

    public ulong Exponent { get; }

    public ulong Modulus { get; }

    public MontgomeryDivisorData Divisor { get; }

    public ulong CycleLength { get; }

    public ulong ReducedExponent { get; }
}

internal static class Pow2MontgomeryModWindowedBenchmarkInputsProvider
{
    private static readonly Lazy<Pow2MontgomeryModWindowedBenchmarkInputs> Cache = new(Create);

    public static Pow2MontgomeryModWindowedBenchmarkInputs Instance => Cache.Value;

    private static Pow2MontgomeryModWindowedBenchmarkInputs Create()
    {
        Pow2MontgomeryModWindowedBenchmarkSeed[] seeds =
        {
            new Pow2MontgomeryModWindowedBenchmarkSeed(138_000_001UL, 1UL),
            new Pow2MontgomeryModWindowedBenchmarkSeed(150_000_013UL, 2UL),
            new Pow2MontgomeryModWindowedBenchmarkSeed(175_000_019UL, 3UL),
            new Pow2MontgomeryModWindowedBenchmarkSeed(190_000_151UL, 4UL),
            new Pow2MontgomeryModWindowedBenchmarkSeed(210_000_089UL, 5UL),
            new Pow2MontgomeryModWindowedBenchmarkSeed(230_000_039UL, 6UL),
            new Pow2MontgomeryModWindowedBenchmarkSeed(260_000_111UL, 7UL),
            new Pow2MontgomeryModWindowedBenchmarkSeed(300_000_007UL, 8UL)
        };

        var cases = new Pow2MontgomeryModWindowedBenchmarkCase[seeds.Length];
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
        for (int i = 0; i < seeds.Length; i++)
        {
            Pow2MontgomeryModWindowedBenchmarkSeed seed = seeds[i];
            ulong modulus = 2UL * seed.Multiplier * seed.Exponent + 1UL;
            MontgomeryDivisorData divisor = MontgomeryDivisorDataPool.Shared.FromModulus(modulus);
            ulong cycleLength = MersenneDivisorCycles.CalculateCycleLengthGpu(gpu, modulus, divisor);
            ulong reducedExponent = seed.Exponent % cycleLength;
            cases[i] = new Pow2MontgomeryModWindowedBenchmarkCase(
                seed.Exponent,
                modulus,
                divisor,
                cycleLength,
                reducedExponent);
        }

		PrimeOrderCalculatorAccelerator.Return(gpu);
        return new Pow2MontgomeryModWindowedBenchmarkInputs(cases);
    }

    private readonly struct Pow2MontgomeryModWindowedBenchmarkSeed
    {
        public Pow2MontgomeryModWindowedBenchmarkSeed(ulong exponent, ulong multiplier)
        {
            Exponent = exponent;
            Multiplier = multiplier;
        }

        public ulong Exponent { get; }

        public ulong Multiplier { get; }
    }
}
