using FluentAssertions;
using Xunit;
using System.Reflection;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;
using EvenPerfectBitScanner.Candidates;
using EvenPerfectBitScanner.Candidates.Transforms;
using System.Numerics;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace EvenPerfectBitScanner.Tests;

[Trait("Category", "Fast")]
public class ProgramTests
{
    private static readonly FieldInfo CliArgumentsField = typeof(Program).GetField("_cliArguments", BindingFlags.NonPublic | BindingFlags.Static)!;
    private static readonly FieldInfo DivisorField = typeof(Program).GetField("_divisor", BindingFlags.NonPublic | BindingFlags.Static)!;
    private static readonly CliArguments DefaultCliArguments = CliArguments.Parse(Array.Empty<string>());

    public ProgramTests()
    {
        ResetCliArguments();
    }

    private static void ConfigureCliArguments(params string[] args)
    {
        SetCliArguments(CliArguments.Parse(args));
    }

    private static void SetCliArguments(CliArguments arguments)
    {
        CliArgumentsField.SetValue(null, arguments);
    }

    private static void ResetCliArguments()
    {
        SetCliArguments(DefaultCliArguments);
        DivisorField.SetValue(null, UInt128.Zero);
    }

    [Fact]
    public void CountOnes_returns_correct_count()
    {
        BitOperations.PopCount(0b101010).Should().Be(3);
    }

    [Theory]
    [InlineData(2UL, true)]
    [InlineData(4UL, false)]
    [InlineData(9UL, false)]
    [InlineData(11UL, true)]
    [InlineData(81UL, false)]
    public void IsPrime_identifies_primes_correctly(ulong n, bool expected)
    {
        PrimeTester.IsPrimeCpu(n).Should().Be(expected);
    }

    [Fact]
    public void TransformPAdd_moves_to_next_candidate()
    {
        ulong remainder = 3UL;
        bool limit = false;
        CandidateAddTransform.Transform(3UL, ref remainder, ref limit).Should().Be(5UL);
        CandidateAddTransform.Transform(5UL, ref remainder, ref limit).Should().Be(7UL);
        CandidateAddTransform.Transform(7UL, ref remainder, ref limit).Should().Be(11UL);
        limit.Should().BeFalse();
    }

    [Fact]
    public void TransformPBit_appends_one_bit_and_skips_to_candidate()
    {
        ulong remainder = 5UL;
        bool limit = false;
        CandidateBitTransform.Transform(5UL, ref remainder, ref limit).Should().Be(13UL);
        limit.Should().BeFalse();
    }

    [Fact]
    public void TransformPBit_detects_overflow_and_stops()
    {
        bool limit = false;
        ulong start = (ulong.MaxValue >> 1) + 1UL;
        ulong remainder = start % 6UL;
        CandidateBitTransform.Transform(start, ref remainder, ref limit);
        limit.Should().BeTrue();
    }

    [Fact]
    public void Gcd_filter_detects_some_composites()
    {
        15UL.IsCompositeByGcd().Should().BeTrue();
        5UL.IsCompositeByGcd().Should().BeFalse();
    }

    [Fact]
    public void Divisor_mode_scans_dynamic_cycles_when_none_specified()
    {
        ConfigureCliArguments("--mersenne=divisor");
        var testerField = typeof(Program).GetField("_divisorTester", BindingFlags.NonPublic | BindingFlags.Static)!;
        var candidatesField = typeof(MersenneNumberDivisorGpuTester).GetField("_divisorCandidates", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;

        testerField.SetValue(null, new MersenneNumberDivisorGpuTester());
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        CandidatesCalculator.OverrideResidueTrackers(new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, 2UL, true), trackAllValues: true));
        candidatesField.SetValue(null, new (ulong, uint)[] { (7UL, 3U), (23UL, 11U) });

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
        try
        {
            Program.IsEvenPerfectCandidate(gpu, 11UL, 32UL, out _, out _).Should().BeFalse();
            Program.IsEvenPerfectCandidate(gpu, 5UL, 32UL, out _, out _).Should().BeTrue();
            Program.IsEvenPerfectCandidate(gpu, 127UL, 32UL, out _, out _).Should().BeTrue();
        }
        finally
        {
			PrimeOrderCalculatorAccelerator.Return(gpu);
            testerField.SetValue(null, null);
            candidatesField.SetValue(null, null);
            primeField.SetValue(null, null);
            CandidatesCalculator.ResetResidueTrackers();
            ResetCliArguments();
        }
    }

    [Fact]
    public void Divisor_mode_sets_detailedCheck_based_on_exhaustion()
    {
        ConfigureCliArguments("--mersenne=divisor");

        var testerField = typeof(Program).GetField("_divisorTester", BindingFlags.NonPublic | BindingFlags.Static)!;
        var candidatesField = typeof(MersenneNumberDivisorGpuTester).GetField("_divisorCandidates", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;

        testerField.SetValue(null, new MersenneNumberDivisorGpuTester());
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        CandidatesCalculator.OverrideResidueTrackers(new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, 2UL, true), trackAllValues: true));
        candidatesField.SetValue(null, Array.Empty<(ulong, uint)>());

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
        try
        {
            Program.IsEvenPerfectCandidate(gpu, 11UL, 0UL, out bool searched, out bool detailedCheck).Should().BeTrue();
            searched.Should().BeTrue();
            detailedCheck.Should().BeFalse();
        }
        finally
        {
			PrimeOrderCalculatorAccelerator.Return(gpu);
            testerField.SetValue(null, null);
            candidatesField.SetValue(null, null);
            primeField.SetValue(null, null);
            CandidatesCalculator.ResetResidueTrackers();
            ResetCliArguments();
        }
    }

    [Fact]
    public void Divisor_mode_accepts_large_search_limit()
    {
        ConfigureCliArguments("--mersenne=divisor");
        DivisorField.SetValue(null, (UInt128)7);

        var testerField = typeof(Program).GetField("_divisorTester", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;

        testerField.SetValue(null, new MersenneNumberDivisorGpuTester());
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        CandidatesCalculator.OverrideResidueTrackers(new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, 2UL, true), trackAllValues: true));

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
        try
        {
            Program.IsEvenPerfectCandidate(gpu, 3UL, 4_000_000_000UL, out bool searched, out bool detailed).Should().BeFalse();
            searched.Should().BeTrue();
            detailed.Should().BeTrue();
        }
        finally
        {
			PrimeOrderCalculatorAccelerator.Return(gpu);
            testerField.SetValue(null, null);
            primeField.SetValue(null, null);
            CandidatesCalculator.ResetResidueTrackers();
            ResetCliArguments();
        }
    }

    [Fact]
    public void Divisor_mode_checks_divisibility_on_gpu()
    {
        ConfigureCliArguments("--mersenne=divisor");
        DivisorField.SetValue(null, (UInt128)23);

        var testerField = typeof(Program).GetField("_divisorTester", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;

        testerField.SetValue(null, new MersenneNumberDivisorGpuTester());
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        CandidatesCalculator.OverrideResidueTrackers(new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, 2UL, true), trackAllValues: true));

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
        try
        {
            Program.IsEvenPerfectCandidate(gpu, 11UL, 0UL, out bool searched, out bool detailed).Should().BeFalse();
            searched.Should().BeTrue();
            detailed.Should().BeTrue();

            Program.IsEvenPerfectCandidate(gpu, 13UL, 0UL, out searched, out detailed).Should().BeTrue();
            searched.Should().BeTrue();
            detailed.Should().BeFalse();
        }
        finally
        {
			PrimeOrderCalculatorAccelerator.Return(gpu);
            testerField.SetValue(null, null);
            primeField.SetValue(null, null);
            CandidatesCalculator.ResetResidueTrackers();
            ResetCliArguments();
        }
    }

    [Fact]
    public void Residue_mode_checks_mersenne_primes_on_gpu()
    {
        ConfigureCliArguments("--mersenne=residue");

        var mersenneField = typeof(Program).GetField("MersenneTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;

        mersenneField.SetValue(null, new ThreadLocal<MersenneNumberTester>(() => new MersenneNumberTester(useIncremental: true, useResidue: true, maxK: 1_000UL), trackAllValues: true));
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        CandidatesCalculator.OverrideResidueTrackers(new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, 2UL, true), trackAllValues: true));

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
        try
        {
            Program.IsEvenPerfectCandidate(gpu, 29UL, 0UL, out bool searched, out bool detailed).Should().BeFalse();
            searched.Should().BeTrue();
            detailed.Should().BeFalse();

            Program.IsEvenPerfectCandidate(gpu, 31UL, 0UL, out searched, out detailed).Should().BeTrue();
            searched.Should().BeTrue();
            detailed.Should().BeTrue();
        }
        finally
        {
			PrimeOrderCalculatorAccelerator.Return(gpu);
            mersenneField.SetValue(null, null);
            primeField.SetValue(null, null);
            CandidatesCalculator.ResetResidueTrackers();
            ResetCliArguments();
        }
    }

    [Fact]
    public void Residue_mode_with_large_divisor_cycle_limits_accepts_known_primes()
    {
        ConfigureCliArguments("--mersenne=residue");

        var mersenneField = typeof(Program).GetField("MersenneTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;

        mersenneField.SetValue(null, new ThreadLocal<MersenneNumberTester>(() => new MersenneNumberTester(
            useIncremental: true,
            useResidue: true,
            maxK: 1_024UL), trackAllValues: true));
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        CandidatesCalculator.OverrideResidueTrackers(new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, 2UL, true), trackAllValues: true));

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
        try
        {
            ulong[] primes = [107UL, 127UL];
            foreach (ulong prime in primes)
            {
                Program.IsEvenPerfectCandidate(gpu, prime, 1024UL, out bool searched, out bool detailed).Should().BeTrue();
                searched.Should().BeTrue();
                detailed.Should().BeTrue();
            }
        }
        finally
        {
			PrimeOrderCalculatorAccelerator.Return(gpu);
            mersenneField.SetValue(null, null);
            primeField.SetValue(null, null);
            CandidatesCalculator.ResetResidueTrackers();
            ResetCliArguments();
        }
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void Residue_mode_accepts_known_primes_in_parallel()
    {
        ConfigureCliArguments("--mersenne=residue");

        var mersenneField = typeof(Program).GetField("MersenneTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;

        mersenneField.SetValue(null, new ThreadLocal<MersenneNumberTester>(() => new MersenneNumberTester(
            useIncremental: true,
            useResidue: true,
            maxK: 5_000_000UL), trackAllValues: true));
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        CandidatesCalculator.OverrideResidueTrackers(new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, 2UL, true), trackAllValues: true));

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
        try
        {
            ulong[] primes = [107UL, 127UL, 521UL, 607UL];
            Parallel.ForEach(primes, prime =>
            {
                Program.IsEvenPerfectCandidate(gpu, prime, 0UL, out bool searched, out bool detailed).Should().BeTrue();
                searched.Should().BeTrue();
                detailed.Should().BeTrue();
            });
        }
        finally
        {
			PrimeOrderCalculatorAccelerator.Return(gpu);
            mersenneField.SetValue(null, null);
            primeField.SetValue(null, null);
            CandidatesCalculator.ResetResidueTrackers();
            ResetCliArguments();
        }
    }

    [Fact]
    public void Residue_mode_with_default_limits_accepts_large_known_primes()
    {
        ConfigureCliArguments("--mersenne=residue");

        var mersenneField = typeof(Program).GetField("MersenneTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;

        mersenneField.SetValue(null, new ThreadLocal<MersenneNumberTester>(() => new MersenneNumberTester(
            useIncremental: true,
            useResidue: true,
            maxK: 5_000_000UL), trackAllValues: true));
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        CandidatesCalculator.OverrideResidueTrackers(new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, 2UL, true), trackAllValues: true));

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
        try
        {
            Program.IsEvenPerfectCandidate(gpu, 107UL, 0UL, out bool searched, out bool detailed).Should().BeTrue();
            searched.Should().BeTrue();
            detailed.Should().BeTrue();

            Program.IsEvenPerfectCandidate(gpu, 127UL, 0UL, out searched, out detailed).Should().BeTrue();
            searched.Should().BeTrue();
            detailed.Should().BeTrue();
        }
        finally
        {
			PrimeOrderCalculatorAccelerator.Return(gpu);
            mersenneField.SetValue(null, null);
            primeField.SetValue(null, null);
            CandidatesCalculator.ResetResidueTrackers();
            ResetCliArguments();
        }
    }

    [Theory]
    [MemberData(nameof(KnownMersennePrimeExponents))]
    public void IsEvenPerfectCandidate_accepts_known_small_mersenne_primes(ulong exponent)
    {
        ConfigureCliArguments("--mersenne=residue");

        var mersenneField = typeof(Program).GetField("MersenneTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;

        mersenneField.SetValue(null, new ThreadLocal<MersenneNumberTester>(() => new MersenneNumberTester(
            useIncremental: true,
            useGpuScan: false,
            useGpuOrder: false,
            useResidue: true,
            maxK: 1_024UL), trackAllValues: true));
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        CandidatesCalculator.OverrideResidueTrackers(new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, exponent, true), trackAllValues: true));

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
        try
        {
            Program.IsEvenPerfectCandidate(gpu, exponent, 0UL, out bool searched, out bool detailedCheck).Should().BeTrue();
            searched.Should().BeTrue();
        }
        finally
        {
			PrimeOrderCalculatorAccelerator.Return(gpu);
            mersenneField.SetValue(null, null);
            primeField.SetValue(null, null);
            CandidatesCalculator.ResetResidueTrackers();
            ResetCliArguments();
        }
    }

    [Theory]
    [MemberData(nameof(KnownMersennePrimeGpuConfigurations))]
    public void IsEvenPerfectCandidate_accepts_known_small_mersenne_primes_on_gpu(
        ulong exponent,
        MersenneGpuCandidateConfig configuration)
    {
        ConfigureCliArguments("--mersenne=residue");

        var mersenneField = typeof(Program).GetField("MersenneTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;

        mersenneField.SetValue(null, new ThreadLocal<MersenneNumberTester>(() => new MersenneNumberTester(
            useIncremental: false,
            kernelType: configuration.KernelType,
            useGpuScan: true,
            useGpuOrder: configuration.UseGpuOrder,
            useResidue: true,
            maxK: configuration.ResidueMaxK), trackAllValues: true));
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        CandidatesCalculator.OverrideResidueTrackers(new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, exponent, true), trackAllValues: true));

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
        try
        {
            Program.IsEvenPerfectCandidate(gpu, exponent, 0UL, out bool searched, out bool detailedCheck).Should().BeTrue();
            searched.Should().BeTrue();
        }
        finally
        {
			PrimeOrderCalculatorAccelerator.Return(gpu);
            mersenneField.SetValue(null, null);
            primeField.SetValue(null, null);
            CandidatesCalculator.ResetResidueTrackers();
            ResetCliArguments();
        }
    }

    public static IEnumerable<object[]> KnownMersennePrimeExponents()
    {
        foreach (ulong exponent in KnownMersennePrimeExponentValues)
        {
            yield return new object[] { exponent };
        }
    }

    public static IEnumerable<object[]> KnownMersennePrimeGpuConfigurations()
    {
        MersenneGpuCandidateConfig[] configs =
        {
            new(GpuKernelType.Incremental, true, 1_024UL),
            new(GpuKernelType.Incremental, false, 2_048UL),
            new(GpuKernelType.Pow2Mod, true, 4_096UL),
            new(GpuKernelType.Pow2Mod, false, 512UL),
        };

        int index = 0;
        foreach (ulong exponent in KnownMersennePrimeExponentValues)
        {
            var config = configs[index % configs.Length];
            yield return new object[] { exponent, config };
            index++;
        }
    }

    private static readonly ulong[] KnownMersennePrimeExponentValues =
    [
        31UL,
        61UL,
        89UL,
        107UL,
        127UL,
        521UL,
        607UL,
        1279UL,
        2203UL,
        2281UL,
        3217UL,
        4253UL,
        4423UL,
        9689UL,
        9941UL,
        11213UL,
    ];

    public readonly record struct MersenneGpuCandidateConfig(
        GpuKernelType KernelType,
        bool UseGpuOrder,
        ulong ResidueMaxK);

    [Fact]
    public void Composite_candidates_flagged_for_residue_output_skip()
    {
        ConfigureCliArguments("--mersenne=residue");

        var mersenneField = typeof(Program).GetField("MersenneTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var compositeField = typeof(Program).GetField("_lastCompositeP", BindingFlags.NonPublic | BindingFlags.Static)!;

        mersenneField.SetValue(null, new ThreadLocal<MersenneNumberTester>(() => new MersenneNumberTester(useIncremental: true, useResidue: false, maxK: 1_000UL), trackAllValues: true));
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        CandidatesCalculator.OverrideResidueTrackers(new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, 2UL, true), trackAllValues: true));

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
        try
        {
            compositeField.SetValue(null, false);

            Program.IsEvenPerfectCandidate(gpu, 9UL, 0UL, out bool searched, out bool detailed).Should().BeFalse();
            searched.Should().BeFalse();
            detailed.Should().BeFalse();
            ((bool)compositeField.GetValue(null)!).Should().BeTrue();

            Program.IsEvenPerfectCandidate(gpu, 11UL, 0UL, out searched, out detailed);
            searched.Should().BeTrue();
            ((bool)compositeField.GetValue(null)!).Should().BeFalse();
        }
        finally
        {
			PrimeOrderCalculatorAccelerator.Return(gpu);
            mersenneField.SetValue(null, null);
            primeField.SetValue(null, null);
            CandidatesCalculator.ResetResidueTrackers();
            compositeField.SetValue(null, false);
            ResetCliArguments();
        }
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void Main_displays_help_when_requested()
    {
        var main = typeof(Program).GetMethod("Main", BindingFlags.NonPublic | BindingFlags.Static)!;
        using var writer = new StringWriter();
        TextWriter original = Console.Out;
        Console.SetOut(writer);

        try
        {
            main.Invoke(null, [new[] { "--help" }]);
        }
        finally
        {
            Console.SetOut(original);
        }

        string output = writer.ToString();
        output.Should().Contain("Usage:");
        output.Should().Contain("--mersenne-device=cpu|gpu");
        output.Should().Contain("--primes-device=cpu|gpu");
    }
}

