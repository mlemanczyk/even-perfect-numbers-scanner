using FluentAssertions;
using Xunit;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Reflection;
using System.Text;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner.Tests;

[Trait("Category", "Fast")]
public class ProgramTests
{
    [Fact]
    public void CountOnes_returns_correct_count()
    {
        Program.CountOnes(0b101010UL).Should().Be(3UL);
    }

    [Theory]
    [InlineData(2UL, true)]
    [InlineData(4UL, false)]
    [InlineData(9UL, false)]
    [InlineData(11UL, true)]
    [InlineData(81UL, false)]
    public void IsPrime_identifies_primes_correctly(ulong n, bool expected)
    {
        new PrimeTester().IsPrime(n, CancellationToken.None).Should().Be(expected);
    }

    [Fact]
    public void TransformPAdd_moves_to_next_candidate()
    {
        ulong remainder = 3UL;
        Program.TransformPAdd(3UL, ref remainder).Should().Be(5UL);
        Program.TransformPAdd(5UL, ref remainder).Should().Be(7UL);
        Program.TransformPAdd(7UL, ref remainder).Should().Be(11UL);
    }

    [Theory]
    [InlineData(false, false, false, 5UL, 1024UL, 5UL)]
    [InlineData(true, false, false, 5UL, 1024UL, 5UL)]
    [InlineData(true, true, true, 7UL, 1024UL, 7UL)]
    [InlineData(true, false, true, 5UL, 1024UL, 1024UL)]
    [InlineData(true, true, false, 5UL, 1024UL, 5UL)]
    public void ResolveResidueMaxK_applies_expected_rules(
        bool useResidue,
        bool residueMaxExplicit,
        bool divisorLimitExplicit,
        ulong residueMaxK,
        ulong divisorLimit,
        ulong expected)
    {
        Program.ResolveResidueMaxK(
                useResidue,
                residueMaxExplicit,
                divisorLimitExplicit,
                residueMaxK,
                divisorLimit).Should().Be((UInt128)expected);
    }

    [Fact]
    public void TransformPBit_appends_one_bit_and_skips_to_candidate()
    {
        ulong remainder = 5UL;
        Program.TransformPBit(5UL, ref remainder).Should().Be(13UL);
    }

    [Fact]
    public void TransformPBit_detects_overflow_and_stops()
    {
        typeof(Program).GetField("_limitReached", BindingFlags.NonPublic | BindingFlags.Static)!
            .SetValue(null, false);

        ulong start = (ulong.MaxValue >> 1) + 1UL;
        ulong remainder = start % 6UL;
        Program.TransformPBit(start, ref remainder);

        bool limit = (bool)typeof(Program).GetField("_limitReached", BindingFlags.NonPublic | BindingFlags.Static)!
            .GetValue(null)!;
        limit.Should().BeTrue();

        typeof(Program).GetField("_limitReached", BindingFlags.NonPublic | BindingFlags.Static)!
            .SetValue(null, false);
    }

    [Fact]
    public void Gcd_filter_detects_some_composites()
    {
        Program.IsCompositeByGcd(15UL).Should().BeTrue();
        Program.IsCompositeByGcd(5UL).Should().BeFalse();
    }

    [Fact]
    public void Divisor_mode_scans_dynamic_cycles_when_none_specified()
    {
        var useDivisorField = typeof(Program).GetField("_useDivisor", BindingFlags.NonPublic | BindingFlags.Static)!;
        var divisorField = typeof(Program).GetField("_divisor", BindingFlags.NonPublic | BindingFlags.Static)!;
        var testerField = typeof(Program).GetField("_divisorTester", BindingFlags.NonPublic | BindingFlags.Static)!;
        var candidatesField = typeof(MersenneNumberDivisorGpuTester).GetField("_divisorCandidates", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var residueField = typeof(Program).GetField("PResidue", BindingFlags.NonPublic | BindingFlags.Static)!;
        var forceCpuProp = typeof(GpuContextPool).GetProperty("ForceCpu");

        useDivisorField.SetValue(null, true);
        divisorField.SetValue(null, UInt128.Zero);
        testerField.SetValue(null, new MersenneNumberDivisorGpuTester());
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        residueField.SetValue(null, new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, 2UL, true), trackAllValues: true));
        candidatesField.SetValue(null, new (ulong, uint)[] { (7UL, 3U), (23UL, 11U) });
        forceCpuProp!.SetValue(null, true);

        try
        {
            Program.IsEvenPerfectCandidate(11UL, 32UL).Should().BeFalse();
            Program.IsEvenPerfectCandidate(5UL, 32UL).Should().BeTrue();
            Program.IsEvenPerfectCandidate(127UL, 32UL).Should().BeTrue();
        }
        finally
        {
            useDivisorField.SetValue(null, false);
            testerField.SetValue(null, null);
            candidatesField.SetValue(null, null);
            primeField.SetValue(null, null);
            residueField.SetValue(null, null);
            forceCpuProp!.SetValue(null, false);
        }
    }

    [Fact]
    public void Divisor_mode_sets_detailedCheck_based_on_exhaustion()
    {
        var useDivisorField = typeof(Program).GetField("_useDivisor", BindingFlags.NonPublic | BindingFlags.Static)!;
        var divisorField = typeof(Program).GetField("_divisor", BindingFlags.NonPublic | BindingFlags.Static)!;
        var testerField = typeof(Program).GetField("_divisorTester", BindingFlags.NonPublic | BindingFlags.Static)!;
        var candidatesField = typeof(MersenneNumberDivisorGpuTester).GetField("_divisorCandidates", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var residueField = typeof(Program).GetField("PResidue", BindingFlags.NonPublic | BindingFlags.Static)!;
        var forceCpuProp = typeof(GpuContextPool).GetProperty("ForceCpu");

        useDivisorField.SetValue(null, true);
        divisorField.SetValue(null, UInt128.Zero);
        testerField.SetValue(null, new MersenneNumberDivisorGpuTester());
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        residueField.SetValue(null, new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, 2UL, true), trackAllValues: true));
        candidatesField.SetValue(null, Array.Empty<(ulong, uint)>());
        forceCpuProp!.SetValue(null, true);

        try
        {
            Program.IsEvenPerfectCandidate(11UL, 0UL, out bool searched, out bool detailedCheck).Should().BeTrue();
            searched.Should().BeTrue();
            detailedCheck.Should().BeFalse();
        }
        finally
        {
            useDivisorField.SetValue(null, false);
            testerField.SetValue(null, null);
            candidatesField.SetValue(null, null);
            primeField.SetValue(null, null);
            residueField.SetValue(null, null);
            forceCpuProp!.SetValue(null, false);
        }
    }

    [Fact]
    public void Divisor_mode_sets_detailedCheck_when_divisor_is_found()
    {
        var useDivisorField = typeof(Program).GetField("_useDivisor", BindingFlags.NonPublic | BindingFlags.Static)!;
        var divisorField = typeof(Program).GetField("_divisor", BindingFlags.NonPublic | BindingFlags.Static)!;
        var testerField = typeof(Program).GetField("_divisorTester", BindingFlags.NonPublic | BindingFlags.Static)!;
        var candidatesField = typeof(MersenneNumberDivisorGpuTester).GetField("_divisorCandidates", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var residueField = typeof(Program).GetField("PResidue", BindingFlags.NonPublic | BindingFlags.Static)!;
        var forceCpuProp = typeof(GpuContextPool).GetProperty("ForceCpu");

        useDivisorField.SetValue(null, true);
        divisorField.SetValue(null, UInt128.Zero);
        testerField.SetValue(null, new MersenneNumberDivisorGpuTester());
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        residueField.SetValue(null, new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, 2UL, true), trackAllValues: true));
        candidatesField.SetValue(null, new (ulong, uint)[] { (23UL, 11U) });
        forceCpuProp!.SetValue(null, true);

        try
        {
            Program.IsEvenPerfectCandidate(11UL, 32UL, out bool searched, out bool detailedCheck).Should().BeFalse();
            searched.Should().BeTrue();
            detailedCheck.Should().BeTrue();
        }
        finally
        {
            useDivisorField.SetValue(null, false);
            testerField.SetValue(null, null);
            candidatesField.SetValue(null, null);
            primeField.SetValue(null, null);
            residueField.SetValue(null, null);
            forceCpuProp!.SetValue(null, false);
        }
    }

    [Fact]
    public void Residue_mode_sets_detailedCheck_based_on_exhaustion()
    {
        var useDivisorField = typeof(Program).GetField("_useDivisor", BindingFlags.NonPublic | BindingFlags.Static)!;
        var residueModeField = typeof(Program).GetField("_useResidueMode", BindingFlags.NonPublic | BindingFlags.Static)!;
        var mersenneField = typeof(Program).GetField("MersenneTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var residueField = typeof(Program).GetField("PResidue", BindingFlags.NonPublic | BindingFlags.Static)!;
        var forceCpuProp = typeof(GpuContextPool).GetProperty("ForceCpu");

        const ulong exponent = 7UL;

        useDivisorField.SetValue(null, false);
        residueModeField.SetValue(null, true);
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        residueField.SetValue(null, new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, exponent, true), trackAllValues: true));
        forceCpuProp!.SetValue(null, true);

        try
        {
            mersenneField.SetValue(null, new ThreadLocal<MersenneNumberTester>(() => new MersenneNumberTester(
                    useGpuScan: false,
                    useGpuOrder: false,
                    useResidue: true,
                    maxK: 0UL,
                    residueDivisorSets: 0UL), trackAllValues: true));

            Program.IsEvenPerfectCandidate(exponent, 0UL, out bool searched, out bool detailedCheck).Should().BeTrue();
            searched.Should().BeTrue();
            detailedCheck.Should().BeFalse();

            mersenneField.SetValue(null, new ThreadLocal<MersenneNumberTester>(() => new MersenneNumberTester(
                    useGpuScan: false,
                    useGpuOrder: false,
                    useResidue: true,
                    maxK: 4UL,
                    residueDivisorSets: 2UL), trackAllValues: true));

            Program.IsEvenPerfectCandidate(exponent, 0UL, out searched, out detailedCheck).Should().BeTrue();
            searched.Should().BeTrue();
            detailedCheck.Should().BeTrue();
        }
        finally
        {
            mersenneField.SetValue(null, null);
            primeField.SetValue(null, null);
            residueField.SetValue(null, null);
            residueModeField.SetValue(null, false);
            forceCpuProp!.SetValue(null, false);
            useDivisorField.SetValue(null, false);
        }
    }

    [Fact]
    public void Divisor_mode_accepts_large_search_limit()
    {
        var useDivisorField = typeof(Program).GetField("_useDivisor", BindingFlags.NonPublic | BindingFlags.Static)!;
        var divisorField = typeof(Program).GetField("_divisor", BindingFlags.NonPublic | BindingFlags.Static)!;
        var testerField = typeof(Program).GetField("_divisorTester", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var residueField = typeof(Program).GetField("PResidue", BindingFlags.NonPublic | BindingFlags.Static)!;
        var forceCpuProp = typeof(GpuContextPool).GetProperty("ForceCpu");

        useDivisorField.SetValue(null, true);
        divisorField.SetValue(null, (UInt128)7);
        testerField.SetValue(null, new MersenneNumberDivisorGpuTester());
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        residueField.SetValue(null, new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, 2UL, true), trackAllValues: true));
        forceCpuProp!.SetValue(null, true);

        try
        {
            Program.IsEvenPerfectCandidate(3UL, 4_000_000_000UL, out bool searched, out bool detailed).Should().BeFalse();
            searched.Should().BeTrue();
            detailed.Should().BeTrue();
        }
        finally
        {
            useDivisorField.SetValue(null, false);
            testerField.SetValue(null, null);
            primeField.SetValue(null, null);
            residueField.SetValue(null, null);
            forceCpuProp!.SetValue(null, false);
        }
    }

    [Fact]
    public void Divisor_mode_checks_divisibility_on_gpu()
    {
        var useDivisorField = typeof(Program).GetField("_useDivisor", BindingFlags.NonPublic | BindingFlags.Static)!;
        var divisorField = typeof(Program).GetField("_divisor", BindingFlags.NonPublic | BindingFlags.Static)!;
        var testerField = typeof(Program).GetField("_divisorTester", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var residueField = typeof(Program).GetField("PResidue", BindingFlags.NonPublic | BindingFlags.Static)!;
        var forceCpuProp = typeof(GpuContextPool).GetProperty("ForceCpu");

        useDivisorField.SetValue(null, true);
        divisorField.SetValue(null, (UInt128)23);
        testerField.SetValue(null, new MersenneNumberDivisorGpuTester());
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        residueField.SetValue(null, new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, 2UL, true), trackAllValues: true));
        forceCpuProp!.SetValue(null, false);

        try
        {
            Program.IsEvenPerfectCandidate(11UL, 0UL, out bool searched, out bool detailed).Should().BeFalse();
            searched.Should().BeTrue();
            detailed.Should().BeTrue();

            Program.IsEvenPerfectCandidate(13UL, 0UL, out searched, out detailed).Should().BeTrue();
            searched.Should().BeTrue();
            detailed.Should().BeFalse();
        }
        finally
        {
            useDivisorField.SetValue(null, false);
            divisorField.SetValue(null, UInt128.Zero);
            testerField.SetValue(null, null);
            primeField.SetValue(null, null);
            residueField.SetValue(null, null);
            forceCpuProp!.SetValue(null, false);
        }
    }

    [Fact]
    public void Residue_mode_checks_mersenne_primes_on_gpu()
    {
        var useDivisorField = typeof(Program).GetField("_useDivisor", BindingFlags.NonPublic | BindingFlags.Static)!;
        var mersenneField = typeof(Program).GetField("MersenneTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var residueField = typeof(Program).GetField("PResidue", BindingFlags.NonPublic | BindingFlags.Static)!;
        var forceCpuProp = typeof(GpuContextPool).GetProperty("ForceCpu");

        useDivisorField.SetValue(null, false);
        mersenneField.SetValue(null, new ThreadLocal<MersenneNumberTester>(() => new MersenneNumberTester(useIncremental: true, useResidue: true, maxK: 1_000UL, residueDivisorSets: 1UL), trackAllValues: true));
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        residueField.SetValue(null, new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, 2UL, true), trackAllValues: true));
        forceCpuProp!.SetValue(null, false);

        try
        {
            Program.IsEvenPerfectCandidate(29UL, 0UL, out bool searched, out bool detailed).Should().BeFalse();
            searched.Should().BeTrue();
            detailed.Should().BeTrue();

            Program.IsEvenPerfectCandidate(31UL, 0UL, out searched, out detailed).Should().BeTrue();
            searched.Should().BeTrue();
            detailed.Should().BeTrue();
        }
        finally
        {
            useDivisorField.SetValue(null, false);
            mersenneField.SetValue(null, null);
            primeField.SetValue(null, null);
            residueField.SetValue(null, null);
            forceCpuProp!.SetValue(null, false);
        }
    }

    [Fact]
    public void Residue_mode_with_large_divisor_cycle_limits_accepts_known_primes()
    {
        var useDivisorField = typeof(Program).GetField("_useDivisor", BindingFlags.NonPublic | BindingFlags.Static)!;
        var mersenneField = typeof(Program).GetField("MersenneTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var residueField = typeof(Program).GetField("PResidue", BindingFlags.NonPublic | BindingFlags.Static)!;
        var forceCpuProp = typeof(GpuContextPool).GetProperty("ForceCpu");

        useDivisorField.SetValue(null, false);
        mersenneField.SetValue(null, new ThreadLocal<MersenneNumberTester>(() => new MersenneNumberTester(
            useIncremental: true,
            useResidue: true,
            maxK: 1_024UL,
            residueDivisorSets: 1_024UL), trackAllValues: true));
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        residueField.SetValue(null, new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, 2UL, true), trackAllValues: true));
        forceCpuProp!.SetValue(null, false);

        try
        {
            ulong[] primes = [107UL, 127UL];
            foreach (ulong prime in primes)
            {
                Program.IsEvenPerfectCandidate(prime, 1024UL, out bool searched, out bool detailed).Should().BeTrue();
                searched.Should().BeTrue();
                detailed.Should().BeTrue();
            }
        }
        finally
        {
            useDivisorField.SetValue(null, false);
            mersenneField.SetValue(null, null);
            primeField.SetValue(null, null);
            residueField.SetValue(null, null);
            forceCpuProp!.SetValue(null, false);
        }
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void Residue_mode_accepts_known_primes_in_parallel()
    {
        var useDivisorField = typeof(Program).GetField("_useDivisor", BindingFlags.NonPublic | BindingFlags.Static)!;
        var mersenneField = typeof(Program).GetField("MersenneTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var residueField = typeof(Program).GetField("PResidue", BindingFlags.NonPublic | BindingFlags.Static)!;
        var gcdFilterField = typeof(Program).GetField("_useGcdFilter", BindingFlags.NonPublic | BindingFlags.Static)!;
        var forceCpuProp = typeof(GpuContextPool).GetProperty("ForceCpu");

        useDivisorField.SetValue(null, false);
        gcdFilterField.SetValue(null, false);
        mersenneField.SetValue(null, new ThreadLocal<MersenneNumberTester>(() => new MersenneNumberTester(
            useIncremental: true,
            useResidue: true,
            maxK: 5_000_000UL,
            residueDivisorSets: PerfectNumberConstants.ExtraDivisorCycleSearchLimit), trackAllValues: true));
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        residueField.SetValue(null, new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, 2UL, true), trackAllValues: true));
        forceCpuProp!.SetValue(null, false);

        try
        {
            ulong[] primes = [107UL, 127UL, 521UL, 607UL];
            Parallel.ForEach(primes, prime =>
            {
                Program.IsEvenPerfectCandidate(prime, 0UL, out bool searched, out bool detailed).Should().BeTrue();
                searched.Should().BeTrue();
                detailed.Should().BeTrue();
            });
        }
        finally
        {
            useDivisorField.SetValue(null, false);
            mersenneField.SetValue(null, null);
            primeField.SetValue(null, null);
            residueField.SetValue(null, null);
            forceCpuProp!.SetValue(null, false);
        }
    }

    [Fact]
    public void Residue_mode_with_default_limits_accepts_large_known_primes()
    {
        var useDivisorField = typeof(Program).GetField("_useDivisor", BindingFlags.NonPublic | BindingFlags.Static)!;
        var mersenneField = typeof(Program).GetField("MersenneTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var residueField = typeof(Program).GetField("PResidue", BindingFlags.NonPublic | BindingFlags.Static)!;
        var forceCpuProp = typeof(GpuContextPool).GetProperty("ForceCpu");

        useDivisorField.SetValue(null, false);
        mersenneField.SetValue(null, new ThreadLocal<MersenneNumberTester>(() => new MersenneNumberTester(
            useIncremental: true,
            useResidue: true,
            maxK: 5_000_000UL,
            residueDivisorSets: PerfectNumberConstants.ExtraDivisorCycleSearchLimit), trackAllValues: true));
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        residueField.SetValue(null, new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, 2UL, true), trackAllValues: true));
        forceCpuProp!.SetValue(null, false);

        try
        {
            Program.IsEvenPerfectCandidate(107UL, 0UL, out bool searched, out bool detailed).Should().BeTrue();
            searched.Should().BeTrue();
            detailed.Should().BeTrue();

            Program.IsEvenPerfectCandidate(127UL, 0UL, out searched, out detailed).Should().BeTrue();
            searched.Should().BeTrue();
            detailed.Should().BeTrue();
        }
        finally
        {
            useDivisorField.SetValue(null, false);
            mersenneField.SetValue(null, null);
            primeField.SetValue(null, null);
            residueField.SetValue(null, null);
            forceCpuProp!.SetValue(null, false);
        }
    }

    [Theory]
    [MemberData(nameof(KnownMersennePrimeExponents))]
    public void IsEvenPerfectCandidate_accepts_known_small_mersenne_primes(ulong exponent)
    {
        var useDivisorField = typeof(Program).GetField("_useDivisor", BindingFlags.NonPublic | BindingFlags.Static)!;
        var mersenneField = typeof(Program).GetField("MersenneTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var residueField = typeof(Program).GetField("PResidue", BindingFlags.NonPublic | BindingFlags.Static)!;
        var gcdFilterField = typeof(Program).GetField("_useGcdFilter", BindingFlags.NonPublic | BindingFlags.Static)!;
        var forceCpuProp = typeof(GpuContextPool).GetProperty("ForceCpu");

        useDivisorField.SetValue(null, false);
        gcdFilterField.SetValue(null, false);
        mersenneField.SetValue(null, new ThreadLocal<MersenneNumberTester>(() => new MersenneNumberTester(
            useIncremental: true,
            useGpuScan: false,
            useGpuOrder: false,
            useResidue: true,
            maxK: 1_024UL,
            residueDivisorSets: 1UL), trackAllValues: true));
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        residueField.SetValue(null, new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, exponent, true), trackAllValues: true));
        forceCpuProp!.SetValue(null, true);

        try
        {
            Program.IsEvenPerfectCandidate(exponent, 0UL, out bool searched, out bool detailedCheck).Should().BeTrue();
            searched.Should().BeTrue();
        }
        finally
        {
            useDivisorField.SetValue(null, false);
            mersenneField.SetValue(null, null);
            primeField.SetValue(null, null);
            residueField.SetValue(null, null);
            gcdFilterField.SetValue(null, false);
            forceCpuProp!.SetValue(null, false);
        }
    }

    [Theory]
    [MemberData(nameof(KnownMersennePrimeGpuConfigurations))]
    public void IsEvenPerfectCandidate_accepts_known_small_mersenne_primes_on_gpu(
        ulong exponent,
        MersenneGpuCandidateConfig configuration)
    {
        var useDivisorField = typeof(Program).GetField("_useDivisor", BindingFlags.NonPublic | BindingFlags.Static)!;
        var mersenneField = typeof(Program).GetField("MersenneTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var residueField = typeof(Program).GetField("PResidue", BindingFlags.NonPublic | BindingFlags.Static)!;
        var gcdFilterField = typeof(Program).GetField("_useGcdFilter", BindingFlags.NonPublic | BindingFlags.Static)!;
        var forceCpuProp = typeof(GpuContextPool).GetProperty("ForceCpu");

        useDivisorField.SetValue(null, false);
        gcdFilterField.SetValue(null, false);
        mersenneField.SetValue(null, new ThreadLocal<MersenneNumberTester>(() => new MersenneNumberTester(
            useIncremental: false,
            kernelType: configuration.KernelType,
            useGpuScan: true,
            useGpuOrder: configuration.UseGpuOrder,
            useResidue: true,
            maxK: (UInt128)configuration.ResidueMaxK,
            residueDivisorSets: (UInt128)configuration.ResidueSetCount), trackAllValues: true));
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        residueField.SetValue(null, new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, exponent, true), trackAllValues: true));
        forceCpuProp!.SetValue(null, false);

        try
        {
            Program.IsEvenPerfectCandidate(exponent, 0UL, out bool searched, out bool detailedCheck).Should().BeTrue();
            searched.Should().BeTrue();
        }
        finally
        {
            useDivisorField.SetValue(null, false);
            mersenneField.SetValue(null, null);
            primeField.SetValue(null, null);
            residueField.SetValue(null, null);
            gcdFilterField.SetValue(null, false);
            forceCpuProp!.SetValue(null, false);
            GpuContextPool.DisposeAll();
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
            new(GpuKernelType.Incremental, true, 1_024UL, 1UL),
            new(GpuKernelType.Incremental, false, 2_048UL, 1UL),
            new(GpuKernelType.Pow2Mod, true, 4_096UL, 2UL),
            new(GpuKernelType.Pow2Mod, false, 512UL, 2UL),
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
        ulong ResidueMaxK,
        ulong ResidueSetCount);

    [Fact]
    public void Composite_candidates_flagged_for_residue_output_skip()
    {
        var useDivisorField = typeof(Program).GetField("_useDivisor", BindingFlags.NonPublic | BindingFlags.Static)!;
        var mersenneField = typeof(Program).GetField("MersenneTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var residueField = typeof(Program).GetField("PResidue", BindingFlags.NonPublic | BindingFlags.Static)!;
        var compositeField = typeof(Program).GetField("_lastCompositeP", BindingFlags.NonPublic | BindingFlags.Static)!;
        var forceCpuProp = typeof(GpuContextPool).GetProperty("ForceCpu");

        useDivisorField.SetValue(null, false);
        mersenneField.SetValue(null, new ThreadLocal<MersenneNumberTester>(() => new MersenneNumberTester(useIncremental: true, useResidue: false, maxK: 1_000UL), trackAllValues: true));
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        residueField.SetValue(null, new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, 2UL, true), trackAllValues: true));
        forceCpuProp!.SetValue(null, true);

        try
        {
            compositeField.SetValue(null, false);

            Program.IsEvenPerfectCandidate(9UL, 0UL, out bool searched, out bool detailed).Should().BeFalse();
            searched.Should().BeFalse();
            detailed.Should().BeFalse();
            ((bool)compositeField.GetValue(null)!).Should().BeTrue();

            Program.IsEvenPerfectCandidate(11UL, 0UL, out searched, out detailed);
            searched.Should().BeTrue();
            ((bool)compositeField.GetValue(null)!).Should().BeFalse();
        }
        finally
        {
            mersenneField.SetValue(null, null);
            primeField.SetValue(null, null);
            residueField.SetValue(null, null);
            compositeField.SetValue(null, false);
            forceCpuProp!.SetValue(null, false);
        }
    }

    [Fact]
    public void Residue_mode_skips_composite_results_in_output()
    {
        var residueModeField = typeof(Program).GetField("_useResidueMode", BindingFlags.NonPublic | BindingFlags.Static)!;
        var outputField = typeof(Program).GetField("_outputBuilder", BindingFlags.NonPublic | BindingFlags.Static)!;
        var writeIndexField = typeof(Program).GetField("_writeIndex", BindingFlags.NonPublic | BindingFlags.Static)!;
        var compositeField = typeof(Program).GetField("_lastCompositeP", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeCountField = typeof(Program).GetField("_primeCount", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeFoundField = typeof(Program).GetField("_primeFoundAfterInit", BindingFlags.NonPublic | BindingFlags.Static)!;
        var consoleCounterField = typeof(Program).GetField("_consoleCounter", BindingFlags.NonPublic | BindingFlags.Static)!;
        var printResult = typeof(Program).GetMethod("PrintResult", BindingFlags.NonPublic | BindingFlags.Static)!;

        var previousResidueMode = (bool)residueModeField.GetValue(null)!;
        var previousOutput = (StringBuilder?)outputField.GetValue(null);
        var previousWriteIndex = (int)writeIndexField.GetValue(null)!;
        var previousComposite = (bool)compositeField.GetValue(null)!;
        var previousPrimeCount = (int)primeCountField.GetValue(null)!;
        var previousPrimeFound = (bool)primeFoundField.GetValue(null)!;
        var previousConsoleCounter = (int)consoleCounterField.GetValue(null)!;

        var builder = StringBuilderPool.Rent();

        try
        {
            builder.Clear();
            residueModeField.SetValue(null, true);
            compositeField.SetValue(null, true);
            outputField.SetValue(null, builder);
            writeIndexField.SetValue(null, 0);
            primeCountField.SetValue(null, 0);
            primeFoundField.SetValue(null, false);
            consoleCounterField.SetValue(null, 0);

            printResult.Invoke(null, new object[] { 9UL, false, false, false });

            builder.ToString().Should().BeEmpty();
            ((int)writeIndexField.GetValue(null)!).Should().Be(0);

            printResult.Invoke(null, new object[] { 11UL, true, false, true });

            builder.ToString().Should().Be("11,True,False,True" + Environment.NewLine);
            ((int)writeIndexField.GetValue(null)!).Should().Be(1);
        }
        finally
        {
            outputField.SetValue(null, previousOutput);
            writeIndexField.SetValue(null, previousWriteIndex);
            residueModeField.SetValue(null, previousResidueMode);
            compositeField.SetValue(null, previousComposite);
            primeCountField.SetValue(null, previousPrimeCount);
            primeFoundField.SetValue(null, previousPrimeFound);
            consoleCounterField.SetValue(null, previousConsoleCounter);
            builder.Clear();
            StringBuilderPool.Return(builder);
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

