using System.Reflection;
using FluentAssertions;
using PerfectNumbers.Core.Cpu;
using PerfectNumbers.Core.Gpu;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class MersenneNumberDivisorGpuTesterTests
{
    [Theory]
    [InlineData(33UL, 7UL, true)]
    [InlineData(35UL, 31UL, true)]
    [InlineData(37UL, 223UL, true)]
    [InlineData(33UL, 13UL, false)]
    [InlineData(35UL, 73UL, false)]
    [InlineData(37UL, 227UL, false)]
    [Trait("Category", "Fast")]
    public void IsDivisible_returns_expected(ulong exponent, ulong divisor, bool expected)
    {
        var tester = new MersenneNumberDivisorGpuTester();
        ReadOnlyGpuUInt128 divisorValue = new ReadOnlyGpuUInt128(divisor);
        tester.IsDivisible(exponent, in divisorValue).Should().Be(expected);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void IsDivisible_handles_large_divisors()
    {
        var tester = new MersenneNumberDivisorGpuTester();
        UInt128 divisor = (UInt128.One << 65) - UInt128.One;
        ReadOnlyGpuUInt128 divisorValue = new ReadOnlyGpuUInt128(divisor);
        tester.IsDivisible(65UL, in divisorValue).Should().BeTrue();
        divisorValue = new ReadOnlyGpuUInt128(divisor + 2); // Reusing divisorValue for the shifted candidate.
        tester.IsDivisible(65UL, in divisorValue).Should().BeFalse();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void IsPrime_sets_divisorsExhausted_false_when_search_range_not_exhausted()
    {
        var tester = new MersenneNumberDivisorGpuTester();
        typeof(MersenneNumberDivisorGpuTester)
            .GetField("_divisorCandidates", BindingFlags.NonPublic | BindingFlags.Static)!
            .SetValue(null, Array.Empty<(ulong, uint)>());

        tester.IsPrime(31UL, UInt128.Zero, 0UL, out bool divisorsExhausted).Should().BeTrue();
        divisorsExhausted.Should().BeFalse();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void IsPrime_sets_divisorsExhausted_true_when_divisible_by_specified_divisor()
    {
        var tester = new MersenneNumberDivisorGpuTester();
        tester.IsPrime(35UL, 31UL, 0UL, out bool divisorsExhausted).Should().BeFalse();
        divisorsExhausted.Should().BeTrue();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void IsPrime_accepts_large_search_limits()
    {
        var tester = new MersenneNumberDivisorGpuTester();
        tester.IsPrime(35UL, 31UL, ulong.MaxValue, out bool exhausted).Should().BeFalse();
        exhausted.Should().BeTrue();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_tester_tracks_divisors_across_primes()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester();
        tester.ConfigureFromMaxPrime(43UL);

        tester.IsPrime(31UL, out bool exhausted).Should().BeTrue();
        exhausted.Should().BeTrue();

        tester.IsPrime(37UL, out exhausted).Should().BeFalse();
        exhausted.Should().BeTrue();

        tester.IsPrime(43UL, out exhausted).Should().BeFalse();
        exhausted.Should().BeTrue();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_tester_requires_configuration()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester();
        Action act = () => tester.IsPrime(31UL, out _);
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_tester_respects_filter_based_limit()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester();
        tester.ConfigureFromMaxPrime(41UL);

        tester.IsPrime(31UL, out _).Should().BeTrue();
        tester.IsPrime(37UL, out bool exhausted).Should().BeFalse();
        exhausted.Should().BeTrue();

        tester.IsPrime(41UL, out bool divisorsExhausted).Should().BeFalse();
        divisorsExhausted.Should().BeTrue();
    }


    [Theory]
    [InlineData(71UL, false)]
    [InlineData(89UL, true)]
    [InlineData(127UL, true)]
    [Trait("Category", "Fast")]
    public void ByDivisor_tester_matches_incremental_expectations(ulong exponent, bool expectedPrime)
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester();
        tester.ConfigureFromMaxPrime(exponent);

        tester.IsPrime(exponent, out bool exhausted).Should().Be(expectedPrime);
        exhausted.Should().BeTrue();
    }
    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_gpu_tester_uses_cycle_remainders_per_divisor()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester
        {
            GpuBatchSize = 8,
        };

        tester.ConfigureFromMaxPrime(43UL);

        tester.IsPrime(41UL, out bool divisorsExhausted).Should().BeFalse();
        divisorsExhausted.Should().BeTrue();
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_session_checks_divisors_across_primes()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester();
        tester.ConfigureFromMaxPrime(43UL);

        using var session = tester.CreateDivisorSession();
        ulong[] primes = { 31UL, 37UL, 41UL, 43UL };
        byte[] hits = new byte[primes.Length];

        ulong cycle223 = MersenneDivisorCycles.CalculateCycleLength(223UL, MontgomeryDivisorData.FromModulus(223UL));
        session.CheckDivisor(223UL, MontgomeryDivisorData.FromModulus(223UL), cycle223, primes, hits);
        hits.Should().ContainInOrder(new byte[] { 0, 1, 0, 0 });

        Array.Fill(hits, (byte)0);
        ulong cycle13367 = MersenneDivisorCycles.CalculateCycleLength(13367UL, MontgomeryDivisorData.FromModulus(13367UL));
        session.CheckDivisor(13367UL, MontgomeryDivisorData.FromModulus(13367UL), cycle13367, primes, hits);
        hits.Should().ContainInOrder(new byte[] { 0, 0, 1, 0 });
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_gpu_session_reuses_cycle_remainders_across_batches()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester
        {
            GpuBatchSize = 2,
        };

        tester.ConfigureFromMaxPrime(47UL);

        using var session = tester.CreateDivisorSession();
        ulong[] primes = { 31UL, 37UL, 41UL, 43UL, 47UL };
        byte[] hits = new byte[primes.Length];

        ulong cycle223 = MersenneDivisorCycles.CalculateCycleLength(223UL, MontgomeryDivisorData.FromModulus(223UL));
        session.CheckDivisor(223UL, MontgomeryDivisorData.FromModulus(223UL), cycle223, primes, hits);

        hits.Should().ContainInOrder(new byte[] { 0, 1, 0, 0, 0 });
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_session_marks_mersenne_numbers_divisible_by_seven_as_composite()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester
        {
            GpuBatchSize = 5,
        };
        tester.ConfigureFromMaxPrime(43UL);

        using var session = tester.CreateDivisorSession();
        ulong[] exponents = { 6UL, 7UL, 9UL, 10UL, 12UL };
        byte[] hits = new byte[exponents.Length];

        MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(7UL);
        ulong cycle = MersenneDivisorCycles.CalculateCycleLength(7UL, divisorData);

        session.CheckDivisor(7UL, divisorData, cycle, exponents, hits);

        var hostBufferField = typeof(MersenneNumberDivisorByDivisorGpuTester.DivisorScanSession)
            .GetField("_hostBuffer", BindingFlags.NonPublic | BindingFlags.Instance)!;
        ulong[] hostBuffer = (ulong[])hostBufferField.GetValue(session)!;
        Span<ulong> residues = hostBuffer.AsSpan(0, exponents.Length);
        byte[] computedHits = new byte[exponents.Length];
        for (int i = 0; i < residues.Length; i++)
        {
            computedHits[i] = residues[i] == divisorData.MontgomeryOne ? (byte)1 : (byte)0;
        }

        computedHits.Should().Equal(new byte[] { 1, 0, 1, 0, 1 });
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_session_marks_mersenne_numbers_divisible_by_eleven_as_composite()
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester
        {
            GpuBatchSize = 5,
        };
        tester.ConfigureFromMaxPrime(61UL);

        using var session = tester.CreateDivisorSession();
        ulong[] exponents = { 10UL, 11UL, 20UL, 21UL, 30UL };
        byte[] hits = new byte[exponents.Length];

        MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(11UL);
        ulong cycle = MersenneDivisorCycles.CalculateCycleLength(11UL, divisorData);

        session.CheckDivisor(11UL, divisorData, cycle, exponents, hits);

        var hostBufferField = typeof(MersenneNumberDivisorByDivisorGpuTester.DivisorScanSession)
            .GetField("_hostBuffer", BindingFlags.NonPublic | BindingFlags.Instance)!;
        ulong[] hostBuffer = (ulong[])hostBufferField.GetValue(session)!;
        Span<ulong> residues = hostBuffer.AsSpan(0, exponents.Length);
        byte[] computedHits = new byte[exponents.Length];
        for (int i = 0; i < residues.Length; i++)
        {
            computedHits[i] = residues[i] == divisorData.MontgomeryOne ? (byte)1 : (byte)0;
        }

        computedHits.Should().Equal(new byte[] { 1, 0, 1, 0, 1 });
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_gpu_tester_skips_divisors_excluded_by_small_cycle_generation()
    {
        var cycles = MersenneDivisorCycles.Shared;
        var tableField = typeof(MersenneDivisorCycles).GetField("_table", BindingFlags.NonPublic | BindingFlags.Instance)!;
        var smallCyclesField = typeof(MersenneDivisorCycles).GetField("_smallCycles", BindingFlags.NonPublic | BindingFlags.Instance)!;

        var originalTable = (List<(ulong Divisor, ulong Cycle)>)tableField.GetValue(cycles)!;
        var originalSmall = (ulong[]?)smallCyclesField.GetValue(cycles);

        try
        {
            var patchedTable = new List<(ulong Divisor, ulong Cycle)>();
            ulong[] patchedSmall = new ulong[PerfectNumberConstants.MaxQForDivisorCycles + 1];

            MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(191UL);
            patchedSmall[191] = MersenneDivisorCycles.CalculateCycleLength(191UL, divisorData);

            tableField.SetValue(cycles, patchedTable);
            smallCyclesField.SetValue(cycles, patchedSmall);
            DivisorCycleCache.Shared.RefreshSnapshot();

            var tester = new MersenneNumberDivisorByDivisorGpuTester();
            tester.ConfigureFromMaxPrime(19UL);

            typeof(MersenneNumberDivisorByDivisorGpuTester)
                .GetField("_divisorLimit", BindingFlags.NonPublic | BindingFlags.Instance)!
                .SetValue(tester, 200UL);

            tester.IsPrime(19UL, out bool divisorsExhausted).Should().BeTrue();
            divisorsExhausted.Should().BeTrue();
        }
        finally
        {
            tableField.SetValue(cycles, originalTable);
            smallCyclesField.SetValue(cycles, originalSmall);
            DivisorCycleCache.Shared.RefreshSnapshot();
            GpuContextPool.DisposeAll();
        }
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void ByDivisor_cpu_session_reuses_cycle_remainders_across_primes()
    {
        var tester = new MersenneNumberDivisorByDivisorCpuTester
        {
            BatchSize = 2,
        };

        tester.ConfigureFromMaxPrime(47UL);

        using var session = tester.CreateDivisorSession();
        ulong[] primes = { 31UL, 37UL, 41UL, 43UL, 47UL };
        byte[] hits = new byte[primes.Length];

        ulong cycle223 = MersenneDivisorCycles.CalculateCycleLength(223UL, MontgomeryDivisorData.FromModulus(223UL));
        session.CheckDivisor(223UL, MontgomeryDivisorData.FromModulus(223UL), cycle223, primes, hits);

        hits.Should().ContainInOrder(new byte[] { 0, 1, 0, 0, 0 });
    }
}

