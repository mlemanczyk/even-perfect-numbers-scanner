using FluentAssertions;
using Xunit;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;
using System.Reflection;

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
    public void Divisor_mode_checks_all_candidates_when_none_specified()
    {
        var useDivisorField = typeof(Program).GetField("_useDivisor", BindingFlags.NonPublic | BindingFlags.Static)!;
        var divisorField = typeof(Program).GetField("_divisor", BindingFlags.NonPublic | BindingFlags.Static)!;
        var testerField = typeof(Program).GetField("_divisorTester", BindingFlags.NonPublic | BindingFlags.Static)!;
        var candidatesField = typeof(Program).GetField("_divisorCandidates", BindingFlags.NonPublic | BindingFlags.Static)!;
        var primeField = typeof(Program).GetField("PrimeTesters", BindingFlags.NonPublic | BindingFlags.Static)!;
        var residueField = typeof(Program).GetField("PResidue", BindingFlags.NonPublic | BindingFlags.Static)!;
        var forceCpuProp = typeof(GpuContextPool).GetProperty("ForceCpu");

        useDivisorField.SetValue(null, true);
        divisorField.SetValue(null, 0UL);
        testerField.SetValue(null, new MersenneNumberDivisorGpuTester());
        primeField.SetValue(null, new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true));
        residueField.SetValue(null, new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, 2UL, true), trackAllValues: true));
        candidatesField.SetValue(null, new (ulong, uint)[] { (7UL, 3U), (23UL, 11U) });
        forceCpuProp!.SetValue(null, true);

        try
        {
            Program.IsEvenPerfectCandidate(11UL).Should().BeTrue();
            Program.IsEvenPerfectCandidate(5UL).Should().BeFalse();
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
