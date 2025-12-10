using FluentAssertions;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class HeuristicCombinedPrimeTesterTests
{
    [Theory]
    [Trait("Category", "Fast")]
    [InlineData((byte)1)]
    [InlineData((byte)7)]
    [InlineData((byte)9)]
    public void Enumerator_uses_one_a_one_b_schedule_by_default(byte lastDigit)
    {
        var enumerator = HeuristicCombinedPrimeTester.CreateHeuristicDivisorEnumerator(500UL * 500UL, lastDigit);
        var results = new List<HeuristicCombinedPrimeTester.HeuristicDivisorCandidate>(32);

        while (results.Count < 20 && enumerator.TryGetNext(out var candidate))
        {
            results.Add(candidate);
        }

        const int segmentsToCheck = 6;
        results.Count.Should().BeGreaterThanOrEqualTo(4 + segmentsToCheck * 2);
        results[0].Value.Should().Be(3UL);
        results[0].Group.Should().Be(HeuristicCombinedPrimeTester.HeuristicDivisorGroup.GroupAConstant);
        results[1].Value.Should().Be(7UL);
        results[1].Group.Should().Be(HeuristicCombinedPrimeTester.HeuristicDivisorGroup.GroupAConstant);
        results[2].Value.Should().Be(11UL);
        results[2].Group.Should().Be(HeuristicCombinedPrimeTester.HeuristicDivisorGroup.GroupAConstant);
        results[3].Value.Should().Be(13UL);
        results[3].Group.Should().Be(HeuristicCombinedPrimeTester.HeuristicDivisorGroup.GroupAConstant);

        int index = 4;

        for (int segment = 0; segment < segmentsToCheck; segment++)
        {
            HeuristicCombinedPrimeTester.HeuristicDivisorCandidate groupACandidate = results[index++];
            groupACandidate.Value.Should().BeGreaterThan(13UL);
            (groupACandidate.Value % 10UL).Should().Be(3UL);
            groupACandidate.Group.Should().NotBe(HeuristicCombinedPrimeTester.HeuristicDivisorGroup.GroupB);

            HeuristicCombinedPrimeTester.HeuristicDivisorCandidate groupBCandidate = results[index++];
            groupBCandidate.Value.Should().BeGreaterThan(13UL);
            (groupBCandidate.Value % 10UL).Should().NotBe(3UL);
            groupBCandidate.Group.Should().Be(HeuristicCombinedPrimeTester.HeuristicDivisorGroup.GroupB);
        }

    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData((byte)1)]
    [InlineData((byte)7)]
    [InlineData((byte)9)]
    public void One_a_one_b_pattern_interleaves_group_a_and_group_b_values(byte lastDigit)
    {
        ReadOnlySpan<ulong> combined = HeuristicCombinedPrimeTester.GetCombinedDivisors(
            lastDigit,
            HeuristicCombinedPrimeTester.CombinedDivisorPattern.OneAOneB);

        combined.Length.Should().BeGreaterThan(18);
        combined[0].Should().Be(3UL);
        combined[1].Should().Be(7UL);
        combined[2].Should().Be(11UL);
        combined[3].Should().Be(13UL);

        int index = 4;
        const int segmentsToCheck = 6;

        for (int segment = 0; segment < segmentsToCheck; segment++)
        {
            ulong groupAValue = combined[index++];
            groupAValue.Should().BeGreaterThan(13UL);
            (groupAValue % 10UL).Should().Be(3UL);

            ulong groupBValue = combined[index++];
            groupBValue.Should().BeGreaterThan(13UL);
            (groupBValue % 10UL).Should().NotBe(3UL);
        }
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData((byte)1)]
    [InlineData((byte)7)]
    [InlineData((byte)9)]
    public void Two_a_one_b_pattern_interleaves_two_group_a_values_before_group_b(byte lastDigit)
    {
        ReadOnlySpan<ulong> combined = HeuristicCombinedPrimeTester.GetCombinedDivisors(
            lastDigit,
            HeuristicCombinedPrimeTester.CombinedDivisorPattern.TwoAOneB);

        combined.Length.Should().BeGreaterThan(20);
        combined[0].Should().Be(3UL);
        combined[1].Should().Be(7UL);
        combined[2].Should().Be(11UL);
        combined[3].Should().Be(13UL);

        int index = 4;
        const int segmentsToCheck = 5;

        for (int segment = 0; segment < segmentsToCheck; segment++)
        {
            for (int i = 0; i < 2; i++)
            {
                ulong groupAValue = combined[index++];
                groupAValue.Should().BeGreaterThan(13UL);
                (groupAValue % 10UL).Should().Be(3UL);
            }

            ulong groupBValue = combined[index++];
            groupBValue.Should().BeGreaterThan(13UL);
            (groupBValue % 10UL).Should().NotBe(3UL);
        }
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData((byte)1)]
    [InlineData((byte)7)]
    [InlineData((byte)9)]
    public void Three_a_two_b_pattern_interleaves_three_group_a_values_before_two_group_b_values(byte lastDigit)
    {
        ReadOnlySpan<ulong> combined = HeuristicCombinedPrimeTester.GetCombinedDivisors(
            lastDigit,
            HeuristicCombinedPrimeTester.CombinedDivisorPattern.ThreeATwoB);

        combined.Length.Should().BeGreaterThan(28);
        combined[0].Should().Be(3UL);
        combined[1].Should().Be(7UL);
        combined[2].Should().Be(11UL);
        combined[3].Should().Be(13UL);

        int index = 4;
        const int segmentsToCheck = 4;

        for (int segment = 0; segment < segmentsToCheck; segment++)
        {
            for (int i = 0; i < 3; i++)
            {
                ulong groupAValue = combined[index++];
                groupAValue.Should().BeGreaterThan(13UL);
                (groupAValue % 10UL).Should().Be(3UL);
            }

            for (int i = 0; i < 2; i++)
            {
                ulong groupBValue = combined[index++];
                groupBValue.Should().BeGreaterThan(13UL);
                (groupBValue % 10UL).Should().NotBe(3UL);
            }
        }
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData((byte)1)]
    [InlineData((byte)7)]
    [InlineData((byte)9)]
    public void Three_a_one_b_pattern_interleaves_three_group_a_values_before_group_b(byte lastDigit)
    {
        ReadOnlySpan<ulong> combined = HeuristicCombinedPrimeTester.GetCombinedDivisors(
            lastDigit,
            HeuristicCombinedPrimeTester.CombinedDivisorPattern.ThreeAOneB);

        combined.Length.Should().BeGreaterThan(24);
        combined[0].Should().Be(3UL);
        combined[1].Should().Be(7UL);
        combined[2].Should().Be(11UL);
        combined[3].Should().Be(13UL);

        int index = 4;
        const int segmentsToCheck = 4;

        for (int segment = 0; segment < segmentsToCheck; segment++)
        {
            for (int i = 0; i < 3; i++)
            {
                ulong groupAValue = combined[index++];
                groupAValue.Should().BeGreaterThan(13UL);
                (groupAValue % 10UL).Should().Be(3UL);
            }

            ulong groupBValue = combined[index++];
            groupBValue.Should().BeGreaterThan(13UL);
            (groupBValue % 10UL).Should().NotBe(3UL);
        }
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(0UL)]
    [InlineData(1UL)]
    [InlineData(2UL)]
    [InlineData(3UL)]
    [InlineData(4UL)]
    [InlineData(5UL)]
    [InlineData(8UL)]
    [InlineData(9UL)]
    [InlineData(15UL)]
    [InlineData(16UL)]
    [InlineData(17UL)]
    [InlineData(24UL)]
    [InlineData(25UL)]
    [InlineData(26UL)]
    [InlineData(255UL)]
    [InlineData(256UL)]
    [InlineData(257UL)]
    [InlineData(1_000UL)]
    [InlineData(1_048_575UL)]
    [InlineData(1_048_576UL)]
    [InlineData(1_048_577UL)]
    [InlineData(4_000_000UL)]
    [InlineData(4_000_001UL)]
    [InlineData(4_000_004UL)]
    [InlineData(65_535_999UL)]
    [InlineData(65_536_000UL)]
    [InlineData(65_536_001UL)]
    [InlineData(4_294_967_295UL)]
    [InlineData(4_294_967_296UL)]
    [InlineData(4_294_967_297UL)]
    [InlineData(18_446_744_073_709_551_615UL)]
    [InlineData(18_446_744_073_709_551_614UL)]
    [InlineData(18_446_744_073_709_551_613UL)]
    public void ComputeHeuristicDivisorSquareLimit_returns_input_value(ulong value)
    {
        ulong limit = HeuristicCombinedPrimeTester.ComputeHeuristicDivisorSquareLimit(value);

        limit.Should().Be(value);
    }
}
