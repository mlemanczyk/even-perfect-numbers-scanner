using System;
using System.Collections.Generic;
using PerfectNumbers.Core;
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
    public void Enumerator_uses_three_a_two_b_schedule_by_default(byte lastDigit)
    {
        Span<HeuristicCombinedPrimeTester.HeuristicGroupBSequenceState> groupBState =
            stackalloc HeuristicCombinedPrimeTester.HeuristicGroupBSequenceState[4];

        var enumerator = HeuristicCombinedPrimeTester.CreateHeuristicDivisorEnumerator(500, lastDigit, groupBState);
        var results = new List<HeuristicCombinedPrimeTester.HeuristicDivisorCandidate>(32);

        while (results.Count < 20 && enumerator.TryGetNext(out var candidate))
        {
            results.Add(candidate);
        }

        results.Count.Should().BeGreaterThanOrEqualTo(16);
        results[0].Value.Should().Be(3UL);
        results[0].Group.Should().Be(HeuristicCombinedPrimeTester.HeuristicDivisorGroup.GroupAConstant);
        results[1].Value.Should().Be(7UL);
        results[1].Group.Should().Be(HeuristicCombinedPrimeTester.HeuristicDivisorGroup.GroupAConstant);
        results[2].Value.Should().Be(11UL);
        results[2].Group.Should().Be(HeuristicCombinedPrimeTester.HeuristicDivisorGroup.GroupAConstant);
        results[3].Value.Should().Be(13UL);
        results[3].Group.Should().Be(HeuristicCombinedPrimeTester.HeuristicDivisorGroup.GroupAConstant);

        int index = 4;
        const int segmentsToCheck = 3;

        for (int segment = 0; segment < segmentsToCheck; segment++)
        {
            for (int i = 0; i < 3; i++)
            {
                HeuristicCombinedPrimeTester.HeuristicDivisorCandidate groupACandidate = results[index++];
                groupACandidate.Value.Should().BeGreaterThan(13UL);
                (groupACandidate.Value % 10UL).Should().Be(3UL);
                groupACandidate.Group.Should().NotBe(HeuristicCombinedPrimeTester.HeuristicDivisorGroup.GroupB);
            }

            for (int i = 0; i < 2; i++)
            {
                HeuristicCombinedPrimeTester.HeuristicDivisorCandidate groupBCandidate = results[index++];
                groupBCandidate.Value.Should().BeGreaterThan(13UL);
                (groupBCandidate.Value % 10UL).Should().NotBe(3UL);
                groupBCandidate.Group.Should().Be(HeuristicCombinedPrimeTester.HeuristicDivisorGroup.GroupB);
            }
        }
    }

    [Theory]
    [Trait("Category", "Fast")]
    [InlineData((byte)1)]
    [InlineData((byte)7)]
    [InlineData((byte)9)]
    public void Three_a_two_b_pattern_interleaves_three_group_a_values_before_two_group_b_values(byte lastDigit)
    {
        ReadOnlySpan<uint> combined = HeuristicCombinedPrimeTester.GetCombinedDivisors(
            lastDigit,
            HeuristicCombinedPrimeTester.CombinedDivisorPattern.ThreeATwoB);

        combined.Length.Should().BeGreaterThan(28);
        combined[0].Should().Be(3U);
        combined[1].Should().Be(7U);
        combined[2].Should().Be(11U);
        combined[3].Should().Be(13U);

        int index = 4;
        const int segmentsToCheck = 4;

        for (int segment = 0; segment < segmentsToCheck; segment++)
        {
            for (int i = 0; i < 3; i++)
            {
                uint groupAValue = combined[index++];
                groupAValue.Should().BeGreaterThan(13U);
                (groupAValue % 10U).Should().Be(3U);
            }

            for (int i = 0; i < 2; i++)
            {
                uint groupBValue = combined[index++];
                groupBValue.Should().BeGreaterThan(13U);
                (groupBValue % 10U).Should().NotBe(3U);
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
        ReadOnlySpan<uint> combined = HeuristicCombinedPrimeTester.GetCombinedDivisors(
            lastDigit,
            HeuristicCombinedPrimeTester.CombinedDivisorPattern.ThreeAOneB);

        combined.Length.Should().BeGreaterThan(24);
        combined[0].Should().Be(3U);
        combined[1].Should().Be(7U);
        combined[2].Should().Be(11U);
        combined[3].Should().Be(13U);

        int index = 4;
        const int segmentsToCheck = 4;

        for (int segment = 0; segment < segmentsToCheck; segment++)
        {
            for (int i = 0; i < 3; i++)
            {
                uint groupAValue = combined[index++];
                groupAValue.Should().BeGreaterThan(13U);
                (groupAValue % 10U).Should().Be(3U);
            }

            uint groupBValue = combined[index++];
            groupBValue.Should().BeGreaterThan(13U);
            (groupBValue % 10U).Should().NotBe(3U);
        }
    }
}
