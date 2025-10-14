using System;
using FluentAssertions;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public static class MersenneNumberDivisorCandidateCpuEvaluatorTests
{
    [Fact]
    public static void EvaluateCandidatesProducesExpectedMask()
    {
        ulong prime = 138_000_001UL;
        ulong step = prime << 1;
        ulong startDivisor = step + 1UL;
        ulong limit = startDivisor + step * 8UL;
        bool lastIsSeven = (prime & 3UL) == 3UL;

        Span<byte> remainders = stackalloc byte[6];
        Span<byte> steps = stackalloc byte[6];
        Span<ulong> candidates = stackalloc ulong[8];
        Span<byte> mask = stackalloc byte[8];

        remainders[0] = (byte)(startDivisor % 10UL);
        remainders[1] = (byte)(startDivisor % 8UL);
        remainders[2] = (byte)(startDivisor % 5UL);
        remainders[3] = (byte)(startDivisor % 3UL);
        remainders[4] = (byte)(startDivisor % 7UL);
        remainders[5] = (byte)(startDivisor % 11UL);

        steps[0] = (byte)(step % 10UL);
        steps[1] = (byte)(step % 8UL);
        steps[2] = (byte)(step % 5UL);
        steps[3] = (byte)(step % 3UL);
        steps[4] = (byte)(step % 7UL);
        steps[5] = (byte)(step % 11UL);

        MersenneNumberDivisorCandidateCpuEvaluator.EvaluateCandidates(
            startDivisor,
            step,
            limit,
            remainders,
            steps,
            lastIsSeven,
            candidates,
            mask);

        for (int i = 0; i < candidates.Length; i++)
        {
            ulong expectedCandidate = startDivisor + step * (ulong)i;
            candidates[i].Should().Be(expectedCandidate);

            byte remainder10 = (byte)(expectedCandidate % 10UL);
            byte remainder8 = (byte)(expectedCandidate % 8UL);
            byte remainder5 = (byte)(expectedCandidate % 5UL);
            byte remainder3 = (byte)(expectedCandidate % 3UL);
            byte remainder7 = (byte)(expectedCandidate % 7UL);
            byte remainder11 = (byte)(expectedCandidate % 11UL);

            bool accept10 = lastIsSeven
                ? (remainder10 == 3 || remainder10 == 7 || remainder10 == 9)
                : (remainder10 == 1 || remainder10 == 3 || remainder10 == 9);

            bool expectedMask = expectedCandidate <= limit
                && accept10
                && (remainder8 == 1 || remainder8 == 7)
                && remainder3 != 0
                && remainder5 != 0
                && remainder7 != 0
                && remainder11 != 0;

            mask[i].Should().Be(expectedMask ? (byte)1 : (byte)0);
        }

        Span<byte> expectedRemainders = stackalloc byte[6];
        expectedRemainders[0] = (byte)(startDivisor % 10UL);
        expectedRemainders[1] = (byte)(startDivisor % 8UL);
        expectedRemainders[2] = (byte)(startDivisor % 5UL);
        expectedRemainders[3] = (byte)(startDivisor % 3UL);
        expectedRemainders[4] = (byte)(startDivisor % 7UL);
        expectedRemainders[5] = (byte)(startDivisor % 11UL);

        for (int i = 0; i < candidates.Length; i++)
        {
            expectedRemainders[0] = AdvanceRemainder(expectedRemainders[0], steps[0], 10);
            expectedRemainders[1] = AdvanceRemainder(expectedRemainders[1], steps[1], 8);
            expectedRemainders[2] = AdvanceRemainder(expectedRemainders[2], steps[2], 5);
            expectedRemainders[3] = AdvanceRemainder(expectedRemainders[3], steps[3], 3);
            expectedRemainders[4] = AdvanceRemainder(expectedRemainders[4], steps[4], 7);
            expectedRemainders[5] = AdvanceRemainder(expectedRemainders[5], steps[5], 11);
        }

        remainders.SequenceEqual(expectedRemainders).Should().BeTrue();
    }

    private static byte AdvanceRemainder(byte value, byte step, byte modulus)
    {
        int next = value + step;
        if (next >= modulus)
        {
            next -= modulus;
        }

        return (byte)next;
    }
}
