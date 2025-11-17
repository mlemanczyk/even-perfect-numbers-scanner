using FluentAssertions;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class MersenneNumberDivisorByDivisorTesterTests
{
    [Fact]
    [Trait("Category", "Fast")]
    public void Run_marks_candidates_divisible_by_seven_or_eleven_as_composite()
    {
        var candidates = new List<ulong> { 5UL, 7UL, 11UL, 13UL };
        var tester = new FakeTester();
        var results = new List<(ulong Prime, bool Searched, bool Detailed, bool Passed)>();
        int compositeMarks = 0;
        int clearedMarks = 0;

        MersenneNumberDivisorByDivisorTester.Run(
            candidates,
            tester,
            previousResults: null,
            startPrime: 0UL,
            markComposite: () => compositeMarks++,
            clearComposite: () => clearedMarks++,
            printResult: (prime, searched, detailed, passed) => results.Add((prime, searched, detailed, passed)),
            threadCount: 1,
            primesPerTask: 2);

        tester.ConfiguredMaxPrime.Should().Be(13UL);
        compositeMarks.Should().Be(2);
        clearedMarks.Should().Be(2);
        results.Should().Equal(
            new[]
            {
                (5UL, true, true, true),
                (7UL, true, true, false),
                (11UL, true, true, false),
                (13UL, true, true, true),
            });
    }

    private sealed class FakeTester : IMersenneNumberDivisorByDivisorTester
    {
        public ulong ConfiguredMaxPrime { get; private set; }

        public int BatchSize { get; set; }

        public ulong DivisorLimit => ConfiguredMaxPrime + 100UL;

        public void ConfigureFromMaxPrime(ulong maxPrime)
        {
            ConfiguredMaxPrime = maxPrime;
        }

        public ulong GetAllowedMaxDivisor(ulong prime)
        {
            return DivisorLimit;
        }

        public bool IsPrime(ulong prime, out bool divisorsExhausted)
        {
            divisorsExhausted = true;
            return prime % 7UL != 0UL && prime % 11UL != 0UL;
        }

        public IMersenneNumberDivisorByDivisorTester.IDivisorScanSession CreateDivisorSession()
        {
            return new DummySession();
        }

        public void PrepareCandidates(in ReadOnlySpan<ulong> primes, Span<ulong> allowedMaxValues)
        {
            for (int i = 0; i < primes.Length; i++)
            {
                allowedMaxValues[i] = 100UL;
            }
        }

        private sealed class DummySession : IMersenneNumberDivisorByDivisorTester.IDivisorScanSession
        {
            public void CheckDivisor(ulong divisor, in MontgomeryDivisorData divisorData, ulong divisorCycle, in ReadOnlySpan<ulong> primes, Span<byte> hits)
            {
                throw new NotSupportedException();
            }

            public void Return()
            {
            }
        }
    }
}
