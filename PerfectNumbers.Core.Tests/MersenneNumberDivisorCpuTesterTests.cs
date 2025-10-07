using System;
using FluentAssertions;
using PerfectNumbers.Core.Cpu;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class MersenneNumberDivisorCpuTesterTests
{
        [Fact]
        [Trait("Category", "Fast")]
        public void ByDivisor_tester_tracks_divisors_across_primes()
        {
                var tester = new MersenneNumberDivisorByDivisorCpuTester();
                tester.ConfigureFromMaxPrime(11UL);

                tester.IsPrime(5UL, out bool divisorsExhausted).Should().BeTrue();
                divisorsExhausted.Should().BeTrue();

                tester.IsPrime(7UL, out divisorsExhausted).Should().BeTrue();
                divisorsExhausted.Should().BeTrue();

                tester.IsPrime(11UL, out divisorsExhausted).Should().BeFalse();
                divisorsExhausted.Should().BeTrue();
        }

        [Fact]
        [Trait("Category", "Fast")]
        public void ByDivisor_session_checks_divisors_across_primes()
        {
                var tester = new MersenneNumberDivisorByDivisorCpuTester();
                tester.ConfigureFromMaxPrime(13UL);

                using var session = tester.CreateDivisorSession();
                ulong[] primes = { 5UL, 7UL, 11UL, 13UL };
                byte[] hits = new byte[primes.Length];

                ulong cycle23 = MersenneDivisorCycles.CalculateCycleLength(23UL, MontgomeryDivisorData.FromModulus(23UL));
                session.CheckDivisor(23UL, MontgomeryDivisorData.FromModulus(23UL), cycle23, primes, hits);
                hits.Should().ContainInOrder(new byte[] { 0, 0, 1, 0 });

                Array.Fill(hits, (byte)0);
                ulong cycle31 = MersenneDivisorCycles.CalculateCycleLength(31UL, MontgomeryDivisorData.FromModulus(31UL));
                session.CheckDivisor(31UL, MontgomeryDivisorData.FromModulus(31UL), cycle31, primes, hits);
                hits.Should().ContainInOrder(new byte[] { 1, 0, 0, 0 });
        }
}

