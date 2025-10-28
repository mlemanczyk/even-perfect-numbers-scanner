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

                var session = tester.CreateDivisorSession();
                try
                {
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
                finally
                {
                        session.Dispose();
                }
        }

        [Fact]
        [Trait("Category", "Fast")]
        public void ByDivisor_session_marks_mersenne_numbers_divisible_by_seven_as_composite()
        {
                var tester = new MersenneNumberDivisorByDivisorCpuTester();
                tester.ConfigureFromMaxPrime(43UL);

                var session = tester.CreateDivisorSession();
                try
                {
                        ulong[] exponents = { 6UL, 7UL, 9UL, 10UL, 12UL };
                        byte[] hits = new byte[exponents.Length];

                        MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(7UL);
                        ulong cycle = MersenneDivisorCycles.CalculateCycleLength(7UL, divisorData);
                        session.CheckDivisor(7UL, divisorData, cycle, exponents, hits);

                        hits.Should().Equal(new byte[] { 1, 0, 1, 0, 1 });
                }
                finally
                {
                        session.Dispose();
                }
        }

        [Fact]
        [Trait("Category", "Fast")]
        public void ByDivisor_session_marks_mersenne_numbers_divisible_by_eleven_as_composite()
        {
                var tester = new MersenneNumberDivisorByDivisorCpuTester();
                tester.ConfigureFromMaxPrime(61UL);

                var session = tester.CreateDivisorSession();
                try
                {
                        ulong[] exponents = { 10UL, 11UL, 20UL, 21UL, 30UL };
                        byte[] hits = new byte[exponents.Length];

                        MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(11UL);
                        ulong cycle = MersenneDivisorCycles.CalculateCycleLength(11UL, divisorData);
                        session.CheckDivisor(11UL, divisorData, cycle, exponents, hits);

                        hits.Should().Equal(new byte[] { 1, 0, 1, 0, 1 });
                }
                finally
                {
                        session.Dispose();
                }
        }
}

