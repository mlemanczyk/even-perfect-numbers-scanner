using System;
using System.Threading;
using EvenPerfectBitScanner;
using PerfectNumbers.Core;
using Xunit;

namespace EvenPerfectBitScanner.Tests;

public class ResidueCandidateEnumerationTests
{
    [Fact]
    public void ModResidueTracker_does_not_flag_small_primes_as_composite()
    {
        var tracker = new ModResidueTracker(ResidueModel.Identity, initialNumber: 13UL, initialized: true);
        var primes = PrimesGenerator.SmallPrimes;
        var primesPow2 = PrimesGenerator.SmallPrimesPow2;

        ulong p = 13UL;
        ulong remainder = p % 6UL;

        for (int i = 0; i < 2000; i++)
        {
            tracker.BeginMerge(p);
            bool composite = false;
            for (int j = 0; j < primes.Length; j++)
            {
                if (primesPow2[j] > p)
                {
                    break;
                }

                tracker.MergeOrAppend(p, primes[j], out bool divisible);
                if (divisible)
                {
                    composite = true;
                    break;
                }
            }

            bool isPrime = PrimeTester.IsPrimeInternal(p, CancellationToken.None);
            if (isPrime && composite)
            {
                throw new InvalidOperationException($"Prime {p} marked as composite.");
            }

            p = Program.TransformPAdd(p, ref remainder);
        }
    }

    [Fact]
    public void TransformPAdd_visits_both_prime_residue_classes()
    {
        ulong p = 13UL;
        ulong remainder = p % 6UL;
        bool sawOneModSix = false;
        bool sawFiveModSix = false;

        for (int i = 0; i < 256; i++)
        {
            ulong residue = p % 6UL;
            if (residue == 1UL)
            {
                sawOneModSix = true;
            }
            else if (residue == 5UL)
            {
                sawFiveModSix = true;
            }

            p = Program.TransformPAdd(p, ref remainder);
        }

        Assert.True(sawOneModSix, "Sequence never produced numbers congruent to 1 mod 6.");
        Assert.True(sawFiveModSix, "Sequence never produced numbers congruent to 5 mod 6.");
    }

}
