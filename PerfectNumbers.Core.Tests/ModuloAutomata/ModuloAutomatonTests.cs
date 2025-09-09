using FluentAssertions;
using Xunit;

namespace PerfectNumbers.Core.Tests;

[Trait("Category", "Fast")]
public class ModuloAutomatonTests
{
    [Fact]
    public void Ending7Automaton_mod9_hits_every_9th()
    {
        // Start below 7; sequence: 7, 17, 27, 37, ...
        var auto = new Ending7Automaton(1, 9);
        int hits = 0;
        ulong firstHit = 0;
        for (int i = 0; i < 40; i++)
        {
            ulong n = auto.Current();
            bool div9 = auto.DivisibleBy(9);
            if (div9)
            {
                hits++;
                if (firstHit == 0)
                {
                    firstHit = n;
                }
            }

            auto.Next();
        }

        // Known CRT class: n ≡ 27 (mod 90) when n ends with 7 and n ≡ 0 (mod 9)
        firstHit.Should().Be(27);
        hits.Should().BeGreaterThan(1);
    }

    [Theory]
    [InlineData(13UL, 2000)]
    [InlineData(17UL, 2000)]
    public void MersenneResidueAutomaton_matches_direct_mods(ulong p, int count)
    {
        // Compare residue progression against direct modular computations
        var auto = new MersenneResidueAutomaton(p);
        var twoP = (System.UInt128)p << 1;
        var q = twoP + 1UL;

        for (int i = 0; i < count; i++)
        {
            ulong r10 = Mod10(q);
            ulong r8 = (ulong)q & 7UL;
            ulong r3 = Mod3(q);
            ulong r5 = Mod5(q);

            auto.Mod10R.Should().Be(r10);
            auto.Mod8R.Should().Be(r8);
            auto.Mod3R.Should().Be(r3);
            auto.Mod5R.Should().Be(r5);

            auto.Next();
            q += twoP;
        }
    }

    private static ulong Mod10(System.UInt128 value)
    {
        ulong high = (ulong)(value >> 64);
        ulong low = (ulong)value;
        ulong highRem = Mod10(high);
        ulong lowRem = Mod10(low);
        ulong combined = lowRem + highRem * 6UL;
        return Mod10(combined);
    }

    private static ulong Mod10(ulong value) => value % 10UL;

    private static ulong Mod3(System.UInt128 value)
    {
        ulong high = (ulong)(value >> 64);
        ulong low = (ulong)value;
        ulong rem = (low % 3UL) + (high % 3UL);
        return rem >= 3UL ? rem - 3UL : rem;
    }

    private static ulong Mod5(System.UInt128 value)
    {
        ulong high = (ulong)(value >> 64);
        ulong low = (ulong)value;
        ulong rem = (low % 5UL) + (high % 5UL);
        return rem >= 5UL ? rem - 5UL : rem;
    }
}
