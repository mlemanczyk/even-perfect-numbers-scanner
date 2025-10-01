using BenchmarkDotNet.Attributes;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
public class ResidueComputationBenchmarks
{
    [Params(3UL, 8191UL, 131071UL, 2147483647UL)]
    public ulong Exponent { get; set; }

    /// <summary>
    /// Legacy modulo pipeline; stays in the 2.41–2.43 ns range even for the 31-bit exponent (2,147,483,647).
    /// </summary>
    /// <remarks>
    /// Observed means: Exponent 3 → 2.412 ns (1.00×), 8,191 → 2.405 ns, 131,071 → 2.416 ns, 2,147,483,647 → 2.427 ns.
    /// </remarks>
    [Benchmark(Baseline = true)]
    public (ulong Step10, ulong Step8, ulong Step3, ulong Step5) LegacyModulo()
    {
        ulong exponent = Exponent;
        ulong step10 = ((exponent % 10UL) << 1) % 10UL;
        ulong step8 = ((exponent & 7UL) << 1) & 7UL;
        ulong step3 = ((exponent % 3UL) << 1) % 3UL;
        ulong step5 = ((exponent % 5UL) << 1) % 5UL;
        return (step10, step8, step3, step5);
    }

    /// <summary>
    /// <see cref="PerfectNumbers.Core.ExponentResidues.Mod10_8_5_3"/> helper; 2.80 ns at exponent 3 but stretching to 3.36 ns at the 31-bit case (1.16–1.38× slower than legacy).
    /// </summary>
    /// <remarks>
    /// Observed means: Exponent 3 → 2.799 ns (1.16×), 8,191 → 2.842 ns, 131,071 → 3.111 ns, 2,147,483,647 → 3.355 ns.
    /// </remarks>
    [Benchmark]
    public (ulong Step10, ulong Step8, ulong Step3, ulong Step5) ModMethodModulo()
    {
        Exponent.Mod10_8_5_3(out ulong mod10, out ulong mod8, out ulong mod5, out ulong mod3);
        mod10 = (mod10 << 1) % 10UL;
        mod8 = (mod8 << 1) & 7UL;
        mod3 = (mod3 << 1) % 3UL;
        mod5 = (mod5 << 1) % 5UL;
        return (mod10, mod8, mod3, mod5);
    }
}

