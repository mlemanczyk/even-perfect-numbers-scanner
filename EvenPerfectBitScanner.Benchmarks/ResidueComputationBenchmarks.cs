using System.Numerics;
using BenchmarkDotNet.Attributes;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
public class ResidueComputationBenchmarks
{
    [Params(3UL, 8191UL, 131071UL, 2147483647UL)]
    public ulong Exponent { get; set; }

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

    [Benchmark]
    public (ulong Step10, ulong Step8, ulong Step3, ulong Step5) ModMethodModulo()
    {
        UInt128 exponent128 = Exponent;
        exponent128.Mod10_8_5_3(out ulong mod10, out ulong mod8, out ulong mod5, out ulong mod3);
        ulong step10 = (mod10 << 1) % 10UL;
        ulong step8 = (mod8 << 1) & 7UL;
        ulong step3 = (mod3 << 1) % 3UL;
        ulong step5 = (mod5 << 1) % 5UL;
        return (step10, step8, step3, step5);
    }
}

