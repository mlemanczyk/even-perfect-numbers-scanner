using System.Numerics;
using BenchmarkDotNet.Attributes;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
public class Mod10_8_5_3StepsBenchmarks
{
    [Params(3UL, 8191UL, 131071UL, 2147483647UL)]
    public ulong Value { get; set; }

    [Benchmark(Baseline = true)]
    public (ulong Mod10, ulong Mod8, ulong Mod3, ulong Mod5) LegacyModulo()
    {
		UInt128 value = Value;
		ulong mod10 = (ulong)(value % 10UL) % 10UL;
		ulong mod8 = (ulong)(value % 8UL);
		ulong mod3 = (ulong)(value % 3UL);
		ulong mod5 = (ulong)(value % 5UL);
        return (mod10, mod8, mod3, mod5);
    }

    [Benchmark]
    public (ulong Mod10, ulong Mod8, ulong Mod3, ulong Mod5) ModMethodModulo()
    {
		Value.Mod10_8_5_3(out ulong mod10, out ulong mod8, out ulong mod5, out ulong mod3);
		// Step residues for q += 2*p
                mod10 %= 10UL;
        return (mod10, mod8, mod3, mod5);
    }
}

