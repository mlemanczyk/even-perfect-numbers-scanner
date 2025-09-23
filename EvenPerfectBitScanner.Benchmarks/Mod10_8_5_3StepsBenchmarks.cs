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
		ulong mod10 = (((ulong)(value % 10UL)) << 1) % 10UL;
		ulong mod8 = (((ulong)(value & 7UL)) << 1) & 7UL;
		ulong mod5 = (((ulong)(value % 5UL)) << 1) % 5UL;
		ulong mod3 = (((ulong)(value % 3UL)) << 1) % 3UL;
        return (mod10, mod8, mod3, mod5);
    }

	/// <summary>
	/// Fastest
	/// </summary>
	[Benchmark]
    public (ulong Mod10, ulong Mod8, ulong Mod3, ulong Mod5) ModMethodModulo()
    {
		Value.Mod10_8_5_3Steps(out ulong mod10, out ulong mod8, out ulong mod5, out ulong mod3);
		// Step residues for q += 2*p
        return (mod10, mod8, mod3, mod5);
    }
}

