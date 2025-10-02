using BenchmarkDotNet.Attributes;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
public class Mod10_8_5_3StepsBenchmarks
{
    [Params(3UL, 8191UL, 131071UL, 2147483647UL)]
    public ulong Value { get; set; }

    /// <summary>
    /// Legacy step-per-bit helper; required 11.89-13.40 ns per call (12.052 ns at value 3, 11.891 ns at 8191, 13.401 ns at 131071,
    /// 12.114 ns at 2147483647), making it 3-4x slower than the streamlined method variant.
    /// </summary>
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
    /// Optimized step helper exposed on <see cref="UInt128Extensions.Mod10_8_5_3Steps"/>; trimmed execution to 3.58-3.77 ns
    /// (3.576 ns at value 3, 3.682 ns at 8191, 3.773 ns at 131071, 3.649 ns at 2147483647), giving a 3.2-3.6x speedup.
    /// </summary>
    [Benchmark]
    public (ulong Mod10, ulong Mod8, ulong Mod3, ulong Mod5) ModMethodModulo()
    {
        Value.Mod10_8_5_3Steps(out ulong mod10, out ulong mod8, out ulong mod5, out ulong mod3);
        return (mod10, mod8, mod3, mod5);
    }
}
