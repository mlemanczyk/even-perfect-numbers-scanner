using BenchmarkDotNet.Attributes;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
public class Mod10_8_5_3Benchmarks
{
    [Params(3UL, 8191UL, 131071UL, 2147483647UL)]
    public ulong Value { get; set; }

    /// <summary>
    /// Baseline scalar modulo path; held steady between 1.13 ns and 1.14 ns across every value (1.138 ns at 3, 1.143 ns at 8191,
    /// 1.134 ns at 131071, 1.141 ns at 2147483647), making it the fastest option overall.
    /// </summary>
    [Benchmark(Baseline = true)]
    public (ulong Mod10, ulong Mod8, ulong Mod3, ulong Mod5) ModuloOperator()
    {
        ulong value = Value;
        return (value % 10UL, value % 8UL, value % 3UL, value % 5UL);
    }

    /// <summary>
    /// Legacy folding helper; useful for cross-checks but stays at 2.83-3.07 ns (2.834 ns at 3, 3.038 ns at 8191, 3.067 ns at
    /// 131071, 3.039 ns at 2147483647), roughly 2.5-2.7x slower than the baseline.
    /// </summary>
    [Benchmark]
    public (ulong Mod10, ulong Mod8, ulong Mod3, ulong Mod5) LegacyModulo()
    {
        ulong mod10;
        ulong mod8;
        ulong mod3;
        ulong mod5;
        UInt128 zero = UInt128.Zero;
        if (Value == zero)
        {
            mod3 = 0UL;
            mod5 = 0UL;
            mod8 = 0UL;
            mod10 = 0UL;
            return (mod10, mod8, mod3, mod5);
        }

        UInt128 high = Value >> 64;
        ulong result = Value;
        mod8 = result & 7UL;

        ulong modRem = (result % 3UL) + ((ulong)high % 3UL);
        mod3 = modRem >= 3UL ? modRem - 3UL : modRem;

        modRem = (result % 5UL) + ((ulong)high % 5UL);
        mod5 = modRem >= 5UL ? modRem - 5UL : modRem;

        while (high != zero)
        {
            result = (result + (ulong)high * 6UL) % 10UL;
            high >>= 64;
        }

        mod10 = result % 10UL;
        return (mod10, mod8, mod3, mod5);
    }

    /// <summary>
    /// Method-based optimized helper; clocks in at 1.63 ns for value 3, 1.64 ns at 8191, 1.875 ns at 131071, and 2.164 ns at
    /// 2147483647 (1.43-1.90x the baseline) so it trails the raw `%` but remains 1.6x quicker than the legacy fold.
    /// </summary>
    [Benchmark]
    public (ulong Mod10, ulong Mod8, ulong Mod3, ulong Mod5) ModMethodModulo()
    {
        Value.Mod10_8_5_3(out ulong mod10, out ulong mod8, out ulong mod5, out ulong mod3);
        return (mod10, mod8, mod3, mod5);
    }
}
