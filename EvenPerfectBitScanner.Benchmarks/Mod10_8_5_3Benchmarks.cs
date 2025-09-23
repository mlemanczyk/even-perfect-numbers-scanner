using BenchmarkDotNet.Attributes;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
public class Mod10_8_5_3Benchmarks
{
    [Params(3UL, 8191UL, 131071UL, 2147483647UL)]
    public ulong Value { get; set; }

	/// <summary>
	/// Fastest
	/// </summary>
	[Benchmark(Baseline = true)]
    public (ulong Mod10, ulong Mod8, ulong Mod3, ulong Mod5) ModuloOperator()
    {
        ulong value = Value;
        return (value % 10UL, value % 8UL, value % 3UL, value % 5UL);
    }

    [Benchmark]
    public (ulong Mod10, ulong Mod8, ulong Mod3, ulong Mod5) LegacyModulo()
    {
                ulong mod10, mod8, mod3, mod5;
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
                // 2^64 ≡ 1 (mod 3)
                ulong modRem = (result % 3UL) + ((ulong)high % 3UL);
                mod3 = modRem >= 3UL ? modRem - 3UL : modRem;
                // 2^64 ≡ 1 (mod 5)
                modRem = (result % 5UL) + ((ulong)high % 5UL);
                mod5 = modRem >= 5UL ? modRem - 5UL : modRem;

                while (high != zero)
                {
                        // 2^64 ≡ 6 (mod 10)
                        result = (result + (ulong)high * 6UL) % 10UL;
                        high >>= 64;
                }

                mod10 = result % 10UL;
        return (mod10, mod8, mod3, mod5);
    }

    [Benchmark]
    public (ulong Mod10, ulong Mod8, ulong Mod3, ulong Mod5) ModMethodModulo()
    {
        Value.Mod10_8_5_3(out ulong mod10, out ulong mod8, out ulong mod5, out ulong mod3);
        return (mod10, mod8, mod3, mod5);
    }
}

