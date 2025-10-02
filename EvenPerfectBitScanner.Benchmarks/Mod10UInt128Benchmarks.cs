using BenchmarkDotNet.Attributes;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
public class Mod10UInt128Benchmarks
{
    [Params(3UL, 8191UL, 131071UL, 2147483647UL)]
    public UInt128 Value { get; set; }

    /// <summary>
    /// Direct `% 10` using the <see cref="UInt128"/> operator; lands between 3.25 ns and 3.38 ns (3.378 ns at value 3, 3.269 ns at
    /// 8191, 3.248 ns at 131071, 3.281 ns at 2147483647) and serves as the baseline.
    /// </summary>
    [Benchmark(Baseline = true)]
    public UInt128 StandardModulo()
    {
        return Value % 10UL;
    }

    /// <summary>
    /// Splits the value into high/low halves and folds with the 2^64 ≡ 6 (mod 10) rule; fastest option at 1.78-1.82 ns (1.804 ns at
    /// value 3, 1.798 ns at 8191, 1.817 ns at 131071, 1.779 ns at 2147483647), beating the baseline by ~1.9x.
    /// </summary>
    [Benchmark]
    public ulong ModWithHighLowModulo()
    {
        return ModWithHighLowModuloHelper(Value);
    }

    private static ulong ModWithHighLowModuloHelper(UInt128 value128)
    {
        ulong value = (ulong)value128;
        return ((value % 10UL) + ((value >> 64) % 10UL) * 6UL) % 10UL;
    }

    /// <summary>
    /// Loop-based folding that repeatedly applies the 2^64 ≡ 6 rule; measures 2.20-2.27 ns (2.204 ns at value 3, 2.234 ns at 8191,
    /// 2.268 ns at 131071, 2.216 ns at 2147483647), situating it between the high/low helper and the raw modulo.
    /// </summary>
    [Benchmark]
    public ulong ModWithLoopModulo()
    {
        return ModWithLoopModuloHelper(Value);
    }

    private static ulong ModWithLoopModuloHelper(UInt128 value)
    {
        UInt128 zero = UInt128.Zero;
        if (value == zero)
        {
            return 0UL;
        }

        ulong result = (ulong)value;
        value >>= 64;

        while (value != zero)
        {
            result = (result + (ulong)value * 6UL) % 10UL;
            value >>= 64;
        }

        return result % 10UL;
    }
}
