using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
public class Mod10_8_5_3PropertiesBenchmarks
{
    public ulong Mod10R;
    public ulong Mod8R;
    public ulong Mod3R;
    public ulong Mod5R;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Mod8(UInt128 value) => (ulong)value & 7UL;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Mod3(UInt128 value)
    {
        ulong high = (ulong)(value >> 64);
        ulong low = (ulong)value;
        ulong rem = (low % 3UL) + (high % 3UL);
        return rem >= 3UL ? rem - 3UL : rem;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Mod5(UInt128 value)
    {
        ulong high = (ulong)(value >> 64);
        ulong low = (ulong)value;
        ulong rem = (low % 5UL) + (high % 5UL);
        return rem >= 5UL ? rem - 5UL : rem;
    }

    [Params(
        3UL,
        6UL,
        128UL,
        256UL,
        2047UL,
        8191UL,
        65535UL,
        131071UL,
        2147483647UL,
        ulong.MaxValue - 1024UL,
        ulong.MaxValue - 511UL,
        ulong.MaxValue - 255UL,
        ulong.MaxValue - 127UL,
        ulong.MaxValue - 63UL,
        ulong.MaxValue - 31UL,
        ulong.MaxValue - 17UL)]
    public ulong Value { get; set; }

    /// <summary>
    /// Legacy property-based helper; stayed within 2.35-2.46 ns on the small inputs and 2.36-2.46 ns across the near-ulong.Max
    /// cases (e.g. 2.413 ns at value 3, 2.378 ns at 6, 2.458 ns at 2147483647, 2.371 ns near ulong.Max), keeping it in front.
    /// </summary>
    [Benchmark(Baseline = true)]
    public (ulong Mod10, ulong Mod8, ulong Mod5, ulong Mod3) LegacyModulo()
    {
        UInt128 value = Value;
        Mod10R = value.Mod10();
        Mod8R = Mod8(value);
        Mod5R = Mod5(value);
        Mod3R = Mod3(value);
        return (Mod10R, Mod8R, Mod5R, Mod3R);
    }

    /// <summary>
    /// Method variant that routes through <see cref="UInt128Extensions.Mod10_8_5_3"/>; ranged from 2.79 ns at value 3 to 4.34 ns on
    /// the largest operands (3.135 ns at 6, 3.033 ns at 2047, 3.441 ns at 2147483647, 4.289 ns near ulong.Max), roughly 1.16-1.83x
    /// slower than the legacy path.
    /// </summary>
    [Benchmark]
    public (ulong Mod10, ulong Mod8, ulong Mod3, ulong Mod5) ModMethodModulo()
    {
        Value.Mod10_8_5_3(out Mod10R, out Mod8R, out Mod5R, out Mod3R);
        return (Mod10R, Mod8R, Mod3R, Mod5R);
    }
}
