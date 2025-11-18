using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
public class Mod10_8_5_3Benchmarks
{
    public ulong Mod10R;
    public ulong Mod8R;
    public ulong Mod3R;
    public ulong Mod5R;

    private static readonly ulong[] ValueCases =
    [
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
        ulong.MaxValue - 17UL,
    ];

    public static IEnumerable<ulong> GetValues() => ValueCases;

    /// <summary>
    /// Baseline scalar modulo path; previously 1.13-1.14 ns across the shared dataset and remains the reference point here.
    /// </summary>
    [Benchmark(Baseline = true)]
    [ArgumentsSource(nameof(GetValues))]
    public (ulong Mod10, ulong Mod8, ulong Mod3, ulong Mod5) ModuloOperator(ulong value)
    {
        return (value % 10UL, value % 8UL, value % 3UL, value % 5UL);
    }

    /// <summary>
    /// Legacy folding helper; 2.83-3.07 ns in the original suite, now exercised on the combined dataset.
    /// </summary>
    [Benchmark]
    [ArgumentsSource(nameof(GetValues))]
    public (ulong Mod10, ulong Mod8, ulong Mod3, ulong Mod5) LegacyModulo(ulong value)
    {
        ulong mod10;
        ulong mod8;
        ulong mod3;
        ulong mod5;
        UInt128 zero = UInt128.Zero;
        if (value == 0UL)
        {
            mod3 = 0UL;
            mod5 = 0UL;
            mod8 = 0UL;
            mod10 = 0UL;
            return (mod10, mod8, mod3, mod5);
        }

        UInt128 high = value >> 64;
        ulong result = value;
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
    /// Method-based optimized helper; 1.63-2.16 ns in the original benchmarks, roughly 1.4-1.9× slower than the raw modulo.
    /// </summary>
    [Benchmark]
    [ArgumentsSource(nameof(GetValues))]
    public (ulong Mod10, ulong Mod8, ulong Mod3, ulong Mod5) ModMethodModulo(ulong value)
    {
        value.Mod10_8_5_3(out ulong mod10, out ulong mod8, out ulong mod5, out ulong mod3);
        return (mod10, mod8, mod3, mod5);
    }

    /// <summary>
    /// Legacy property-based helper; 2.35-2.46 ns at small values and up to 2.46 ns near <c>ulong.Max</c>.
    /// </summary>
    [Benchmark]
    [ArgumentsSource(nameof(GetValues))]
    public (ulong Mod10, ulong Mod8, ulong Mod5, ulong Mod3) LegacyPropertyModulo(ulong value)
    {
        UInt128 wideValue = value;
        Mod10R = wideValue.Mod10();
        Mod8R = Mod8(wideValue);
        Mod5R = Mod5(wideValue);
        Mod3R = Mod3(wideValue);
        return (Mod10R, Mod8R, Mod5R, Mod3R);
    }

    /// <summary>
    /// Method variant for the property helper; 2.79-4.34 ns in the original run, about 1.2-1.8× slower than the legacy property path.
    /// </summary>
    [Benchmark]
    [ArgumentsSource(nameof(GetValues))]
    public (ulong Mod10, ulong Mod8, ulong Mod3, ulong Mod5) ModMethodPropertyModulo(ulong value)
    {
        value.Mod10_8_5_3(out Mod10R, out Mod8R, out Mod5R, out Mod3R);
        return (Mod10R, Mod8R, Mod3R, Mod5R);
    }

    /// <summary>
    /// Legacy step-per-bit helper; 11.9-13.4 ns in the original suite and remains the slowest option.
    /// </summary>
    [Benchmark]
    [ArgumentsSource(nameof(GetValues))]
    public (ulong Mod10, ulong Mod8, ulong Mod3, ulong Mod5) LegacyStepsModulo(ulong value)
    {
        UInt128 wideValue = value;
        ulong mod10 = (((ulong)(wideValue % 10UL)) << 1) % 10UL;
        ulong mod8 = (((ulong)(wideValue & 7UL)) << 1) & 7UL;
        ulong mod5 = (((ulong)(wideValue % 5UL)) << 1) % 5UL;
        ulong mod3 = (((ulong)(wideValue % 3UL)) << 1) % 3UL;
        return (mod10, mod8, mod3, mod5);
    }

    /// <summary>
    /// Step helper exposed via <see cref="UInt128Extensions.Mod10_8_5_3Steps"/>; 3.58-3.77 ns previously, delivering 3.2-3.6× speedups over the legacy step path.
    /// </summary>
    [Benchmark]
    [ArgumentsSource(nameof(GetValues))]
    public (ulong Mod10, ulong Mod8, ulong Mod3, ulong Mod5) ModMethodStepsModulo(ulong value)
    {
        value.Mod10_8_5_3Steps(out ulong mod10, out ulong mod8, out ulong mod5, out ulong mod3);
        return (mod10, mod8, mod3, mod5);
    }

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
}
