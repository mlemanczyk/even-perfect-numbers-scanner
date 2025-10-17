using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod11ULongBenchmarks
{
    [Params(11UL, 8191UL, 131071UL, 2147483647UL, ulong.MaxValue - 127UL)]
    public ulong Value { get; set; }

    /// <summary>
    /// `% 11` baseline for `ulong`; timings hovered near 0.26-0.27 ns on the large inputs (0.2647 ns at 2147483647, 0.2709 ns near
    /// ulong.Max) with the smaller samples in the same 0.23-0.27 ns band.
    /// </summary>
    [Benchmark(Baseline = true)]
    public ulong ModuloOperator()
    {
        return Value % 11UL;
    }

    /// <summary>
    /// Lookup-based helper mirroring `UIntExtensions.Mod11`; costs 0.66-0.75 ns on 32-bit scale inputs and rises to 1.15 ns near
    /// ulong.Max, remaining 2.5-4.3x slower than the `%` baseline.
    /// </summary>
    [Benchmark]
    public ulong ExtensionMethod()
    {
        return Mod11(Value);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Mod11(ulong value)
    {
        ulong remainder = ((uint)value % 11U) + ((uint)(value >> 32) % 11U) << 2;
        while (remainder >= 11UL)
        {
            remainder -= 11UL;
        }

        return remainder;
    }
}
