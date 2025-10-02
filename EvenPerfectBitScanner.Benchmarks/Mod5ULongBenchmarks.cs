using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod5ULongBenchmarks
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Mod5(ulong value) =>
        ((uint)(value & ULongExtensions.WordBitMask)
         + (uint)((value >> 16) & ULongExtensions.WordBitMask)
         + (uint)((value >> 32) & ULongExtensions.WordBitMask)
         + (uint)((value >> 48) & ULongExtensions.WordBitMask)) % 5UL;

    [Params(5UL, 8191UL, 131071UL, 2147483647UL, ulong.MaxValue - 31UL)]
    public ulong Value { get; set; }

    /// <summary>
    /// `% 5` baseline; measured 0.2657 ns at value 5, 0.2622 ns at 8191, 0.2640 ns at 131071, 0.2645 ns at 2147483647, and 0.2696 ns
    /// near ulong.Max.
    /// </summary>
    [Benchmark(Baseline = true)]
    public ulong ModuloOperator()
    {
        return Value % 5UL;
    }

    /// <summary>
    /// Extension helper; slower at 0.431-0.448 ns (0.4305 ns at value 5, 0.4404 ns at 8191, 0.4359 ns at 131071, 0.4361 ns at
    /// 2147483647, 0.4481 ns near ulong.Max), roughly 1.6-1.7x the `%` cost.
    /// </summary>
    [Benchmark]
    public ulong ExtensionMethod()
    {
        return Mod5(Value);
    }
}
