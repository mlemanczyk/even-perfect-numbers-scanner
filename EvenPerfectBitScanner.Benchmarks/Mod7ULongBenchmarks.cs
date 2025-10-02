using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod7ULongBenchmarks
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Mod7(ulong value)
    {
        ulong remainder = ((uint)value % 7U) + (((uint)(value >> 32) % 7U) << 2);
        while (remainder >= 7UL)
        {
            remainder -= 7UL;
        }

        return remainder;
    }

    [Params(7UL, 8191UL, 131071UL, 2147483647UL, ulong.MaxValue - 63UL)]
    public ulong Value { get; set; }

    /// <summary>
    /// `% 7` baseline; 0.274 ns at value 7, 0.270 ns at 8191, 0.268 ns at 131071, 0.270 ns at 2147483647, and 0.275 ns near ulong.Max.
    /// </summary>
    [Benchmark(Baseline = true)]
    public ulong ModuloOperator()
    {
        return Value % 7UL;
    }

    /// <summary>
    /// Extension helper; 0.433 ns at value 7, 0.443 ns at 8191, 0.436 ns at 131071, 0.438 ns at 2147483647, but 0.684 ns near
    /// ulong.Max—about 1.6-2.5x slower overall.
    /// </summary>
    [Benchmark]
    public ulong ExtensionMethod()
    {
        return Mod7(Value);
    }
}
