using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
public class Mod8ULongBenchmarks
{
    [Params(8UL, 8191UL, 131071UL, 2147483647UL, ulong.MaxValue - 127UL)]
    public ulong Value { get; set; }

    /// <summary>
    /// Uses the built-in <c>% 8</c>; measured 0.0224 ns on Value=8, ~0 ns within the measurement noise floor for 8,191,
    /// 0.0069 ns for 131,071, 0.0117 ns for 2,147,483,647, and 0.107 ns for the near-maximum value, making it the fastest
    /// option except when the branch-free mask wins on huge operands.
    /// </summary>
    [Benchmark(Baseline = true)]
    public ulong ModuloOperator()
    {
        return Value % 8UL;
    }

    /// <summary>
    /// Applies the branch-free mask helper; costs 0.0344 ns on Value=8, 0.0296 ns on 8,191, 0.0114 ns on 131,071,
    /// 0.0106 ns on 2,147,483,647, and just 0.0109 ns on the near-maximum sample, overtaking <c>%</c> once the divisor
    /// approaches 2<sup>64</sup>.
    /// </summary>
    [Benchmark]
    public ulong ExtensionMethod()
    {
        return Mod8(Value);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Mod8(ulong value)
    {
        return value & 7UL;
    }
}
