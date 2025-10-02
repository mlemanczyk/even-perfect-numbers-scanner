using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod7UIntBenchmarks
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Mod7(uint value)
    {
        uint remainder = 0U;
        while (value != 0U)
        {
            remainder += value & 7U;
            value >>= 3;
            if (remainder >= 7U)
            {
                remainder -= 7U;
            }
        }

        return remainder >= 7U ? remainder - 7U : remainder;
    }

    [Params(7U, 2047U, 65535U, 2147483647U)]
    public uint Value { get; set; }

    /// <summary>
    /// `% 7` baseline; again near the noise floor with means of 0.0354 ns (value 7), 0.0276 ns (2047), 0.0282 ns (65535), and
    /// 0.0422 ns (2147483647).
    /// </summary>
    [Benchmark(Baseline = true)]
    public uint ModuloOperator()
    {
        return Value % 7U;
    }

    /// <summary>
    /// Extension helper; substantially slower at 0.268 ns (value 7), 1.217 ns (2047), 2.334 ns (65535), and 4.579 ns (2147483647)—up to
    /// 110x the baseline.
    /// </summary>
    [Benchmark]
    public uint ExtensionMethod()
    {
        return Mod7(Value);
    }
}
