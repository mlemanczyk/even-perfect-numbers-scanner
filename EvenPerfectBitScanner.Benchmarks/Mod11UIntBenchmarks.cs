using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod11UIntBenchmarks
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Mod11(uint value)
    {
        uint remainder = 0U;
        while (value != 0U)
        {
            remainder += value & 1023U;
            value >>= 10;
            remainder -= 11U * (remainder / 11U);
        }

        return remainder;
    }

    [Params(11U, 2047U, 65535U, 2147483647U)]
    public uint Value { get; set; }

    /// <summary>
    /// `% 11` baseline; essentially free at this scale with means of 0.0299 ns (value 11), 0.0333 ns (2047), 0.0393 ns (65535), and
    /// 0.0367 ns (2147483647).
    /// </summary>
    [Benchmark(Baseline = true)]
    public uint ModuloOperator()
    {
        return Value % 11U;
    }

    /// <summary>
    /// Looping extension method; useful for validation but slower across the board at 0.286 ns (value 11), 0.957 ns (2047),
    /// 0.922 ns (65535), and 3.170 ns (2147483647), i.e. 9-86x the direct modulo.
    /// </summary>
    [Benchmark]
    public uint ExtensionMethod()
    {
        return Mod11(Value);
    }
}
