using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod128Benchmarks
{
    private static readonly ModBenchmarkCase[] Cases =
    [
        new(ModNumericType.UInt32, 3U, "UInt32:3"),
        new(ModNumericType.UInt32, 5U, "UInt32:5"),
        new(ModNumericType.UInt32, 6U, "UInt32:6"),
        new(ModNumericType.UInt32, 8U, "UInt32:8"),
        new(ModNumericType.UInt32, 10U, "UInt32:10"),
        new(ModNumericType.UInt32, 11U, "UInt32:11"),
        new(ModNumericType.UInt32, 32U, "UInt32:32"),
        new(ModNumericType.UInt32, 64U, "UInt32:64"),
        new(ModNumericType.UInt32, 128U, "UInt32:128"),
        new(ModNumericType.UInt32, 256U, "UInt32:256"),
        new(ModNumericType.UInt32, 2047U, "UInt32:2047"),
        new(ModNumericType.UInt32, 8191U, "UInt32:8191"),
        new(ModNumericType.UInt32, 65535U, "UInt32:65535"),
        new(ModNumericType.UInt32, 131071U, "UInt32:131071"),
        new(ModNumericType.UInt32, 2147483647U, "UInt32:2147483647"),
        new(ModNumericType.UInt32, uint.MaxValue - 1024U, "UInt32:Max-1024"),
        new(ModNumericType.UInt32, uint.MaxValue - 511U, "UInt32:Max-511"),
        new(ModNumericType.UInt32, uint.MaxValue - 255U, "UInt32:Max-255"),
        new(ModNumericType.UInt32, uint.MaxValue - 127U, "UInt32:Max-127"),
        new(ModNumericType.UInt32, uint.MaxValue - 63U, "UInt32:Max-63"),
        new(ModNumericType.UInt32, uint.MaxValue - 31U, "UInt32:Max-31"),
        new(ModNumericType.UInt32, uint.MaxValue - 17U, "UInt32:Max-17"),
        new(ModNumericType.UInt32, uint.MaxValue, "UInt32:Max"),
        new(ModNumericType.UInt64, 3UL, "UInt64:3"),
        new(ModNumericType.UInt64, 5UL, "UInt64:5"),
        new(ModNumericType.UInt64, 6UL, "UInt64:6"),
        new(ModNumericType.UInt64, 8UL, "UInt64:8"),
        new(ModNumericType.UInt64, 10UL, "UInt64:10"),
        new(ModNumericType.UInt64, 11UL, "UInt64:11"),
        new(ModNumericType.UInt64, 32UL, "UInt64:32"),
        new(ModNumericType.UInt64, 64UL, "UInt64:64"),
        new(ModNumericType.UInt64, 128UL, "UInt64:128"),
        new(ModNumericType.UInt64, 256UL, "UInt64:256"),
        new(ModNumericType.UInt64, 2047UL, "UInt64:2047"),
        new(ModNumericType.UInt64, 8191UL, "UInt64:8191"),
        new(ModNumericType.UInt64, 65535UL, "UInt64:65535"),
        new(ModNumericType.UInt64, 131071UL, "UInt64:131071"),
        new(ModNumericType.UInt64, 2147483647UL, "UInt64:2147483647"),
        new(ModNumericType.UInt64, 4294966271UL, "UInt64:4294966271"),
        new(ModNumericType.UInt64, 4294966784UL, "UInt64:4294966784"),
        new(ModNumericType.UInt64, 4294967040UL, "UInt64:4294967040"),
        new(ModNumericType.UInt64, 4294967168UL, "UInt64:4294967168"),
        new(ModNumericType.UInt64, 4294967232UL, "UInt64:4294967232"),
        new(ModNumericType.UInt64, 4294967264UL, "UInt64:4294967264"),
        new(ModNumericType.UInt64, 4294967278UL, "UInt64:4294967278"),
        new(ModNumericType.UInt64, 4294967295UL, "UInt64:4294967295"),
        new(ModNumericType.UInt128, 128UL, "UInt128:128"),
        new(ModNumericType.UInt128, 8191UL, "UInt128:8191"),
        new(ModNumericType.UInt128, 131071UL, "UInt128:131071"),
        new(ModNumericType.UInt128, 2147483647UL, "UInt128:2147483647"),
        new(ModNumericType.UInt128, ulong.MaxValue - 511UL, "UInt128:Max-511"),
    ];

    public static IEnumerable<ModBenchmarkCase> GetCases() => Cases;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint Mod128(uint value) => value & 127U;

    /// <summary>
    /// `% 128` baseline; sits at 0.01-0.03 ns on <c>uint</c> and <c>ulong</c>, and 2.95-3.42 ns on <see cref="UInt128"/> inputs.
    /// </summary>
    [Benchmark(Baseline = true)]
    [ArgumentsSource(nameof(GetCases))]
    public ulong ModuloOperator(ModBenchmarkCase data)
    {
        return data.Type switch
        {
            ModNumericType.UInt32 => (uint)data.Value % 128U,
            ModNumericType.UInt64 => (ulong)data.Value % 128UL,
            ModNumericType.UInt128 => (ulong)(data.Value % 128UL),
            _ => throw new ArgumentOutOfRangeException(nameof(data)),
        };
    }

    /// <summary>
    /// Bitmask helpers; usually match `%` on 32-bit inputs, are faster on <c>ulong</c> with the exception of the 4,294,966,784
    /// outlier, and provide 150-1000× wins on <see cref="UInt128"/>.
    /// </summary>
    [Benchmark]
    [ArgumentsSource(nameof(GetCases))]
    public ulong ExtensionMethod(ModBenchmarkCase data)
    {
        return data.Type switch
        {
            ModNumericType.UInt32 => Mod128((uint)data.Value),
            ModNumericType.UInt64 => ((ulong)data.Value).Mod128(),
            ModNumericType.UInt128 => data.Value.Mod128(),
            _ => throw new ArgumentOutOfRangeException(nameof(data)),
        };
    }
}
