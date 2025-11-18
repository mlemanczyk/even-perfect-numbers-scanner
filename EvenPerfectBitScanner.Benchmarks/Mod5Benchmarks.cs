using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod5Benchmarks
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
        new(ModNumericType.UInt64, 5UL, "UInt64:5"),
        new(ModNumericType.UInt64, 8191UL, "UInt64:8191"),
        new(ModNumericType.UInt64, 131071UL, "UInt64:131071"),
        new(ModNumericType.UInt64, 2147483647UL, "UInt64:2147483647"),
        new(ModNumericType.UInt64, ulong.MaxValue - 31UL, "UInt64:Max-31"),
        new(ModNumericType.UInt128, 5UL, "UInt128:5"),
        new(ModNumericType.UInt128, 8191UL, "UInt128:8191"),
        new(ModNumericType.UInt128, 131071UL, "UInt128:131071"),
        new(ModNumericType.UInt128, 2147483647UL, "UInt128:2147483647"),
        new(ModNumericType.UInt128, ulong.MaxValue - 31UL, "UInt128:Max-31"),
    ];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Mod5(ulong value) =>
        ((uint)(value & ULongExtensions.WordBitMask)
         + (uint)((value >> 16) & ULongExtensions.WordBitMask)
         + (uint)((value >> 32) & ULongExtensions.WordBitMask)
         + (uint)((value >> 48) & ULongExtensions.WordBitMask)) % 5UL;

    public static IEnumerable<ModBenchmarkCase> GetCases() => Cases;

    /// <summary>
    /// `% 5` baseline; maintained 0.03-0.12 ns on <c>uint</c>, 0.26-0.27 ns on <c>ulong</c>, and 2.84-2.93 ns on
    /// <see cref="UInt128"/> values across the unified dataset.
    /// </summary>
    [Benchmark(Baseline = true)]
    [ArgumentsSource(nameof(GetCases))]
    public ulong ModuloOperator(ModBenchmarkCase data)
    {
        return data.Type switch
        {
            ModNumericType.UInt32 => (uint)data.Value % 5U,
            ModNumericType.UInt64 => (ulong)data.Value % 5UL,
            ModNumericType.UInt128 => (ulong)(data.Value % 5UL),
            _ => throw new ArgumentOutOfRangeException(nameof(data)),
        };
    }

    /// <summary>
    /// Extension helpers; effectively tied with modulo on <c>uint</c>, about 1.6-1.7× slower on <c>ulong</c>, and roughly 6×
    /// faster than `%` on <see cref="UInt128"/> inputs.
    /// </summary>
    [Benchmark]
    [ArgumentsSource(nameof(GetCases))]
    public ulong ExtensionMethod(ModBenchmarkCase data)
    {
        return data.Type switch
        {
            ModNumericType.UInt32 => ((uint)data.Value).Mod5(),
            ModNumericType.UInt64 => Mod5((ulong)data.Value),
            ModNumericType.UInt128 => data.Value.Mod5(),
            _ => throw new ArgumentOutOfRangeException(nameof(data)),
        };
    }
}
