using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod3Benchmarks
{
    private static readonly ModBenchmarkCase[] Cases =
    [
        new(ModNumericType.UInt32, 3U, "UInt32:3"),
        new(ModNumericType.UInt32, 2047U, "UInt32:2047"),
        new(ModNumericType.UInt32, 65535U, "UInt32:65535"),
        new(ModNumericType.UInt32, 2147483647U, "UInt32:2147483647"),
        new(ModNumericType.UInt64, 3UL, "UInt64:3"),
        new(ModNumericType.UInt64, 8191UL, "UInt64:8191"),
        new(ModNumericType.UInt64, 131071UL, "UInt64:131071"),
        new(ModNumericType.UInt64, 2147483647UL, "UInt64:2147483647"),
        new(ModNumericType.UInt64, ulong.MaxValue - 17UL, "UInt64:Max-17"),
        new(ModNumericType.UInt128, 3UL, "UInt128:3"),
        new(ModNumericType.UInt128, 8191UL, "UInt128:8191"),
        new(ModNumericType.UInt128, 131071UL, "UInt128:131071"),
        new(ModNumericType.UInt128, 2147483647UL, "UInt128:2147483647"),
        new(ModNumericType.UInt128, ulong.MaxValue - 17UL, "UInt128:Max-17"),
    ];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Mod3(ulong value) =>
        ((uint)(value & ULongExtensions.WordBitMask)
         + (uint)((value >> 16) & ULongExtensions.WordBitMask)
         + (uint)((value >> 32) & ULongExtensions.WordBitMask)
         + (uint)((value >> 48) & ULongExtensions.WordBitMask)) % 3UL;

    public static IEnumerable<ModBenchmarkCase> GetCases() => Cases;

    /// <summary>
    /// `% 3` baseline across all widths; previously landed at 0.03-0.09 ns on <c>uint</c>, 0.26-0.49 ns on <c>ulong</c>,
    /// and 2.88-2.97 ns on <see cref="UInt128"/> inputs throughout the merged dataset.
    /// </summary>
    [Benchmark(Baseline = true)]
    [ArgumentsSource(nameof(GetCases))]
    public ulong ModuloOperator(ModBenchmarkCase data)
    {
        return data.Type switch
        {
            ModNumericType.UInt32 => (uint)data.Value % 3U,
            ModNumericType.UInt64 => (ulong)data.Value % 3UL,
            ModNumericType.UInt128 => (ulong)(data.Value % 3UL),
            _ => throw new ArgumentOutOfRangeException(nameof(data)),
        };
    }

    /// <summary>
    /// Extension helpers that fold operands by width; they matched the modulo noise floor on <c>uint</c>, trailed by
    /// ~1.5-1.7× on <c>ulong</c>, and delivered a 6.4× speedup on <see cref="UInt128"/>.
    /// </summary>
    [Benchmark]
    [ArgumentsSource(nameof(GetCases))]
    public ulong ExtensionMethod(ModBenchmarkCase data)
    {
        return data.Type switch
        {
            ModNumericType.UInt32 => UIntExtensions.Mod3((uint)data.Value),
            ModNumericType.UInt64 => Mod3((ulong)data.Value),
            ModNumericType.UInt128 => data.Value.Mod3(),
            _ => throw new ArgumentOutOfRangeException(nameof(data)),
        };
    }
}
