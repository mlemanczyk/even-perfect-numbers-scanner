using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod8Benchmarks
{
    private static readonly ModBenchmarkCase[] Cases =
    [
        new(ModNumericType.UInt32, 8U, "UInt32:8"),
        new(ModNumericType.UInt32, 2047U, "UInt32:2047"),
        new(ModNumericType.UInt32, 65535U, "UInt32:65535"),
        new(ModNumericType.UInt32, 2147483647U, "UInt32:2147483647"),
        new(ModNumericType.UInt64, 8UL, "UInt64:8"),
        new(ModNumericType.UInt64, 8191UL, "UInt64:8191"),
        new(ModNumericType.UInt64, 131071UL, "UInt64:131071"),
        new(ModNumericType.UInt64, 2147483647UL, "UInt64:2147483647"),
        new(ModNumericType.UInt64, ulong.MaxValue - 127UL, "UInt64:Max-127"),
        new(ModNumericType.UInt128, 8UL, "UInt128:8"),
        new(ModNumericType.UInt128, 8191UL, "UInt128:8191"),
        new(ModNumericType.UInt128, 131071UL, "UInt128:131071"),
        new(ModNumericType.UInt128, 2147483647UL, "UInt128:2147483647"),
        new(ModNumericType.UInt128, ulong.MaxValue - 255UL, "UInt128:Max-255"),
    ];

    public static IEnumerable<ModBenchmarkCase> GetCases() => Cases;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Mod8(ulong value)
    {
        return value & 7UL;
    }

    /// <summary>
    /// `% 8` baseline across the merged dataset; effectively at the noise floor on <c>uint</c> and <c>ulong</c>, and 2.86-2.89 ns
    /// on <see cref="UInt128"/> inputs.
    /// </summary>
    [Benchmark(Baseline = true)]
    [ArgumentsSource(nameof(GetCases))]
    public ulong ModuloOperator(ModBenchmarkCase data)
    {
        return data.Type switch
        {
            ModNumericType.UInt32 => (uint)data.Value % 8U,
            ModNumericType.UInt64 => (ulong)data.Value % 8UL,
            ModNumericType.UInt128 => (ulong)(data.Value % 8UL),
            _ => throw new ArgumentOutOfRangeException(nameof(data)),
        };
    }

    /// <summary>
    /// Bitmask helpers; match `%` on 32-bit inputs, are typically faster on <c>ulong</c>, and deliver 140-280× wins on
    /// <see cref="UInt128"/> by avoiding division.
    /// </summary>
    [Benchmark]
    [ArgumentsSource(nameof(GetCases))]
    public ulong ExtensionMethod(ModBenchmarkCase data)
    {
        return data.Type switch
        {
            ModNumericType.UInt32 => ((uint)data.Value).Mod8(),
            ModNumericType.UInt64 => Mod8((ulong)data.Value),
            ModNumericType.UInt128 => data.Value.Mod8(),
            _ => throw new ArgumentOutOfRangeException(nameof(data)),
        };
    }
}
