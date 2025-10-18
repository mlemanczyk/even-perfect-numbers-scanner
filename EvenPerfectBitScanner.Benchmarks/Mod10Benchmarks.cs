using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod10Benchmarks
{
    private static readonly ModBenchmarkCase[] Cases =
    [
        new(ModNumericType.UInt32, 10U, "UInt32:10"),
        new(ModNumericType.UInt32, 2047U, "UInt32:2047"),
        new(ModNumericType.UInt32, 65535U, "UInt32:65535"),
        new(ModNumericType.UInt32, 2147483647U, "UInt32:2147483647"),
        new(ModNumericType.UInt64, 3UL, "UInt64:3"),
        new(ModNumericType.UInt64, 8191UL, "UInt64:8191"),
        new(ModNumericType.UInt64, 131071UL, "UInt64:131071"),
        new(ModNumericType.UInt64, 2147483647UL, "UInt64:2147483647"),
        new(ModNumericType.UInt128, 3UL, "UInt128:3"),
        new(ModNumericType.UInt128, 8191UL, "UInt128:8191"),
        new(ModNumericType.UInt128, 131071UL, "UInt128:131071"),
        new(ModNumericType.UInt128, 2147483647UL, "UInt128:2147483647"),
    ];

    private static readonly ModBenchmarkCase[] NonUInt128Cases =
    [
        new(ModNumericType.UInt32, 10U, "UInt32:10"),
        new(ModNumericType.UInt32, 2047U, "UInt32:2047"),
        new(ModNumericType.UInt32, 65535U, "UInt32:65535"),
        new(ModNumericType.UInt32, 2147483647U, "UInt32:2147483647"),
        new(ModNumericType.UInt64, 3UL, "UInt64:3"),
        new(ModNumericType.UInt64, 8191UL, "UInt64:8191"),
        new(ModNumericType.UInt64, 131071UL, "UInt64:131071"),
        new(ModNumericType.UInt64, 2147483647UL, "UInt64:2147483647"),
    ];

    private static readonly ModBenchmarkCase[] UInt128Cases =
    [
        new(ModNumericType.UInt128, 3UL, "UInt128:3"),
        new(ModNumericType.UInt128, 8191UL, "UInt128:8191"),
        new(ModNumericType.UInt128, 131071UL, "UInt128:131071"),
        new(ModNumericType.UInt128, 2147483647UL, "UInt128:2147483647"),
    ];

    public static IEnumerable<ModBenchmarkCase> GetCases() => Cases;

    public static IEnumerable<ModBenchmarkCase> GetNonUInt128Cases() => NonUInt128Cases;

    public static IEnumerable<ModBenchmarkCase> GetUInt128Cases() => UInt128Cases;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint Mod10(uint value) => value - (uint)((value * UIntExtensions.Mod5Mask) >> 35) * 10U;

    private static ulong ModWithHighLowModuloHelper(UInt128 value128)
    {
        ulong value = (ulong)value128;
        return ((value % 10UL) + ((value >> 64) % 10UL) * 6UL) % 10UL;
    }

    private static ulong ModWithLoopModuloHelper(UInt128 value)
    {
        UInt128 zero = UInt128.Zero;
        if (value == zero)
        {
            return 0UL;
        }

        ulong result = (ulong)value;
        value >>= 64;

        while (value != zero)
        {
            result = (result + (ulong)value * 6UL) % 10UL;
            value >>= 64;
        }

        return result % 10UL;
    }

    /// <summary>
    /// `% 10` baseline; ranged from 0.04-0.32 ns on <c>uint</c>, ~0.26 ns on <c>ulong</c>, and 3.25-3.38 ns on
    /// <see cref="UInt128"/> inputs across the consolidated dataset.
    /// </summary>
    [Benchmark(Baseline = true)]
    [ArgumentsSource(nameof(GetCases))]
    public ulong ModuloOperator(ModBenchmarkCase data)
    {
        return data.Type switch
        {
            ModNumericType.UInt32 => (uint)data.Value % 10U,
            ModNumericType.UInt64 => (ulong)data.Value % 10UL,
            ModNumericType.UInt128 => (ulong)(data.Value % 10UL),
            _ => throw new ArgumentOutOfRangeException(nameof(data)),
        };
    }

    /// <summary>
    /// Mask-based helpers for <c>uint</c> and <c>ulong</c>; they trail the raw `%` on 64-bit inputs and only match it on the
    /// smaller width.
    /// </summary>
    [Benchmark]
    [ArgumentsSource(nameof(GetNonUInt128Cases))]
    public ulong ExtensionMethod(ModBenchmarkCase data)
    {
        return data.Type switch
        {
            ModNumericType.UInt32 => Mod10((uint)data.Value),
            ModNumericType.UInt64 => ((ulong)data.Value).Mod10(),
            _ => throw new ArgumentOutOfRangeException(nameof(data)),
        };
    }

    /// <summary>
    /// High/low folding helper for <see cref="UInt128"/>; measured 1.78-1.82 ns in the original suite, about 1.9× faster than `%`.
    /// </summary>
    [Benchmark]
    [ArgumentsSource(nameof(GetUInt128Cases))]
    public ulong ModWithHighLowModulo(ModBenchmarkCase data)
    {
        return ModWithHighLowModuloHelper(data.Value);
    }

    /// <summary>
    /// Loop-based folding helper for <see cref="UInt128"/>; lands between the high/low helper and the raw modulo at 2.20-2.27 ns.
    /// </summary>
    [Benchmark]
    [ArgumentsSource(nameof(GetUInt128Cases))]
    public ulong ModWithLoopModulo(ModBenchmarkCase data)
    {
        return ModWithLoopModuloHelper(data.Value);
    }
}
