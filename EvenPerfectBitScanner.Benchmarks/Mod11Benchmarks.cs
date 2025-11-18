using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod11Benchmarks
{
    private static readonly ModBenchmarkCase[] Cases =
    [
        new(ModNumericType.UInt32, 11U, "UInt32:11"),
        new(ModNumericType.UInt32, 2047U, "UInt32:2047"),
        new(ModNumericType.UInt32, 65535U, "UInt32:65535"),
        new(ModNumericType.UInt32, 2147483647U, "UInt32:2147483647"),
        new(ModNumericType.UInt64, 11UL, "UInt64:11"),
        new(ModNumericType.UInt64, 8191UL, "UInt64:8191"),
        new(ModNumericType.UInt64, 131071UL, "UInt64:131071"),
        new(ModNumericType.UInt64, 2147483647UL, "UInt64:2147483647"),
        new(ModNumericType.UInt64, ulong.MaxValue - 127UL, "UInt64:Max-127"),
        new(ModNumericType.UInt128, 11UL, "UInt128:11"),
        new(ModNumericType.UInt128, 8191UL, "UInt128:8191"),
        new(ModNumericType.UInt128, 131071UL, "UInt128:131071"),
        new(ModNumericType.UInt128, 2147483647UL, "UInt128:2147483647"),
        new(ModNumericType.UInt128, ulong.MaxValue - 255UL, "UInt128:Max-255"),
    ];

    public static IEnumerable<ModBenchmarkCase> GetCases() => Cases;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint Mod11(uint value)
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Mod11(ulong value)
    {
        ulong remainder = ((uint)value % 11U) + (((uint)(value >> 32) % 11U) << 2);
        while (remainder >= 11UL)
        {
            remainder -= 11UL;
        }

        return remainder;
    }

    /// <summary>
    /// `% 11` baseline across the consolidated dataset; near-empty-loop costs on <c>uint</c>, 0.23-0.27 ns on <c>ulong</c>, and
    /// 2.85-2.87 ns on <see cref="UInt128"/> inputs.
    /// </summary>
    [Benchmark(Baseline = true)]
    [ArgumentsSource(nameof(GetCases))]
    public ulong ModuloOperator(ModBenchmarkCase data)
    {
        return data.Type switch
        {
            ModNumericType.UInt32 => (uint)data.Value % 11U,
            ModNumericType.UInt64 => (ulong)data.Value % 11UL,
            ModNumericType.UInt128 => (ulong)(data.Value % 11UL),
            _ => throw new ArgumentOutOfRangeException(nameof(data)),
        };
    }

    /// <summary>
    /// Looping helpers; slower than `%` on <c>uint</c> and <c>ulong</c> but ~4.2× faster on <see cref="UInt128"/> thanks to digit
    /// folding.
    /// </summary>
    [Benchmark]
    [ArgumentsSource(nameof(GetCases))]
    public ulong ExtensionMethod(ModBenchmarkCase data)
    {
        return data.Type switch
        {
            ModNumericType.UInt32 => Mod11((uint)data.Value),
            ModNumericType.UInt64 => Mod11((ulong)data.Value),
            ModNumericType.UInt128 => data.Value.Mod11(),
            _ => throw new ArgumentOutOfRangeException(nameof(data)),
        };
    }
}
