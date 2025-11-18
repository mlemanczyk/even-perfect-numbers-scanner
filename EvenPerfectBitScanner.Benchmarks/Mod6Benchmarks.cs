using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 3)]
[MemoryDiagnoser]
public class Mod6Benchmarks
{
    private static readonly byte[] Mod6Lookup = [0, 3, 4, 1, 2, 5];

    private static readonly ModBenchmarkCase[] Cases =
    [
        new(ModNumericType.UInt32, 6U, "UInt32:6"),
        new(ModNumericType.UInt32, 2047U, "UInt32:2047"),
        new(ModNumericType.UInt32, 65535U, "UInt32:65535"),
        new(ModNumericType.UInt32, 2147483647U, "UInt32:2147483647"),
        new(ModNumericType.UInt64, 3UL, "UInt64:3"),
        new(ModNumericType.UInt64, 131071UL, "UInt64:131071"),
        new(ModNumericType.UInt64, 2147483647UL, "UInt64:2147483647"),
        new(ModNumericType.UInt64, ulong.MaxValue - 1024UL, "UInt64:Max-1024"),
        new(ModNumericType.UInt128, 6UL, "UInt128:6"),
        new(ModNumericType.UInt128, 8191UL, "UInt128:8191"),
        new(ModNumericType.UInt128, 131071UL, "UInt128:131071"),
        new(ModNumericType.UInt128, 2147483647UL, "UInt128:2147483647"),
        new(ModNumericType.UInt128, ulong.MaxValue - 63UL, "UInt128:Max-63"),
    ];

    private static readonly ModBenchmarkCase[] UInt64Cases =
    [
        new(ModNumericType.UInt64, 3UL, "UInt64:3"),
        new(ModNumericType.UInt64, 131071UL, "UInt64:131071"),
        new(ModNumericType.UInt64, 2147483647UL, "UInt64:2147483647"),
        new(ModNumericType.UInt64, ulong.MaxValue - 1024UL, "UInt64:Max-1024"),
    ];

    private static readonly Mod6LoopCase[] LoopCases =
    [
        new(3UL, "Loop:3"),
        new(131071UL, "Loop:131071"),
        new(ulong.MaxValue - 1024UL, "Loop:Max-1024"),
    ];

    private const int Iterations = 10_000_000;
    private const ulong Divisor = 6UL;
    private static readonly ulong FastDivMul = (ulong)(((UInt128)1 << 64) / Divisor);

    public static IEnumerable<ModBenchmarkCase> GetCases() => Cases;

    public static IEnumerable<ModBenchmarkCase> GetUInt64Cases() => UInt64Cases;

    public static IEnumerable<Mod6LoopCase> GetLoopCases() => LoopCases;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint Mod6(uint value) => Mod6Lookup[(int)(((value % 3U) << 1) | (value & 1U))];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Mod6(ulong value) => Mod6Lookup[(int)(((value % 3UL) << 1) | (value & 1UL))];

    /// <summary>
    /// `% 6` baseline across the merged scalar dataset; previously timed at 0.03-0.05 ns on <c>uint</c>, 0.26-0.27 ns on
    /// <c>ulong</c>, and 2.87-2.91 ns on <see cref="UInt128"/> inputs.
    /// </summary>
    [Benchmark(Baseline = true)]
    [ArgumentsSource(nameof(GetCases))]
    public ulong ModuloOperator(ModBenchmarkCase data)
    {
        return data.Type switch
        {
            ModNumericType.UInt32 => (uint)data.Value % 6U,
            ModNumericType.UInt64 => (ulong)data.Value % 6UL,
            ModNumericType.UInt128 => (ulong)(data.Value % 6UL),
            _ => throw new ArgumentOutOfRangeException(nameof(data)),
        };
    }

    /// <summary>
    /// Division-based scalar helpers where available; matches the modulo cost on <c>ulong</c> inputs.
    /// </summary>
    [Benchmark]
    [ArgumentsSource(nameof(GetUInt64Cases))]
    public ulong DivisionBased(ModBenchmarkCase data)
    {
        ulong value = (ulong)data.Value;
        ulong quotient = value / Divisor;
        return value - quotient * Divisor;
    }

    /// <summary>
    /// FastDivHigh scalar helper for <c>ulong</c>; remains 1.0-1.6× slower than the `%` baseline across the dataset.
    /// </summary>
    [Benchmark]
    [ArgumentsSource(nameof(GetUInt64Cases))]
    public ulong FastDivHigh(ModBenchmarkCase data)
    {
        ulong value = (ulong)data.Value;
        ulong quotient = value.FastDiv64(Divisor, FastDivMul);
        return value - quotient * Divisor;
    }

    /// <summary>
    /// Lookup helpers; significantly slower on <c>uint</c> and <c>ulong</c> but 2.4-2.5× faster on <see cref="UInt128"/>.
    /// </summary>
    [Benchmark]
    [ArgumentsSource(nameof(GetCases))]
    public ulong ExtensionMethod(ModBenchmarkCase data)
    {
        return data.Type switch
        {
            ModNumericType.UInt32 => Mod6((uint)data.Value),
            ModNumericType.UInt64 => Mod6((ulong)data.Value),
            ModNumericType.UInt128 => data.Value.Mod6(),
            _ => throw new ArgumentOutOfRangeException(nameof(data)),
        };
    }

    /// <summary>
    /// `% 6` baseline inside the 10,000,000-iteration loop; measured 1.02-1.04 ns depending on the starting value.
    /// </summary>
    [Benchmark(Baseline = true, OperationsPerInvoke = Iterations)]
    [ArgumentsSource(nameof(GetLoopCases))]
    public ulong ModuloOperatorLoop(Mod6LoopCase data)
    {
        ulong checksum = 0UL;
        ulong value = data.StartValue;

        for (int i = 0; i < Iterations; i++)
        {
            checksum += value % Divisor;
            value += (ulong)((i & 15) + 1);
        }

        return checksum;
    }

    /// <summary>
    /// Division-based loop helper; mirrors the modulo timings with small deviations across the dataset.
    /// </summary>
    [Benchmark(OperationsPerInvoke = Iterations)]
    [ArgumentsSource(nameof(GetLoopCases))]
    public ulong DivisionBasedLoop(Mod6LoopCase data)
    {
        ulong checksum = 0UL;
        ulong value = data.StartValue;

        for (int i = 0; i < Iterations; i++)
        {
            ulong quotient = value / Divisor;
            checksum += value - quotient * Divisor;
            value += (ulong)((i & 15) + 1);
        }

        return checksum;
    }

    /// <summary>
    /// FastDivHigh loop helper; remained ~1.6× slower than the modulo baseline across the looped dataset.
    /// </summary>
    [Benchmark(OperationsPerInvoke = Iterations)]
    [ArgumentsSource(nameof(GetLoopCases))]
    public ulong FastDivHighLoop(Mod6LoopCase data)
    {
        ulong checksum = 0UL;
        ulong value = data.StartValue;

        for (int i = 0; i < Iterations; i++)
        {
            ulong quotient = value.FastDiv64(Divisor, FastDivMul);
            checksum += value - quotient * Divisor;
            value += (ulong)((i & 15) + 1);
        }

        return checksum;
    }

    /// <summary>
    /// Lookup-based loop helper; effectively matched the modulo baseline while removing the `%` instruction.
    /// </summary>
    [Benchmark(OperationsPerInvoke = Iterations)]
    [ArgumentsSource(nameof(GetLoopCases))]
    public ulong ExtensionMethodLoop(Mod6LoopCase data)
    {
        ulong checksum = 0UL;
        ulong value = data.StartValue;

        for (int i = 0; i < Iterations; i++)
        {
            checksum += Mod6(value);
            value += (ulong)((i & 15) + 1);
        }

        return checksum;
    }

    public readonly struct Mod6LoopCase
    {
        public Mod6LoopCase(ulong startValue, string display)
        {
            StartValue = startValue;
            Display = display;
        }

        public ulong StartValue { get; }

        private string Display { get; }

        public override string ToString()
        {
            return Display;
        }
    }
}
