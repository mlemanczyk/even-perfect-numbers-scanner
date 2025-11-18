using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod7Benchmarks
{
    private static readonly ModBenchmarkCase[] Cases =
    [
        new(ModNumericType.UInt32, 7U, "UInt32:7"),
        new(ModNumericType.UInt32, 2047U, "UInt32:2047"),
        new(ModNumericType.UInt32, 65535U, "UInt32:65535"),
        new(ModNumericType.UInt32, 2147483647U, "UInt32:2147483647"),
        new(ModNumericType.UInt64, 7UL, "UInt64:7"),
        new(ModNumericType.UInt64, 8191UL, "UInt64:8191"),
        new(ModNumericType.UInt64, 131071UL, "UInt64:131071"),
        new(ModNumericType.UInt64, 2147483647UL, "UInt64:2147483647"),
        new(ModNumericType.UInt64, ulong.MaxValue - 63UL, "UInt64:Max-63"),
        new(ModNumericType.UInt128, 7UL, "UInt128:7"),
        new(ModNumericType.UInt128, 8191UL, "UInt128:8191"),
        new(ModNumericType.UInt128, 131071UL, "UInt128:131071"),
        new(ModNumericType.UInt128, 2147483647UL, "UInt128:2147483647"),
        new(ModNumericType.UInt128, ulong.MaxValue - 127UL, "UInt128:Max-127"),
    ];

    public static IEnumerable<ModBenchmarkCase> GetCases() => Cases;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint Mod7(uint value)
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Mod7(ulong value)
    {
        ulong remainder = ((uint)value % 7U) + (((uint)(value >> 32) % 7U) << 2);
        while (remainder >= 7UL)
        {
            remainder -= 7UL;
        }

        return remainder;
    }

    /// <summary>
    /// `% 7` baseline across the merged dataset; hovered near the noise floor on <c>uint</c>, 0.27 ns on <c>ulong</c>, and
    /// 2.86-2.91 ns on <see cref="UInt128"/> inputs.
    /// </summary>
    [Benchmark(Baseline = true)]
    [ArgumentsSource(nameof(GetCases))]
    public ulong ModuloOperator(ModBenchmarkCase data)
    {
        return data.Type switch
        {
            ModNumericType.UInt32 => (uint)data.Value % 7U,
            ModNumericType.UInt64 => (ulong)data.Value % 7UL,
            ModNumericType.UInt128 => (ulong)(data.Value % 7UL),
            _ => throw new ArgumentOutOfRangeException(nameof(data)),
        };
    }

    /// <summary>
    /// Extension helpers; dramatically slower on <c>uint</c>, modestly slower on <c>ulong</c>, and 3.2-4.2× faster on
    /// <see cref="UInt128"/> inputs where the folding helpers shine.
    /// </summary>
    [Benchmark]
    [ArgumentsSource(nameof(GetCases))]
    public ulong ExtensionMethod(ModBenchmarkCase data)
    {
        return data.Type switch
        {
            ModNumericType.UInt32 => Mod7((uint)data.Value),
            ModNumericType.UInt64 => Mod7((ulong)data.Value),
            ModNumericType.UInt128 => data.Value.Mod7(),
            _ => throw new ArgumentOutOfRangeException(nameof(data)),
        };
    }
}
