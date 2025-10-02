using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 3)]
[MemoryDiagnoser]
public class Mod6ComparisonBenchmarks
{
    private static readonly byte[] Mod6Lookup = [0, 3, 4, 1, 2, 5];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Mod6(ulong value) => Mod6Lookup[(int)(((value % 3UL) << 1) | (value & 1UL))];

    private const int Iterations = 10_000_000;
    private const ulong Divisor = 6UL;
    private static readonly ulong FastDivMul = (ulong)(((UInt128)1 << 64) / Divisor);

    [Params(3UL, 131071UL, ulong.MaxValue - 1024UL)]
    public ulong StartValue { get; set; }

    /// <summary>
    /// `% 6` baseline inside the loop; measured 1.02 ns at StartValue 3, 1.035 ns at 131071, and 1.011 ns near ulong.Max.
    /// </summary>
    [Benchmark(Baseline = true, OperationsPerInvoke = Iterations)]
    public ulong ModuloOperator()
    {
        ulong checksum = 0UL;
        ulong value = StartValue;

        for (int i = 0; i < Iterations; i++)
        {
            checksum += value % Divisor;
            value += (ulong)((i & 15) + 1);
        }

        return checksum;
    }

    /// <summary>
    /// `value / 6` with a multiply back; varied between 0.962 ns and 1.063 ns (0.9915 ns at StartValue 3, 1.0633 ns at 131071,
    /// 0.9617 ns near ulong.Max), roughly matching the baseline.
    /// </summary>
    [Benchmark(OperationsPerInvoke = Iterations)]
    public ulong DivisionBased()
    {
        ulong checksum = 0UL;
        ulong value = StartValue;

        for (int i = 0; i < Iterations; i++)
        {
            ulong quotient = value / Divisor;
            checksum += value - quotient * Divisor;
            value += (ulong)((i & 15) + 1);
        }

        return checksum;
    }

    /// <summary>
    /// FastDivHigh helper; landed at 1.67-1.71 ns (1.691 ns at StartValue 3, 1.706 ns at 131071, 1.673 ns near ulong.Max), making it
    /// ~1.65x slower than `%`.
    /// </summary>
    [Benchmark(OperationsPerInvoke = Iterations)]
    public ulong FastDivHigh()
    {
        ulong checksum = 0UL;
        ulong value = StartValue;

        for (int i = 0; i < Iterations; i++)
        {
            ulong quotient = value.FastDiv64(Divisor, FastDivMul);
            checksum += value - quotient * Divisor;
            value += (ulong)((i & 15) + 1);
        }

        return checksum;
    }

    /// <summary>
    /// Lookup-based helper argued for adoption; ran in 1.06-1.07 ns (1.0605 ns at StartValue 3, 1.0696 ns at 131071, 1.0604 ns near
    /// ulong.Max), essentially matching the baseline with slightly higher stability.
    /// </summary>
    [Benchmark(OperationsPerInvoke = Iterations)]
    public ulong ExtensionMethod()
    {
        ulong checksum = 0UL;
        ulong value = StartValue;

        for (int i = 0; i < Iterations; i++)
        {
            checksum += Mod6(value);
            value += (ulong)((i & 15) + 1);
        }

        return checksum;
    }
}
