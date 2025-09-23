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

