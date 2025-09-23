using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 3)]
[MemoryDiagnoser]
public class Mod6ULongBenchmarks
{
	private static readonly byte[] Mod6Lookup = [0, 3, 4, 1, 2, 5];
    private static readonly ulong Divisor = 6UL;
    private static readonly ulong FastDivMul = (ulong)(((UInt128)1 << 64) / Divisor);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong Mod6(ulong value) => Mod6Lookup[(int)(((value % 3UL) << 1) | (value & 1UL))];

    [Params(3UL, 131071UL, ulong.MaxValue)]
    public ulong Value { get; set; }

	/// <summary>
	/// Fastest
	/// </summary>
	[Benchmark(Baseline = true)]
    public ulong ModuloOperator()
    {
                return Value % Divisor;
    }

    [Benchmark]
    public ulong DivisionBased()
    {
        ulong quotient = Value / Divisor;
        return Value - quotient * Divisor;
    }

    [Benchmark]
    public ulong FastDivHigh()
    {
        ulong quotient = Value.FastDiv64(Divisor, FastDivMul);
        return Value - quotient * Divisor;
    }

    [Benchmark]
    public ulong ExtensionMethod()
    {
        return Mod6(Value);
    }
}
