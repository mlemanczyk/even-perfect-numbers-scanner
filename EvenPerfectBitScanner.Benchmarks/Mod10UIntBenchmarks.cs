using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class Mod10UIntBenchmarks
{
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static uint Mod10(uint value) => value - (uint)((value * UIntExtensions.Mod5Mask) >> 35) * 10U;

    [Params(10U, 2047U, 65535U, 2147483647U)]
    public uint Value { get; set; }

	/// <summary>
	/// Fastest
	/// </summary>
	[Benchmark(Baseline = true)]
    public uint ModuloOperator()
    {
        return Value % 10U;
    }

    [Benchmark]
    public uint ExtensionMethod()
    {
        return Mod10(Value);
    }
}

