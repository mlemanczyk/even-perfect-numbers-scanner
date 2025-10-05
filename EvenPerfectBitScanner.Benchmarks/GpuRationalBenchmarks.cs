using BenchmarkDotNet.Attributes;
using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
public class GpuRationalBenchmarks
{
    private static readonly GpuRational Left = new(new GpuUInt128(3UL), new GpuUInt128(4UL));
    private static readonly GpuRational Right = new(new GpuUInt128(5UL), new GpuUInt128(6UL));
    private static readonly GpuUInt128 IntegerLeft = new(3UL);
    private static readonly GpuUInt128 IntegerRight = new(5UL);

    [Benchmark]
    public GpuRational AddRational()
    {
        GpuRational value = Left;
        value = value + Right;
        return value;
    }

    [Benchmark]
    public GpuUInt128 AddInteger()
    {
        GpuUInt128 value = IntegerLeft;
        value.Add(IntegerRight);
        return value;
    }
}
