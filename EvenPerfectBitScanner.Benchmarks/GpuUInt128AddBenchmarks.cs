using System.Collections.Generic;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class GpuUInt128AddBenchmarks
{
    private static readonly AddInput[] Inputs = new AddInput[]
    {
        new(new GpuUInt128(0UL, 0UL), new GpuUInt128(0UL, 1UL), "IncrementLow"),
        new(new GpuUInt128(0UL, ulong.MaxValue), new GpuUInt128(0UL, 1UL), "CarryIntoHigh"),
        new(new GpuUInt128(1UL, ulong.MaxValue), new GpuUInt128(3UL, 7UL), "MixedOperands"),
        new(new GpuUInt128(ulong.MaxValue, ulong.MaxValue), new GpuUInt128(ulong.MaxValue, ulong.MaxValue), "AllBitsSet"),
    };

    [ParamsSource(nameof(GetInputs))]
    public AddInput Input { get; set; }

    public static IEnumerable<AddInput> GetInputs() => Inputs;

    [Benchmark(Baseline = true)]
    public GpuUInt128 CarryMaterialisedInLocal()
    {
        GpuUInt128 value = Input.Left;
        value.Add(Input.Right);
        return value;
    }

    [Benchmark]
    public GpuUInt128 CarryInExpression()
    {
        GpuUInt128 value = Input.Left;
        AddInline(ref value, Input.Right);
        return value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddInline(ref GpuUInt128 left, GpuUInt128 right)
    {
        ulong low = left.Low + right.Low;
        left.High = left.High + right.High + (low < left.Low ? 1UL : 0UL);
        left.Low = low;
    }

    public readonly record struct AddInput(GpuUInt128 Left, GpuUInt128 Right, string Name)
    {
        public override string ToString()
        {
            return Name;
        }
    }
}
