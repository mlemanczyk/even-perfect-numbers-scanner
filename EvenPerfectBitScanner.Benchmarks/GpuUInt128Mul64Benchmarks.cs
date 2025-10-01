using System.Collections.Generic;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class GpuUInt128Mul64Benchmarks
{
    private static readonly Mul64Input[] Inputs = new Mul64Input[]
    {
        new(new GpuUInt128(0UL, 1UL), new GpuUInt128(0UL, 1UL), "TinyOperands"),
        new(new GpuUInt128(0UL, ulong.MaxValue), new GpuUInt128(0UL, ulong.MaxValue), "LowWordMax"),
        new(new GpuUInt128(1UL, 0UL), new GpuUInt128(0UL, ulong.MaxValue), "HighByLow"),
        new(new GpuUInt128(ulong.MaxValue, ulong.MaxValue), new GpuUInt128(ulong.MaxValue, ulong.MaxValue), "AllBitsSet"),
    };

    [ParamsSource(nameof(GetInputs))]
    public Mul64Input Input { get; set; }

    public static IEnumerable<Mul64Input> GetInputs() => Inputs;

    [Benchmark(Baseline = true)]
    public GpuUInt128 HighProductMaterializedInLocal()
    {
        GpuUInt128 value = Input.Multiplicand;
        value.Mul64(Input.Multiplier);
        return value;
    }

    [Benchmark]
    public GpuUInt128 HighProductInline()
    {
        GpuUInt128 value = Input.Multiplicand;
        Mul64Inline(ref value, Input.Multiplier);
        return value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Mul64Inline(ref GpuUInt128 left, GpuUInt128 right)
    {
        ulong operand = left.Low;
        left.Low = operand * right.Low;
        left.High = operand * right.High + GpuUInt128.MulHigh(operand, right.Low);
    }

    public readonly record struct Mul64Input(GpuUInt128 Multiplicand, GpuUInt128 Multiplier, string Name)
    {
        public override string ToString()
        {
            return Name;
        }
    }
}
