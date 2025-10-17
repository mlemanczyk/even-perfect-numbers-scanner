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

    /// <summary>
    /// Baseline that materializes the high product in a local; measured 1.53–1.55 ns across all operand patterns, keeping it in
    /// front of the inline variant.
    /// </summary>
    /// <remarks>
    /// Observed means: AllBitsSet 1.551 ns (1.00×), HighByLow 1.532 ns, LowWordMax 1.548 ns, TinyOperands 1.529 ns.
    /// </remarks>
    [Benchmark(Baseline = true)]
    public GpuUInt128 HighProductMaterializedInLocal()
    {
        GpuUInt128 value = Input.Multiplicand;
        value.Mul64(Input.Multiplier);
        return value;
    }

    /// <summary>
    /// Inline variant that avoids the temporary struct field update, but costs roughly 4–5% extra work with 1.59–1.63 ns means.
    /// </summary>
    /// <remarks>
    /// Observed means: AllBitsSet 1.631 ns (1.05×), HighByLow 1.594 ns, LowWordMax 1.623 ns, TinyOperands 1.602 ns.
    /// </remarks>
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
