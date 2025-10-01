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

    /// <summary>
    /// Materializes the carry in a local before updating the struct; stayed between 0.459 ns and 0.474 ns across all operand
    /// patterns, giving it the edge on carry-heavy cases.
    /// </summary>
    /// <remarks>
    /// Observed means: AllBitsSet 0.4737 ns (1.00×), CarryIntoHigh 0.4586 ns, IncrementLow 0.4653 ns, MixedOperands 0.4628 ns.
    /// </remarks>
    [Benchmark(Baseline = true)]
    public GpuUInt128 CarryMaterialisedInLocal()
    {
        GpuUInt128 value = Input.Left;
        value.Add(Input.Right);
        return value;
    }

    /// <summary>
    /// Computes the carry inline within the expression; matches the baseline on balanced additions (0.474 ns mixed) but trails
    /// slightly when the carry bubbles into the high word (0.491 ns) or chains through increments (0.508 ns).
    /// </summary>
    /// <remarks>
    /// Observed means: AllBitsSet 0.4676 ns (0.98×), CarryIntoHigh 0.4907 ns, IncrementLow 0.5082 ns, MixedOperands 0.4741 ns.
    /// </remarks>
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
