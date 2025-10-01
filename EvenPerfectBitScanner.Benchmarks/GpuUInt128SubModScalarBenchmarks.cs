using System.Collections.Generic;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class GpuUInt128SubModScalarBenchmarks
{
    private static readonly SubModInput[] Inputs = new SubModInput[]
    {
        new(new GpuUInt128(0UL, 3UL), 5UL, new GpuUInt128(0UL, 11UL), "UnderflowAddsModulus"),
        new(new GpuUInt128(0UL, ulong.MaxValue - 3UL), 7UL, new GpuUInt128(0UL, ulong.MaxValue - 11UL), "LargeLowWord"),
        new(new GpuUInt128(17UL, 2UL), 1UL, new GpuUInt128(3UL, 7UL), "HighWordBorrow"),
    };

    [ParamsSource(nameof(GetInputs))]
    public SubModInput Input { get; set; }

    public static IEnumerable<SubModInput> GetInputs() => Inputs;

    /// <summary>
    /// In-place subtraction that hits 0.509–0.717 ns across the scenarios, edging out the legacy helper especially when a borrow
    /// is required.
    /// </summary>
    /// <remarks>
    /// Observed means: HighWordBorrow 0.5112 ns (1.00×), LargeLowWord 0.5092 ns, UnderflowAddsModulus 0.7169 ns.
    /// </remarks>
    [Benchmark(Baseline = true)]
    public GpuUInt128 CurrentInPlace()
    {
        GpuUInt128 value = Input.Left;
        value.SubMod(Input.Scalar, Input.Modulus);
        return value;
    }

    /// <summary>
    /// Legacy version that stages intermediates in temporaries; nearly matches the in-place path but remains 1–4% slower on every
    /// case (0.514–0.722 ns).
    /// </summary>
    /// <remarks>
    /// Observed means: HighWordBorrow 0.5308 ns (1.04×), LargeLowWord 0.5144 ns, UnderflowAddsModulus 0.7224 ns.
    /// </remarks>
    [Benchmark]
    public GpuUInt128 LegacyTemporaries()
    {
        GpuUInt128 value = Input.Left;
        LegacySubMod(ref value, Input.Scalar, Input.Modulus);
        return value;
    }

    private static void LegacySubMod(ref GpuUInt128 value, ulong scalar, GpuUInt128 modulus)
    {
        ulong low;
        ulong high;
        if (value.High == 0UL && value.Low < scalar)
        {
            low = value.Low + modulus.Low;
            ulong carry = low < value.Low ? 1UL : 0UL;
            high = value.High + modulus.High + carry;
        }
        else
        {
            high = value.High;
            low = value.Low;
        }

        ulong borrow = low < scalar ? 1UL : 0UL;
        value.High = high - borrow;
        value.Low = low - scalar;
    }

    public readonly record struct SubModInput(GpuUInt128 Left, ulong Scalar, GpuUInt128 Modulus, string Name)
    {
        public override string ToString()
        {
            return Name;
        }
    }
}
