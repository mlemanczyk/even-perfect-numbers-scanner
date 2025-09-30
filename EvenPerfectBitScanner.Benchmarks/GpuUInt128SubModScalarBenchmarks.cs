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

    [Benchmark(Baseline = true)]
    public GpuUInt128 CurrentInPlace()
    {
        GpuUInt128 value = Input.Left;
        value.SubMod(Input.Scalar, Input.Modulus);
        return value;
    }

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
