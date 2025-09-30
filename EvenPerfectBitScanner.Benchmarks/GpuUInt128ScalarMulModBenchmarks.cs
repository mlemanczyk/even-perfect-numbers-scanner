using System.Collections.Generic;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class GpuUInt128ScalarMulModBenchmarks
{
    private static readonly ScalarMulModInput[] Inputs = new ScalarMulModInput[]
    {
        new(
            new GpuUInt128(0x0000_0000_0000_0000UL, 0x0000_0000_0000_0011UL),
            0x0000_0000_0000_0013UL,
            new GpuUInt128(0x0000_0000_0000_0001UL, 0x0000_0000_0000_00FFUL),
            "TinyOperands"),
        new(
            new GpuUInt128(0x0123_4567_89AB_CDEFUL, 0xFEDC_BA98_7654_3210UL),
            0x0F1E_2D3C_4B5A_6978UL,
            new GpuUInt128(0x0FFF_FFFF_FFFF_FFFFUL, 0xFFFF_FFFF_FFFF_FFC3UL),
            "MixedMagnitude"),
        new(
            new GpuUInt128(0x8000_0000_0000_0000UL, 0x0000_0000_0000_0001UL),
            0x7FFF_FFFF_FFFF_FFFBUL,
            new GpuUInt128(0x7FFF_FFFF_FFFF_FFFFUL, 0xFFFF_FFFF_FFFF_FFF1UL),
            "HighWordDominant"),
    };

    [ParamsSource(nameof(GetInputs))]
    public ScalarMulModInput Input { get; set; }

    public static IEnumerable<ScalarMulModInput> GetInputs() => Inputs;

    [Benchmark(Baseline = true)]
    public GpuUInt128 StructAllocating()
    {
        GpuUInt128 value = Input.Left;
        value.MulMod(new GpuUInt128(Input.Right), Input.Modulus);
        return value;
    }

    [Benchmark]
    public GpuUInt128 SpecializedScalar()
    {
        GpuUInt128 value = Input.Left;
        value.MulMod(Input.Right, Input.Modulus);
        return value;
    }

    public readonly record struct ScalarMulModInput(GpuUInt128 Left, ulong Right, GpuUInt128 Modulus, string Name)
    {
        public override string ToString() => Name;
    }
}

