using System.Numerics;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class GpuUInt128BinaryGcdBenchmarks
{
    private static readonly BinaryGcdInput[] Inputs = new BinaryGcdInput[]
    {
        new(new GpuUInt128(0UL, 48UL), new GpuUInt128(0UL, 18UL), "SmallOperands"),
        new(
            new GpuUInt128(0x0123_4567_89AB_CDEFUL, 0xFEDC_BA98_7654_3210UL),
            new GpuUInt128(0x0FED_CBA9_8765_4321UL, 0x0123_4567_89AB_CDEFUL),
            "HighEntropy"),
        new(new GpuUInt128(0UL, ulong.MaxValue), new GpuUInt128(0UL, ulong.MaxValue - 1UL), "LowWordHeavy"),
        new(new GpuUInt128(0x8000_0000_0000_0000UL, 0UL), new GpuUInt128(0x7FFF_FFFF_FFFF_FFFFUL, 0UL), "HighWordOnly"),
    };

    [ParamsSource(nameof(GetInputs))]
    public BinaryGcdInput Input { get; set; }

    public static IEnumerable<BinaryGcdInput> GetInputs() => Inputs;

    [Benchmark(Baseline = true)]
    public GpuUInt128 ReusedVariables()
    {
        return BinaryGcdWithReusedVariables(Input.Left, Input.Right);
    }

    [Benchmark]
    public GpuUInt128 TemporaryStructPerIteration()
    {
        return GpuUInt128.BinaryGcd(Input.Left, Input.Right);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static GpuUInt128 BinaryGcdWithReusedVariables(GpuUInt128 u, GpuUInt128 v)
    {
        if (u.IsZero)
        {
            return v;
        }

        if (v.IsZero)
        {
            return u;
        }

        ulong combinedLow = u.Low | v.Low;
        ulong combinedHigh = u.High | v.High;
        int shift = combinedLow != 0UL
            ? BitOperations.TrailingZeroCount(combinedLow)
            : 64 + BitOperations.TrailingZeroCount(combinedHigh);

        int zu = GpuUInt128.TrailingZeroCount(u);
        u >>= zu;

        do
        {
            int zv = GpuUInt128.TrailingZeroCount(v);
            v >>= zv;
            if (u > v)
            {
                (u, v) = (v, u);
            }

            v -= u;
        }
        while (!v.IsZero);

        return u << shift;
    }

    public readonly record struct BinaryGcdInput(GpuUInt128 Left, GpuUInt128 Right, string Name)
    {
        public override string ToString()
        {
            return Name;
        }
    }
}
