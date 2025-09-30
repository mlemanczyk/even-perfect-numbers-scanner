using System.Collections.Generic;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class GpuPow2ModBenchmarks
{
    private static readonly Pow2ModInput[] Inputs = new Pow2ModInput[]
    {
        new(63UL, new GpuUInt128(0UL, 97UL), "SmallExponentSmallModulus"),
        new(1_048_575UL, new GpuUInt128(0UL, 0xFFFF_FFFBUL), "MediumExponentPrimeModulus"),
        new(6_710_886_400UL, new GpuUInt128(0x0000_0000_0000_0001UL, 0xFFFF_FFFF_FFFF_FFC3UL), "LargeExponentHighWordModulus"),
        new(ulong.MaxValue, new GpuUInt128(0xFFFF_FFFF_FFFF_FFFBUL, 0xFFFF_FFFF_FFFF_FFC5UL), "FullWidthExponentFullWidthModulus"),
    };

    [ParamsSource(nameof(GetInputs))]
    public Pow2ModInput Input { get; set; }

    public static IEnumerable<Pow2ModInput> GetInputs() => Inputs;

    [Benchmark(Baseline = true)]
    public GpuUInt128 ProcessEightBitWindows()
    {
        return GpuUInt128.Pow2Mod(Input.Exponent, Input.Modulus);
    }

    [Benchmark]
    public GpuUInt128 ProcessSingleBits()
    {
        return Pow2ModSingleBit(Input.Exponent, Input.Modulus);
    }

    private static GpuUInt128 Pow2ModSingleBit(ulong exponent, in GpuUInt128 modulus)
    {
        if (modulus.IsZero || modulus == GpuUInt128.One)
        {
            return new GpuUInt128();
        }

        GpuUInt128 result = new(1UL);
        GpuUInt128 baseVal = new(2UL);

        ulong e = exponent;
        while (e != 0UL)
        {
            if ((e & 1UL) != 0UL)
            {
                result.MulMod(baseVal, modulus);
            }

            baseVal.MulMod(baseVal, modulus);
            e >>= 1;
        }

        return result;
    }

    public readonly record struct Pow2ModInput(ulong Exponent, GpuUInt128 Modulus, string Name)
    {
        public override string ToString()
        {
            return Name;
        }
    }
}
