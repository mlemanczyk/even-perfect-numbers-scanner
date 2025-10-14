using System.Collections.Generic;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class GpuPow2ModBenchmarks
{
    private const ulong TargetExponent = 138_000_001UL;

    private static readonly Pow2ModInput[] Inputs = new Pow2ModInput[]
    {
        new(TargetExponent, new GpuUInt128(0UL, 276_000_003UL), "P138M_K1_Modulus"),
        new(TargetExponent, new GpuUInt128(0UL, 1_380_000_011UL), "P138M_K5_Modulus"),
        new(276_000_002_000UL, new GpuUInt128(0UL, 276_000_002_001UL), "OrderCandidate_K1000"),
        new(TargetExponent, new GpuUInt128(7UL, 8_872_792_484_033_138_689UL), "P138M_K5e11_Modulus"),
        new(ulong.MaxValue, new GpuUInt128(0xFFFF_FFFF_FFFF_FFFBUL, 0xFFFF_FFFF_FFFF_FFC5UL), "FullWidthStress"),
    };

    [ParamsSource(nameof(GetInputs))]
    public Pow2ModInput Input { get; set; }

    public static IEnumerable<Pow2ModInput> GetInputs() => Inputs;

    /// <summary>
    /// Adaptive windowed exponentiation tuned for the 138M+ Mersenne workloads. The helper now picks a GPU-friendly window size
    /// per exponent and reuses a stack-allocated table so we avoid rebuilding all 128 odd powers for every scan.
    /// </summary>
    [Benchmark(Baseline = true)]
    public GpuUInt128 ProcessEightBitWindows()
    {
        return GpuUInt128.Pow2ModWindowed(Input.Exponent, Input.Modulus);
    }

    /// <summary>
    /// Bit-by-bit fallback kept for comparison and for callers that insist on the legacy ladder. The 138M benchmarks still keep
    /// it around the GPU sweeps, but production code should rarely need it once batching kicks in.
    /// </summary>
    [Benchmark]
    public GpuUInt128 ProcessSingleBits()
    {
        return Pow2ModSingleBit(Input.Exponent, Input.Modulus);
    }

    private static GpuUInt128 Pow2ModSingleBit(ulong exponent, GpuUInt128 modulus)
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
