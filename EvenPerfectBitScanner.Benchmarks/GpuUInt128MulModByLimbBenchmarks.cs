using System.Collections.Generic;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class GpuUInt128MulModByLimbBenchmarks
{
    private static readonly MulModByLimbInput[] Inputs = new MulModByLimbInput[]
    {
        new(
            new GpuUInt128(0UL, 17UL),
            new GpuUInt128(0UL, 19UL),
            new GpuUInt128(0x0000_0000_0000_0001UL, 0x0000_0000_0000_00FFUL),
            "TinyOperands"),
        new(
            new GpuUInt128(0x0123_4567_89AB_CDEFUL, 0xFEDC_BA98_7654_3210UL),
            new GpuUInt128(0x0F1E_2D3C_4B5A_6978UL, 0x8776_6554_4332_2110UL),
            new GpuUInt128(0x0FFF_FFFF_FFFF_FFFFUL, 0xFFFF_FFFF_FFFF_FFC3UL),
            "MixedMagnitude"),
        new(
            new GpuUInt128(0x8000_0000_0000_0000UL, 0x0000_0000_0000_0001UL),
            new GpuUInt128(0x7FFF_FFFF_FFFF_FFFFUL, 0x8000_0000_0000_0000UL),
            new GpuUInt128(0x7FFF_FFFF_FFFF_FFFFUL, 0xFFFF_FFFF_FFFF_FFF1UL),
            "HighWordDominant"),
    };

    [ParamsSource(nameof(GetInputs))]
    public MulModByLimbInput Input { get; set; }

    public static IEnumerable<MulModByLimbInput> GetInputs() => Inputs;

    [Benchmark(Baseline = true)]
    public GpuUInt128 LegacyAllocating()
    {
        return MulModByLimbLegacy(Input.Left, Input.Right, Input.Modulus);
    }

    [Benchmark]
    public GpuUInt128 InPlaceReduction()
    {
        // TODO: Switch the extension method back to legacy method, as it's faster. Keep the benchmarks.
        GpuUInt128 value = Input.Left;
        value.MulModByLimb(Input.Right, Input.Modulus);
        return value;
    }

    private static GpuUInt128 MulModByLimbLegacy(GpuUInt128 left, GpuUInt128 right, GpuUInt128 modulus)
    {
        var (p3, p2, p1, p0) = MultiplyFullLegacy(left, right);

        GpuUInt128 remainder = new(p3, p2);
        while (remainder.CompareTo(modulus) >= 0)
        {
            remainder.Sub(modulus);
        }

        ulong limb = p1;
        for (int i = 0; i < 2; i++)
        {
            remainder <<= 64;
            remainder = new GpuUInt128(remainder.High, limb);
            while (remainder.CompareTo(modulus) >= 0)
            {
                remainder.Sub(modulus);
            }

            limb = p0;
        }

        return remainder;
    }

    private static (ulong P3, ulong P2, ulong P1, ulong P0) MultiplyFullLegacy(GpuUInt128 left, GpuUInt128 right)
    {
        var (h0, l0) = Mul64Legacy(left.Low, right.Low);
        var (h1, l1) = Mul64Legacy(left.Low, right.High);
        var (h2, l2) = Mul64Legacy(left.High, right.Low);
        var (h3, l3) = Mul64Legacy(left.High, right.High);

        ulong carry = 0UL;
        ulong sum0 = l0;
        ulong sum1 = h0;

        sum1 += l1;
        if (sum1 < l1)
        {
            carry++;
        }

        sum1 += l2;
        if (sum1 < l2)
        {
            carry++;
        }

        ulong sum2 = h1 + h2;
        ulong carry2 = sum2 < h2 ? 1UL : 0UL;
        sum2 += l3;
        if (sum2 < l3)
        {
            carry2++;
        }

        sum2 += carry;
        if (sum2 < carry)
        {
            carry2++;
        }

        ulong p1 = sum1;
        ulong p2 = sum2;
        ulong p3 = h3 + carry2;

        return (p3, p2, p1, sum0);
    }

    private static (ulong High, ulong Low) Mul64Legacy(ulong left, ulong right)
    {
        ulong a0 = (uint)left;
        ulong a1 = left >> 32;
        ulong b0 = (uint)right;
        ulong b1 = right >> 32;

        ulong lo = a0 * b0;
        ulong mid1 = a1 * b0;
        b0 = a0 * b1;
        b1 *= a1;

        a0 = (lo >> 32) + (uint)mid1 + (uint)b0;
        a1 = (lo & 0xFFFF_FFFFUL) | (a0 << 32);
        b1 += (mid1 >> 32) + (b0 >> 32) + (a0 >> 32);

        return (b1, a1);
    }

    public readonly record struct MulModByLimbInput(GpuUInt128 Left, GpuUInt128 Right, GpuUInt128 Modulus, string Name)
    {
        public override string ToString() => Name;
    }
}

