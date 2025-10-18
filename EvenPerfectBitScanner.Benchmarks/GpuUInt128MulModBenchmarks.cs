using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class GpuUInt128MulModBenchmarks
{
    private static readonly GpuMulModCase[] GeneralCases =
    [
        new(
            new GpuUInt128(0UL, 5UL),
            new GpuUInt128(0UL, 7UL),
            new GpuUInt128(0UL, 97UL),
            "TinyOperands"),
        new(
            new GpuUInt128(0UL, ulong.MaxValue - 3UL),
            new GpuUInt128(0UL, ulong.MaxValue - 7UL),
            new GpuUInt128(0UL, ulong.MaxValue - 11UL),
            "LowWordHeavy"),
        new(
            new GpuUInt128(0x1234_5678_9ABC_DEF0UL, 0x0FED_CBA9_8765_4321UL),
            new GpuUInt128(0x0ACE_BDF0_1357_9BDFUL, 0x0246_8ACE_ECA8_6421UL),
            new GpuUInt128(0x1FFF_FFFF_FFFF_FFFFUL, 0xFFFF_FFFF_FFFF_FFF7UL),
            "HighWordModulus"),
        new(
            new GpuUInt128(0x8000_0000_0000_0000UL, 0x0000_0000_0000_0001UL),
            new GpuUInt128(0x7FFF_FFFF_FFFF_FFFFUL, 0x8000_0000_0000_0000UL),
            new GpuUInt128(0x7FFF_FFFF_FFFF_FFFFUL, 0xFFFF_FFFF_FFFF_FFF1UL),
            "MixedMagnitude"),
    ];

    private static readonly GpuMulModCase[] ByLimbCases =
    [
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
    ];

    public static IEnumerable<GpuMulModCase> GetGeneralCases() => GeneralCases;

    public static IEnumerable<GpuMulModCase> GetByLimbCases() => ByLimbCases;

    /// <summary>
    /// In-place multiply-modulo; great for low-word-heavy operands (6.88 ns) and tiny pairs (7.15 ns) but notably slower on
    /// large moduli (193–201 ns).
    /// </summary>
    /// <remarks>
    /// Observed means: HighWordModulus 192.794 ns (1.00×), LowWordHeavy 6.883 ns, MixedMagnitude 201.468 ns, TinyOperands 7.146 ns.
    /// </remarks>
    [Benchmark(Baseline = true)]
    [ArgumentsSource(nameof(GetGeneralCases))]
    public GpuUInt128 InPlaceMulMod(GpuMulModCase input)
    {
        // TODO: Switch the extension method to allocate per iteration, as it's faster. Keep the benchmarks.
        GpuUInt128 value = input.Left;
        value.MulMod(input.Right, input.Modulus);
        return value;
    }

    /// <summary>
    /// Allocates fresh temporaries per iteration; pays off on every dataset with 4.66 ns on tiny operands, 117 ns on
    /// low-word-heavy inputs, 138 ns on mixed magnitude, and 184 ns on high-word moduli.
    /// </summary>
    /// <remarks>
    /// Observed means: HighWordModulus 184.322 ns (0.96×), LowWordHeavy 116.993 ns, MixedMagnitude 138.684 ns, TinyOperands 4.659 ns.
    /// </remarks>
    [Benchmark]
    [ArgumentsSource(nameof(GetGeneralCases))]
    public GpuUInt128 AllocatePerIteration(GpuMulModCase input)
    {
        return MulModInline(input.Left, input.Right, input.Modulus);
    }

    /// <summary>
    /// Legacy implementation that allocates intermediate limbs; delivered 10.7 ns on tiny operands, 12.7 ns on high-word-dominant
    /// pairs, and 16.3 ns on mixed magnitude, making it the faster choice overall.
    /// </summary>
    /// <remarks>
    /// Observed means: HighWordDominant 12.69 ns (1.00×), MixedMagnitude 16.26 ns, TinyOperands 10.74 ns.
    /// </remarks>
    [Benchmark(Baseline = true)]
    [ArgumentsSource(nameof(GetByLimbCases))]
    public GpuUInt128 LegacyAllocating(GpuMulModCase input)
    {
        return MulModByLimbLegacy(input.Left, input.Right, input.Modulus);
    }

    /// <summary>
    /// In-place reduction variant; avoids allocations but costs 12.9–25.1 ns, trailing the legacy path by 20–55% depending on the
    /// operand mix.
    /// </summary>
    /// <remarks>
    /// Observed means: HighWordDominant 17.88 ns (1.41×), MixedMagnitude 25.14 ns, TinyOperands 12.92 ns.
    /// </remarks>
    [Benchmark]
    [ArgumentsSource(nameof(GetByLimbCases))]
    public GpuUInt128 InPlaceReduction(GpuMulModCase input)
    {
        // TODO: Switch the extension method back to legacy method, as it's faster. Keep the benchmarks.
        GpuUInt128 value = input.Left;
        value.MulModByLimb(input.Right, input.Modulus);
        return value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static GpuUInt128 MulModInline(GpuUInt128 left, GpuUInt128 right, GpuUInt128 modulus)
    {
        // Reuse the variables to new instances and lower registry pressure, if possible.
        GpuUInt128 a = left;
        GpuUInt128 b = right;
        GpuUInt128 result = new();

        while (!b.IsZero)
        {
            if ((b.Low & 1UL) != 0UL)
            {
                result.Add(a);
                if (result.CompareTo(modulus) >= 0)
                {
                    result.Sub(modulus);
                }
            }

            a <<= 1;
            if (a.CompareTo(modulus) >= 0)
            {
                a.Sub(modulus);
            }

            if (b.High == 0UL)
            {
                b = new GpuUInt128(b.Low >> 1);
            }
            else
            {
                ulong low = (b.Low >> 1) | (b.High << 63);
                ulong high = b.High >> 1;
                b = new GpuUInt128(high, low);
            }
        }

        return result;
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

    public readonly record struct GpuMulModCase(GpuUInt128 Left, GpuUInt128 Right, GpuUInt128 Modulus, string Name)
    {
        public override string ToString()
        {
            return Name;
        }
    }
}
