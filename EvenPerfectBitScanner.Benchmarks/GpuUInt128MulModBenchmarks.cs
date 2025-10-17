using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class GpuUInt128MulModBenchmarks
{
    private static readonly MulModInput[] Inputs =
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

    [ParamsSource(nameof(GetInputs))]
    public MulModInput Input { get; set; }

    public static IEnumerable<MulModInput> GetInputs() => Inputs;

    /// <summary>
    /// In-place multiply-modulo; great for low-word-heavy operands (6.88 ns) and tiny pairs (7.15 ns) but notably slower on
    /// large moduli (193–201 ns).
    /// </summary>
    /// <remarks>
    /// Observed means: HighWordModulus 192.794 ns (1.00×), LowWordHeavy 6.883 ns, MixedMagnitude 201.468 ns, TinyOperands 7.146 ns.
    /// </remarks>
    [Benchmark(Baseline = true)]
    public GpuUInt128 InPlaceMulMod()
    {
        // TODO: Switch the extension method to allocate per iteration, as it's faster. Keep the benchmarks.
        GpuUInt128 value = Input.Left;
        value.MulMod(Input.Right, Input.Modulus);
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
    public GpuUInt128 AllocatePerIteration()
    {
        return MulModInline(Input.Left, Input.Right, Input.Modulus);
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

    public readonly record struct MulModInput(GpuUInt128 Left, GpuUInt128 Right, GpuUInt128 Modulus, string Name)
    {
        public override string ToString()
        {
            return Name;
        }
    }
}
