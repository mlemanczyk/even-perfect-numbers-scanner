using System.Collections.Generic;
using System.Numerics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[ShortRunJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class MutableUInt128MultiplyAddBenchmarks
{
    private static readonly MultiplyAddInput[] Inputs =
    [
        new(Create(0UL, 0x0000_0000_0000_0007UL), 0x3UL, 0x5UL, "TinyOperands"),
        new(Create(0x1234_5678_9ABC_DEF0UL, 0x0FED_CBA9_8765_4321UL), 0xFFFF_FFFF_0000_0001UL, 0x1357_9BDF_2468_ACEEUL, "MixedOperands"),
        new(Create(0xFFFF_FFFF_FFFF_FFFFUL, 0xFFFF_FFFF_FFFF_FFFFUL), 0xFFFF_FFFF_FFFF_FFFBUL, 0xFFFF_FFFF_FFFF_FFF7UL, "NearMaxOperands"),
    ];

    [ParamsSource(nameof(GetInputs))]
    public MultiplyAddInput Input { get; set; }

    public static IEnumerable<MultiplyAddInput> GetInputs() => Inputs;

    private static UInt128 Create(ulong high, ulong low)
    {
        return ((UInt128)high << 64) | low;
    }

    /// <summary>
    /// Baseline MultiplyAdd implementation that chains Multiply and Add; measured 4.59 ns with mixed operands,
    /// 4.68 ns when both limbs and inputs are near their maxima, and 4.49 ns with tiny operands.
    /// </summary>
    /// <remarks>
    /// Observed means: MixedOperands 4.593 ns, NearMaxOperands 4.681 ns, TinyOperands 4.487 ns.
    /// </remarks>
    [Benchmark(Baseline = true)]
    public UInt128 MultiplyAddChained()
    {
        MutableUInt128 value = new(Input.InitialValue);
        value.MultiplyAdd(Input.Multiplier, Input.Addend);
        return value.ToUInt128();
    }

    /// <summary>
    /// Fully inlined MultiplyAdd implementation that kept all arithmetic inside the method; landed at 4.11 ns on mixed operands,
    /// 4.25 ns on near-max inputs, and 4.27 ns on tiny operands.
    /// </summary>
    /// <remarks>
    /// Observed means: MixedOperands 4.109 ns, NearMaxOperands 4.253 ns, TinyOperands 4.271 ns.
    /// </remarks>
    [Benchmark]
    public UInt128 MultiplyAddFullyInlined()
    {
        MutableUInt128 value = new(Input.InitialValue);
        value.MultiplyAddInline(Input.Multiplier, Input.Addend);
        return value.ToUInt128();
    }

    public readonly record struct MultiplyAddInput(UInt128 InitialValue, ulong Multiplier, ulong Addend, string Name)
    {
        public override string ToString()
        {
            return Name;
        }
    }
}
