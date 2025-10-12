using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void MultiplyAdd(ref MutableUInt128 source, ulong multiplier, ulong addend)
	{
		source.Mul(multiplier);
		source.Add(addend);
	}

	/// <summary>
	/// Baseline MultiplyAdd implementation that chains Multiply and Add; measured 3.370 ns with mixed operands,
	/// 3.361 ns when both limbs and inputs are near their maxima, and 3.368 ns with tiny operands.
	/// </summary>
	/// <remarks>
	/// Observed means: MixedOperands 3.370 ns, NearMaxOperands 3.361 ns, TinyOperands 3.368 ns.
	/// </remarks>
	[Benchmark(Baseline = true)]
	public MutableUInt128 MultiplyAddChained()
	{
		MutableUInt128 value = new(Input.InitialValue);
		MultiplyAdd(ref value, Input.Multiplier, Input.Addend);
		return value;
	}

    /// <summary>
    /// Fully inlined MultiplyAdd implementation that kept all arithmetic inside the method; landed at 3.175 ns on mixed operands,
    /// 3.177 ns on near-max inputs, and 3.178 ns on tiny operands.
    /// </summary>
    /// <remarks>
    /// Observed means: MixedOperands 3.175 ns, NearMaxOperands 3.177 ns, TinyOperands 3.178 ns.
	/// Status: Implemented
    /// </remarks>
    [Benchmark]
    public MutableUInt128 MultiplyAddFullyInlined()
    {
        MutableUInt128 value = new(Input.InitialValue);
        value.MulAdd(Input.Multiplier, Input.Addend);
        return value;
    }

    public readonly record struct MultiplyAddInput(UInt128 InitialValue, ulong Multiplier, ulong Addend, string Name)
    {
        public override string ToString()
        {
            return Name;
        }
    }
}
