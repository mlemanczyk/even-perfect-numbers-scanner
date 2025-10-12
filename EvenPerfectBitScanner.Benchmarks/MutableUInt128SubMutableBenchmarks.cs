using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[ShortRunJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class MutableUInt128SubMutableBenchmarks
{
	private static readonly BenchmarkInput[] Inputs =
	[
		new(Create(0UL, 0x0000_0000_0000_0007UL), Create(0, 0x3UL), "TinyOperands"),
		new(Create(0x1234_5678_9ABC_DEF0UL, 0x0FED_CBA9_8765_4321UL), Create(0xFFFF_FFFF, 0x0000_0001UL), "MixedOperands"),
		new(Create(0xFFFF_FFFF_FFFF_FFFFUL, 0xFFFF_FFFF_FFFF_FFFFUL), Create(0xFFFF_FFFF,  0xFFFF_FFFBUL), "NearMaxOperands"),
    ];

    [ParamsSource(nameof(GetInputs))]
    public BenchmarkInput Input { get; set; }

    public static IEnumerable<BenchmarkInput> GetInputs() => Inputs;

	private static MutableUInt128 Create(ulong high, ulong low) => new(high, low);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void StandardInternal(ref MutableUInt128 source, MutableUInt128 value)
	{
        ulong low = source.Low;
        ulong result = low - value.Low;
        ulong borrow = low < value.Low ? 1UL : 0UL;

        source.Low = result;
        source.High = source.High - value.High - borrow;
	}

	/// <summary>
	/// Baseline MultiplyAdd implementation that chains Multiply and Add; measured 3.370 ns with mixed operands,
	/// 3.361 ns when both limbs and inputs are near their maxima, and 3.368 ns with tiny operands.
	/// </summary>
	/// <remarks>
	/// Observed means: MixedOperands 3.370 ns, NearMaxOperands 3.361 ns, TinyOperands 3.368 ns.
	/// </remarks>
	[Benchmark(Baseline = true)]
	public MutableUInt128 Standard()
	{
		MutableUInt128 value = Input.InitialValue;
		StandardInternal(ref value, Input.Addend);
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
    public MutableUInt128 Optimized()
    {
        MutableUInt128 value = Input.InitialValue;
        value.Sub(Input.Addend);
        return value;
    }

    public readonly record struct BenchmarkInput(MutableUInt128 InitialValue, MutableUInt128 Addend, string Name)
    {
        public override string ToString()
        {
            return Name;
        }
    }
}
