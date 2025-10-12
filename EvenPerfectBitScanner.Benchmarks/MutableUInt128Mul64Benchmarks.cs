using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[ShortRunJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class MutableUInt128Mul64Benchmarks
{
	private static readonly BenchmarkInput[] Inputs =
	[
		new(Create(0UL, 0x0000_0000_0000_0007UL), 0x3UL, "TinyOperands"),
		new(Create(0x1234_5678_9ABC_DEF0UL, 0x0FED_CBA9_8765_4321UL), 0xFFFF_FFFF_0000_0001UL, "MixedOperands"),
		new(Create(0xFFFF_FFFF_FFFF_FFFFUL, 0xFFFF_FFFF_FFFF_FFFFUL), 0xFFFF_FFFF_FFFF_FFFBUL, "NearMaxOperands"),
    ];

    [ParamsSource(nameof(GetInputs))]
    public BenchmarkInput Input { get; set; }

    public static IEnumerable<BenchmarkInput> GetInputs() => Inputs;

	private static MutableUInt128 Create(ulong high, ulong low) => new(high, low);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Mul64(ulong left, ulong right, out ulong low)
    {
        const ulong Mask32 = 0xFFFF_FFFFUL;

        ulong leftLow = left & Mask32;
        ulong leftHigh = left >> 32;
        ulong rightLow = right & Mask32;
        ulong rightHigh = right >> 32;

        ulong lowLow = leftLow * rightLow;
        ulong lowHigh = leftLow * rightHigh;
        ulong highLow = leftHigh * rightLow;
        ulong highHigh = leftHigh * rightHigh;

        ulong carry = (lowLow >> 32) + (lowHigh & Mask32) + (highLow & Mask32);
        ulong high = highHigh + (lowHigh >> 32) + (highLow >> 32) + (carry >> 32);

        low = ((carry & Mask32) << 32) | (lowLow & Mask32);
        return high;
    }

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void StandardInternal(ref MutableUInt128 source, ulong value)
	{
        ulong currentHigh = source.High;
        ulong currentLow = source.Low;

        ulong carry = Mul64(currentLow, value, out ulong newLow);
        Mul64(currentHigh, value, out ulong shiftedHigh);
        ulong newHigh = shiftedHigh + carry;

        source.Low = newLow;
        source.High = newHigh;
	}

	/// <summary>
	/// Baseline MultiplyAdd implementation that chains Multiply and Add; measured 3.113 ns with mixed operands,
	/// 3.128 ns when both limbs and inputs are near their maxima, and 3.131 ns with tiny operands.
	/// </summary>
	/// <remarks>
	/// Observed means: MixedOperands 3.113 ns, NearMaxOperands 3.128 ns, TinyOperands 3.131 ns.
	/// </remarks>
	[Benchmark(Baseline = true)]
	public MutableUInt128 Standard()
	{
		MutableUInt128 value = Input.InitialValue;
		StandardInternal(ref value, Input.Addend);
		return value;
	}

    /// <summary>
    /// Fully inlined MultiplyAdd implementation that kept all arithmetic inside the method; landed at 2.969 ns on mixed operands,
    /// 2.912 ns on near-max inputs, and 2.935 ns on tiny operands.
    /// </summary>
    /// <remarks>
    /// Observed means: MixedOperands 2.969 ns, NearMaxOperands 2.912 ns, TinyOperands 2.935 ns.
	/// Status: Implemented
    /// </remarks>
    [Benchmark]
    public MutableUInt128 Optimized()
    {
        MutableUInt128 value = Input.InitialValue;
        value.Mul(Input.Addend);
        return value;
    }

    public readonly record struct BenchmarkInput(MutableUInt128 InitialValue, ulong Addend, string Name)
    {
        public override string ToString()
        {
            return Name;
        }
    }
}
