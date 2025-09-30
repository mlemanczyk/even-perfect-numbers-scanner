using System.Collections.Generic;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class MulMod64Benchmarks
{
    private static readonly MulMod64Input[] Inputs =
    [
        new(0UL, 0UL, 3UL, "ZeroOperands"),
        new(ulong.MaxValue, ulong.MaxValue - 1UL, ulong.MaxValue - 58UL, "NearFullRange"),
        new(0xFFFF_FFFF_0000_0000UL, 0x0000_0000_FFFF_FFFFUL, 0xFFFF_FFFF_0000_002BUL, "CrossWordBlend"),
        new(0x1234_5678_9ABC_DEF0UL, 0x0FED_CBA9_8765_4321UL, 0x1FFF_FFFF_FFFF_FFFBUL, "MixedBitPattern"),
        new(0x0000_0001_0000_0000UL, 0x0000_0000_0000_0003UL, 0x0000_0000_FFFF_FFC3UL, "SparseOperands"),
        new(0x7FFF_FFFF_FFFF_FFA3UL, 0x6FFF_FFFF_FFFF_FF81UL, 0x7FFF_FFFF_FFFF_FFE7UL, "PrimeSizedModulus"),
    ];

    [ParamsSource(nameof(GetInputs))]
    public MulMod64Input Input { get; set; }

    public static IEnumerable<MulMod64Input> GetInputs()
    {
        return Inputs;
    }

    /// <summary>
    /// Calls the UInt128-based <see cref="ULongExtensions.MulMod64(ulong, ulong, ulong)"/> helper, which stayed between
    /// 3.66 ns and 3.80 ns for CrossWordBlend, MixedBitPattern, PrimeSizedModulus, SparseOperands, and ZeroOperands inputs,
    /// and 3.73 ns for NearFullRange, making it the fastest option across every distribution.
    /// </summary>
    /// <remarks>
    /// Observed means: CrossWordBlend 3.784 ns (1.00×), MixedBitPattern 3.790 ns, NearFullRange 3.727 ns, PrimeSizedModulus
    /// 3.791 ns, SparseOperands 3.795 ns, ZeroOperands 3.659 ns.
    /// </remarks>
    [Benchmark(Baseline = true)]
    public ulong ExtensionBaseline()
    {
        return Input.Left.MulMod64(Input.Right, Input.Modulus);
    }

    /// <summary>
    /// Uses manual UInt128 multiplication in the benchmark method; dense operands cost 23.6–33.6 ns (6.2–8.9× slower than
    /// baseline), while SparseOperands and ZeroOperands remain inexpensive at 3.58–3.87 ns.
    /// </summary>
    /// <remarks>
    /// Observed means: CrossWordBlend 23.611 ns, MixedBitPattern 32.894 ns, NearFullRange 28.243 ns, PrimeSizedModulus
    /// 33.576 ns, SparseOperands 3.870 ns, ZeroOperands 3.583 ns.
    /// </remarks>
    [Benchmark]
    public ulong InlineUInt128Operands()
    {
        return MulMod64Inline(Input.Left, Input.Right, Input.Modulus);
    }

    /// <summary>
    /// Reduces both operands before multiplying with inline UInt128; reductions pay off for NearFullRange (4.52 ns) and
    /// SparseOperands (4.39 ns) but dense patterns still cost 24.0–34.0 ns.
    /// </summary>
    /// <remarks>
    /// Observed means: CrossWordBlend 24.012 ns, MixedBitPattern 33.051 ns, NearFullRange 4.522 ns, PrimeSizedModulus
    /// 33.993 ns, SparseOperands 4.394 ns, ZeroOperands 4.211 ns.
    /// </remarks>
    [Benchmark]
    public ulong InlineUInt128OperandsWithOperandReduction()
    {
        return MulMod64InlineWithOperandReduction(Input.Left, Input.Right, Input.Modulus);
    }

    /// <summary>
    /// Splits the UInt128 product into locals; performance mirrors the operand-inlining variant at 23.8–32.8 ns for dense
    /// inputs while SparseOperands and ZeroOperands stay near 3.60–3.91 ns.
    /// </summary>
    /// <remarks>
    /// Observed means: CrossWordBlend 23.789 ns, MixedBitPattern 32.817 ns, NearFullRange 28.666 ns, PrimeSizedModulus
    /// 32.747 ns, SparseOperands 3.599 ns, ZeroOperands 3.906 ns.
    /// </remarks>
    [Benchmark]
    public ulong InlineUInt128WithLocals()
    {
        return MulMod64InlineWithLocals(Input.Left, Input.Right, Input.Modulus);
    }

    /// <summary>
    /// Combines locals with operand reduction; NearFullRange, SparseOperands, and ZeroOperands improve to 11.9–12.7 ns but
    /// dense cases degrade further to 32.1–42.5 ns (8.5–11.3× slower than baseline).
    /// </summary>
    /// <remarks>
    /// Observed means: CrossWordBlend 32.109 ns, MixedBitPattern 41.136 ns, NearFullRange 12.691 ns, PrimeSizedModulus
    /// 42.527 ns, SparseOperands 12.288 ns, ZeroOperands 11.917 ns.
    /// </remarks>
    [Benchmark]
    public ulong InlineUInt128WithLocalsAndOperandReduction()
    {
        return MulMod64InlineWithLocalsAndOperandReduction(Input.Left, Input.Right, Input.Modulus);
    }

    /// <summary>
    /// Reconstructs the 128-bit product from MulHigh/MulLow pieces; dense patterns run in 24.5–34.8 ns, while
    /// NearFullRange, SparseOperands, and ZeroOperands finish in 5.21–5.55 ns after the extra reduction.
    /// </summary>
    /// <remarks>
    /// Observed means: CrossWordBlend 24.502 ns, MixedBitPattern 33.045 ns, NearFullRange 5.552 ns, PrimeSizedModulus
    /// 34.793 ns, SparseOperands 5.214 ns, ZeroOperands 5.231 ns.
    /// </remarks>
    [Benchmark]
    public ulong MultiplyHighDecomposition()
    {
        return MulMod64MulHigh(Input.Left, Input.Right, Input.Modulus);
    }

    /// <summary>
    /// Uses the GPU-friendly <see cref="ULongExtensions.MulMod64GpuCompatible(ulong, ulong, ulong)"/> helper; the shift-add
    /// reduction drives dense patterns to 99.7–186.4 ns, while NearFullRange and SparseOperands finish around 40.1–40.7 ns
    /// and ZeroOperands drop to 24.2 ns.
    /// </summary>
    /// <remarks>
    /// Observed means (single-method run): CrossWordBlend 99.677 ns, MixedBitPattern 186.449 ns, NearFullRange 40.682 ns,
    /// PrimeSizedModulus 112.461 ns, SparseOperands 40.067 ns, ZeroOperands 24.223 ns.
    /// </remarks>
    [Benchmark]
    public ulong GpuCompatibleBaseline()
    {
        return Input.Left.MulMod64GpuCompatible(Input.Right, Input.Modulus);
    }

    /// <summary>
    /// Uses the native-modulo GPU helper <see cref="ULongExtensions.MulMod64GpuCompatibleDeferred(ulong, ulong, ulong)"/>;
    /// final reduction via `%` trims dense cases to 353–689 ns while NearFullRange, SparseOperands, and ZeroOperands drop to
    /// 47.98 ns, 21.79 ns, and 5.24 ns respectively (single-method run).
    /// </summary>
    /// <remarks>
    /// Observed means (single-method run): CrossWordBlend 353.778 ns, MixedBitPattern 564.491 ns, NearFullRange 47.981 ns,
    /// PrimeSizedModulus 689.126 ns, SparseOperands 21.786 ns, ZeroOperands 5.236 ns.
    /// </remarks>
    [Benchmark]
    public ulong GpuCompatibleDeferred()
    {
        return Input.Left.MulMod64GpuCompatibleDeferred(Input.Right, Input.Modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MulMod64Inline(ulong a, ulong b, ulong modulus)
    {
        return (ulong)(((UInt128)a * b) % modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MulMod64InlineWithOperandReduction(ulong a, ulong b, ulong modulus)
    {
        ulong left = a % modulus;
        ulong right = b % modulus;
        return (ulong)(((UInt128)left * right) % modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MulMod64InlineWithLocals(ulong a, ulong b, ulong modulus)
    {
        UInt128 left128 = a;
        UInt128 right128 = b;
        UInt128 product = left128 * right128;
        UInt128 reduced = product % modulus;
        return (ulong)reduced;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MulMod64InlineWithLocalsAndOperandReduction(ulong a, ulong b, ulong modulus)
    {
        UInt128 left128 = a;
        UInt128 right128 = b;
        left128 %= modulus;
        right128 %= modulus;
        UInt128 product = left128 * right128;
        UInt128 reduced = product % modulus;
        return (ulong)reduced;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MulMod64MulHigh(ulong a, ulong b, ulong modulus)
    {
        ulong left = a % modulus;
        ulong right = b % modulus;
        ulong low = left * right;
        ulong high = left.MulHigh(right);
        UInt128 product = ((UInt128)high << 64) | low;
        return (ulong)(product % modulus);
    }

    public readonly record struct MulMod64Input(ulong Left, ulong Right, ulong Modulus, string Name)
    {
        public override string ToString()
        {
            return Name;
        }
    }
}

