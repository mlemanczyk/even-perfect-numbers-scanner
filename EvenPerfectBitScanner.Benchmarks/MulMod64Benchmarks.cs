using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;

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
    /// Calls the UInt128-based <see cref="ULongExtensions.MulMod64(ulong, ulong, ulong)"/> helper; every input landed in the
    /// 3.56–3.61 ns window, keeping this baseline consistently ahead of the other CPU implementations.
    /// </summary>
    /// <remarks>
    /// Observed means: CrossWordBlend 3.606 ns (1.00×), MixedBitPattern 3.564 ns, NearFullRange 3.583 ns,
    /// PrimeSizedModulus 3.593 ns, SparseOperands 3.555 ns, ZeroOperands 3.599 ns.
    /// </remarks>
    [Benchmark(Baseline = true)]
    public ulong ExtensionBaseline()
    {
        return Input.Left.MulMod64(Input.Right, Input.Modulus);
    }

    /// <summary>
    /// Calls the UInt128-based <see cref="ULongExtensions.MulMod(ulong, ulong, ulong)"/> helper; sparse operands (NearFullRange
    /// 5.71 ns, SparseOperands 5.67 ns) and zeros (1.98 ns) benefit, but dense patterns cost 22.9–64.8 ns.
    /// </summary>
    /// <remarks>
    /// Observed means: CrossWordBlend 22.870 ns (6.34×), MixedBitPattern 64.800 ns, NearFullRange 5.708 ns,
    /// PrimeSizedModulus 52.766 ns, SparseOperands 5.671 ns, ZeroOperands 1.976 ns.
    /// </remarks>
    [Benchmark]
    public ulong GpuCompatibleMulModExtension()
    {
        GpuUInt128 gpuUInt128 = new(Input.Left);
        return gpuUInt128.MulMod(Input.Right, Input.Modulus);
    }

    /// <summary>
    /// Calls the UInt128-based <see cref="ULongExtensions.MulModSimplified(ulong, ulong, ulong)"/> helper; performance mirrors
    /// the full helper with 5.24–5.95 ns on sparse inputs and 1.99 ns when zeros dominate, but 22.8–59.2 ns on dense blends.
    /// </summary>
    /// <remarks>
    /// Observed means: CrossWordBlend 22.780 ns (6.32×), MixedBitPattern 59.165 ns, NearFullRange 5.952 ns,
    /// PrimeSizedModulus 50.490 ns, SparseOperands 5.239 ns, ZeroOperands 1.986 ns.
    /// </remarks>
    [Benchmark]
    public ulong GpuCompatibleMulModSimplifiedExtension()
    {
        GpuUInt128 gpuUInt128 = new(Input.Left);
        return gpuUInt128.MulModSimplified(Input.Right, Input.Modulus);
    }

    /// <summary>
    /// Uses manual UInt128 multiplication inside the benchmark; dense operands land between 22.2 ns and 31.4 ns, while sparse
    /// or zero inputs stay near the 3.54–3.56 ns baseline.
    /// </summary>
    /// <remarks>
    /// Observed means: CrossWordBlend 22.218 ns (6.16×), MixedBitPattern 30.539 ns, NearFullRange 27.159 ns,
    /// PrimeSizedModulus 31.409 ns, SparseOperands 3.563 ns, ZeroOperands 3.542 ns.
    /// </remarks>
    [Benchmark]
    public ulong InlineUInt128Operands()
    {
        return MulMod64Inline(Input.Left, Input.Right, Input.Modulus);
    }

    /// <summary>
    /// Reduces both operands before the inline UInt128 multiply; reductions shine on NearFullRange (4.36 ns) and sparse inputs
    /// (4.62 ns) but dense mixes still take 22.6–32.2 ns.
    /// </summary>
    /// <remarks>
    /// Observed means: CrossWordBlend 22.626 ns (6.27×), MixedBitPattern 31.237 ns, NearFullRange 4.357 ns,
    /// PrimeSizedModulus 32.176 ns, SparseOperands 4.617 ns, ZeroOperands 4.286 ns.
    /// </remarks>
    [Benchmark]
    public ulong InlineUInt128OperandsWithReductionFirst()
    {
        return MulMod64InlineWithReductionFirst(Input.Left, Input.Right, Input.Modulus);
    }

    /// <summary>
    /// Reduces each operand individually before multiplication; behaves like the reduction-first path with 4.28–4.32 ns on
    /// sparse and NearFullRange inputs, yet still spends 22.5–31.8 ns on dense patterns.
    /// </summary>
    /// <remarks>
    /// Observed means: CrossWordBlend 22.548 ns (6.25×), MixedBitPattern 30.210 ns, NearFullRange 4.319 ns,
    /// PrimeSizedModulus 31.786 ns, SparseOperands 4.282 ns, ZeroOperands 4.256 ns.
    /// </remarks>
    [Benchmark]
    public ulong InlineUInt128OperandsWithOperandReduction()
    {
        return MulMod64InlineWithOperandReduction(Input.Left, Input.Right, Input.Modulus);
    }

    /// <summary>
    /// Splits the UInt128 product across locals; dense workloads cost 22.2–31.4 ns, while SparseOperands and ZeroOperands stay
    /// near 3.57–3.59 ns.
    /// </summary>
    /// <remarks>
    /// Observed means: CrossWordBlend 22.226 ns (6.16×), MixedBitPattern 30.882 ns, NearFullRange 27.137 ns,
    /// PrimeSizedModulus 31.351 ns, SparseOperands 3.566 ns, ZeroOperands 3.586 ns.
    /// </remarks>
    [Benchmark]
    public ulong InlineUInt128WithLocals()
    {
        return MulMod64InlineWithLocals(Input.Left, Input.Right, Input.Modulus);
    }

    /// <summary>
    /// Combines locals with operand reduction; reductions help sparsity (11.86–15.10 ns) but dense cases inflate to 29.7–40.5 ns
    /// making it the slowest inline variant.
    /// </summary>
    /// <remarks>
    /// Observed means: CrossWordBlend 29.746 ns (8.24×), MixedBitPattern 38.013 ns, NearFullRange 12.187 ns,
    /// PrimeSizedModulus 40.459 ns, SparseOperands 15.100 ns, ZeroOperands 11.863 ns.
    /// </remarks>
    [Benchmark]
    public ulong InlineUInt128WithLocalsAndOperandReduction()
    {
        return MulMod64InlineWithLocalsAndOperandReduction(Input.Left, Input.Right, Input.Modulus);
    }

    /// <summary>
    /// Reconstructs the 128-bit product from MulHigh/MulLow pieces; dense blends remain costly at 23.7–32.0 ns but sparse and
    /// NearFullRange inputs land around 5.20–5.55 ns.
    /// </summary>
    /// <remarks>
    /// Observed means: CrossWordBlend 23.685 ns (6.57×), MixedBitPattern 30.114 ns, NearFullRange 5.221 ns,
    /// PrimeSizedModulus 31.950 ns, SparseOperands 5.204 ns, ZeroOperands 5.171 ns.
    /// </remarks>
    [Benchmark]
    public ulong MultiplyHighDecomposition()
    {
        return MulMod64MulHigh(Input.Left, Input.Right, Input.Modulus);
    }

    /// <summary>
    /// Uses the GPU-friendly <see cref="ULongExtensions.MulMod64GpuCompatible(ulong, ulong, ulong)"/> helper; dense inputs now
    /// cost 24.4–61.9 ns, while NearFullRange and SparseOperands finish near 7.27 ns and ZeroOperands stay baseline-fast at
    /// 3.59 ns.
    /// </summary>
    /// <remarks>
    /// Observed means: CrossWordBlend 24.422 ns (6.86×), MixedBitPattern 61.850 ns, NearFullRange 7.267 ns,
    /// PrimeSizedModulus 55.114 ns, SparseOperands 7.261 ns, ZeroOperands 3.587 ns.
    /// </remarks>
    [Benchmark]
    public ulong GpuCompatibleBaseline()
    {
        // TODO: Callers in production should migrate to ULongExtensions.MulMod64 where GPU parity
        // is not required; that baseline stayed 6–17× faster for dense 64-bit operands.
        return Input.Left.MulMod64GpuCompatible(Input.Right, Input.Modulus);
    }

    /// <summary>
    /// Uses the deferred native-modulo helper <see cref="ULongExtensions.MulMod64GpuCompatibleDeferred(ulong, ulong, ulong)"/>;
    /// excels when operands are tiny (2.01 ns on ZeroOperands) yet remains the slowest choice on dense data at 146–295 ns.
    /// </summary>
    /// <remarks>
    /// Observed means: CrossWordBlend 146.559 ns (40.63×), MixedBitPattern 255.555 ns, NearFullRange 21.390 ns,
    /// PrimeSizedModulus 294.486 ns, SparseOperands 18.182 ns, ZeroOperands 2.009 ns.
    /// </remarks>
    [Benchmark]
    public ulong GpuCompatibleDeferred()
    {
        // TODO: Retire the deferred GPU shim from runtime paths and keep it only for benchmarks;
        // ULongExtensions.MulMod64 avoids the 6×–82× slowdown on real-world operand mixes.
        return Input.Left.MulMod64GpuCompatibleDeferred(Input.Right, Input.Modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MulMod64Inline(ulong a, ulong b, ulong modulus) => (ulong)(((UInt128)a * b) % modulus);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MulMod64InlineWithReductionFirst(ulong a, ulong b, ulong modulus) => (ulong)(((UInt128)(a % modulus) * (b % modulus)) % modulus);

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

