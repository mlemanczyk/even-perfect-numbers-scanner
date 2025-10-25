using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core.Gpu;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class GpuUInt128Montgomery64Benchmarks
{
    private static readonly MontgomeryInput[] Inputs = new MontgomeryInput[]
    {
        Create(0x0000_0000_0000_0065UL, 0x0000_0000_0000_0037UL, 0x0000_0000_0000_007FUL, "TinyOperands"),
        Create(0x7FFF_FFFF_FFFF_FFABUL, 0x7FFF_FFFF_FFFF_FFC5UL, 0xFFFF_FFFF_FFFF_FFC5UL, "NearModulus"),
        Create(0x0123_4567_89AB_CDEFUL, 0x0FED_CBA9_8765_4321UL, 0xFFFF_FFFF_FFFF_FF4FUL, "MixedMagnitude"),
        Create(0xFFFF_FFFF_FFFF_FF01UL, 0xFFFF_FFFF_FFFF_FDCBUL, 0xFFFF_FFFF_FFFF_FFCFUL, "DenseOperands"),
    };

    [ParamsSource(nameof(GetInputs))]
    public MontgomeryInput Input { get; set; }

    public static IEnumerable<MontgomeryInput> GetInputs() => Inputs;

    /// <summary>
    /// Extension-based Montgomery multiply that ran in 3.64 ns across every operand mix (dense, mixed, near-modulus, tiny),
    /// making it the reference CPU helper.
    /// </summary>
    /// <remarks>
    /// Observed means: DenseOperands 3.663 ns (1.00×), MixedMagnitude 3.636 ns, NearModulus 3.651 ns, TinyOperands 3.639 ns.
    /// </remarks>
    [Benchmark(Baseline = true)]
    public ulong ExtensionMontgomeryMultiply()
    {
        return Input.Left.MontgomeryMultiply(Input.Right, Input.Modulus, Input.NPrime);
    }

    /// <summary>
    /// Struct-based GPU-style multiply; matches device arithmetic but costs 24.9 ns regardless of the operand pattern,
    /// approximately 6.8× slower than the extension helper.
    /// </summary>
    /// <remarks>
    /// Observed means: DenseOperands 24.904 ns (6.80×), MixedMagnitude 24.886 ns, NearModulus 24.893 ns, TinyOperands 24.869 ns.
    /// </remarks>
    [Benchmark]
    public ulong GpuStructMontgomeryMultiply()
    {
        // TODO: Migrate remaining callers of MulModMontgomery64 to the extension helper; the struct
        // emulation is ~6.8× slower but kept for GPU parity benchmarks.
        GpuUInt128 state = new(Input.Left);
        return state.MulModMontgomery64(Input.Right, Input.Modulus, Input.NPrime, Input.R2);
    }

    private static MontgomeryInput Create(ulong left, ulong right, ulong modulus, string name)
    {
        ulong adjustedModulus = (modulus & 1UL) == 0UL ? modulus | 1UL : modulus;
        ulong nPrime = ComputeMontgomeryNPrime(adjustedModulus);
        ulong r2 = ComputeMontgomeryR2(adjustedModulus);
        return new MontgomeryInput(left, right, adjustedModulus, nPrime, r2, name);
    }

    private static ulong ComputeMontgomeryR2(ulong modulus)
    {
        UInt128 r = UInt128.One << 64;
        return (ulong)((r * r) % modulus);
    }

    private static ulong ComputeMontgomeryNPrime(ulong modulus)
    {
        ulong inv = modulus;
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        return unchecked(0UL - inv);
    }

    public readonly record struct MontgomeryInput(
        ulong Left,
        ulong Right,
        ulong Modulus,
        ulong NPrime,
        ulong R2,
        string Name)
    {
        public override string ToString() => Name;
    }
}
