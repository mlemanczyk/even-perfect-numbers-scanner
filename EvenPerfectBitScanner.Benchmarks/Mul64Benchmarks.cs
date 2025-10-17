using System.Numerics;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class Mul64Benchmarks
{
    private static readonly BigInteger ProductMask = (BigInteger.One << 128) - BigInteger.One;

    private static readonly Mul64Input[] Inputs =
    [
        new(Create(0UL, 0x0000_0000_0000_0001UL), Create(0UL, 0x0000_0000_0000_0001UL), "TinyOperands"),
        new(Create(0UL, 0xFFFF_FFFF_FFFF_FFFBUL), Create(0UL, 0xFFFF_FFFF_FFFF_FFFFUL), "LowWordMax"),
        new(Create(0x0000_0000_0000_0001UL, 0x0000_0000_0000_0000UL), Create(0x0000_0000_0000_0001UL, 0x0000_0000_0000_0000UL), "HighWordOnly"),
        new(Create(0xFFFF_FFFF_FFFF_FFFFUL, 0xFFFF_FFFF_FFFF_FFFFUL), Create(0xFFFF_FFFF_FFFF_FFFFUL, 0xFFFF_FFFF_FFFF_FFFFUL), "AllBitsSet"),
        new(Create(0x1234_5678_9ABC_DEF0UL, 0x1357_9BDF_2468_ACEEUL), Create(0x0FED_CBA9_8765_4321UL, 0x0246_8ACE_1357_9BDFUL), "InterleavedHighLow"),
        new(Create(0x0000_0000_0000_0000UL, 0x0000_0001_0000_0000UL), Create(0xFFFF_FFFF_0000_0000UL, 0x0000_0000_FFFF_FFFFUL), "ShiftHeavyMixed"),
    ];

    [ParamsSource(nameof(GetInputs))]
    public Mul64Input Input { get; set; }

    public static IEnumerable<Mul64Input> GetInputs() => Inputs;

    /// <summary>
    /// Mul64 layout that keeps the high-word accumulation in locals; measured 1.49–1.57 ns across all operand mixes, making it
    /// the fastest CPU layout we profiled.
    /// </summary>
    /// <remarks>
    /// Observed means: AllBitsSet 1.568 ns, HighWordOnly 1.495 ns, InterleavedHighLow 1.526 ns,
    /// LowWordMax 1.506 ns, ShiftHeavyMixed 1.481 ns, TinyOperands 1.503 ns.
    /// </remarks>
    [Benchmark(Baseline = true)]
    public UInt128 HighWordAccumulatedInLocals()
    {
        return Input.Left.Mul64(Input.Right);
    }

    /// <summary>
    /// Mul64 layout that folds the high-word expression into a single return value; costs roughly 3–5% extra with 1.54–1.58 ns
    /// runtimes depending on the operand mix.
    /// </summary>
    /// <remarks>
    /// Observed means: AllBitsSet 1.583 ns, HighWordOnly 1.542 ns, InterleavedHighLow 1.575 ns,
    /// LowWordMax 1.581 ns, ShiftHeavyMixed 1.562 ns, TinyOperands 1.546 ns.
    /// </remarks>
    [Benchmark]
    public UInt128 FoldedHighReturnValue()
    {
        return Mul64FoldedHighReturn(Input.Left, Input.Right);
    }

    /// <summary>
    /// Mul64 layout shaped for the GPU helper to keep the cross products in registers; roughly 2.35× slower than the baseline at
    /// 3.68–3.72 ns, but useful when mirroring GPU arithmetic on the CPU.
    /// </summary>
    /// <remarks>
    /// Observed means: AllBitsSet 3.682 ns, HighWordOnly 3.679 ns, InterleavedHighLow 3.719 ns,
    /// LowWordMax 3.684 ns, ShiftHeavyMixed 3.707 ns, TinyOperands 3.683 ns.
    /// </remarks>
    [Benchmark]
    public UInt128 GpuFriendlyLayout()
    {
        return Mul64GpuFriendly(Input.Left, Input.Right);
    }

    /// <summary>
    /// Reference implementation that multiplies with <see cref="BigInteger"/> to expose the full 256-bit product; handy for
    /// validation but 22–57× slower at 34–87 ns depending on the operand distribution.
    /// </summary>
    /// <remarks>
    /// Observed means: AllBitsSet 86.087 ns, HighWordOnly 78.860 ns, InterleavedHighLow 87.367 ns,
    /// LowWordMax 74.418 ns, ShiftHeavyMixed 81.020 ns, TinyOperands 34.286 ns.
    /// </remarks>
    [Benchmark]
    public UInt128 BigIntegerReference()
    {
        return Mul64WithBigInteger(Input.Left, Input.Right);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static UInt128 Mul64FoldedHighReturn(UInt128 left, UInt128 right)
    {
        ulong leftLow = (ulong)left;
        ulong rightLow = (ulong)right;
        return ((UInt128)(leftLow * (ulong)(right >> 64) + leftLow.MulHigh(rightLow)) << 64) | (UInt128)(leftLow * rightLow);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static UInt128 Mul64GpuFriendly(UInt128 left, UInt128 right)
    {
        GpuUInt128 gpuLeft = new((ulong)(left >> 64), (ulong)left);
        GpuUInt128 gpuRight = new((ulong)(right >> 64), (ulong)right);
        gpuLeft.Mul(gpuRight);
        return (UInt128)gpuLeft;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static UInt128 Mul64WithBigInteger(UInt128 left, UInt128 right)
    {
        BigInteger product = (BigInteger)left * (BigInteger)right;
        return (UInt128)(product & ProductMask);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static UInt128 Create(ulong high, ulong low)
    {
        return ((UInt128)high << 64) | low;
    }

    public readonly record struct Mul64Input(UInt128 Left, UInt128 Right, string Name)
    {
        public override string ToString()
        {
            return Name;
        }
    }
}

