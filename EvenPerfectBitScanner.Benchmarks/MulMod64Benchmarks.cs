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

    [Benchmark(Baseline = true)]
    public ulong ExtensionBaseline()
    {
        return Input.Left.MulMod64(Input.Right, Input.Modulus);
    }

    [Benchmark]
    public ulong InlineUInt128Operands()
    {
        return MulMod64Inline(Input.Left, Input.Right, Input.Modulus);
    }

    [Benchmark]
    public ulong InlineUInt128OperandsWithOperandReduction()
    {
        return MulMod64InlineWithOperandReduction(Input.Left, Input.Right, Input.Modulus);
    }

    [Benchmark]
    public ulong InlineUInt128WithLocals()
    {
        return MulMod64InlineWithLocals(Input.Left, Input.Right, Input.Modulus);
    }

    [Benchmark]
    public ulong InlineUInt128WithLocalsAndOperandReduction()
    {
        return MulMod64InlineWithLocalsAndOperandReduction(Input.Left, Input.Right, Input.Modulus);
    }

    [Benchmark]
    public ulong MultiplyHighDecomposition()
    {
        return MulMod64MulHigh(Input.Left, Input.Right, Input.Modulus);
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

