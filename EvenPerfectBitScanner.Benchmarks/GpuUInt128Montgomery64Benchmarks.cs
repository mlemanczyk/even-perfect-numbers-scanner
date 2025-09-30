using System;
using System.Collections.Generic;
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

    [Benchmark(Baseline = true)]
    public ulong ExtensionMontgomeryMultiply()
    {
        return Input.Left.MontgomeryMultiply(Input.Right, Input.Modulus, Input.NPrime);
    }

    [Benchmark]
    public ulong GpuStructMontgomeryMultiply()
    {
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
