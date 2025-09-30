using System.Collections.Generic;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class GpuUInt128NativeModuloBenchmarks
{
    private static readonly NativeModuloInput[] Inputs = new NativeModuloInput[]
    {
        new(0x0000_0000_0000_0003UL, 0x0000_0000_0000_0005UL, 0x0000_0000_0000_007FUL, "TinyOperands"),
        new(0x7FFF_FFFF_FFFF_FFC3UL, 0x7FFF_FFFF_FFFF_FF21UL, 0x7FFF_FFFF_FFFF_FFC5UL, "NearModulus"),
        new(0xFFFF_FFFF_FFFF_FFFBUL, 0xFFFF_FFFF_FFFF_FFCFUL, 0xFFFF_FFFF_FFFF_FFF5UL, "DenseOperands"),
        new(0x0123_4567_89AB_CDEFUL, 0x0FED_CBA9_8765_4321UL, 0x1FFF_FFFF_FFFF_FFFBUL, "MixedMagnitude"),
    };

    [ParamsSource(nameof(GetInputs))]
    public NativeModuloInput Input { get; set; }

    public static IEnumerable<NativeModuloInput> GetInputs() => Inputs;

    [Benchmark(Baseline = true)]
    public ulong ImmediateModulo()
    {
        GpuUInt128 state = new(Input.Left % Input.Modulus);
        return state.MulMod(Input.Right, Input.Modulus);
    }

    [Benchmark]
    public ulong DeferredNativeModulo()
    {
        GpuUInt128 state = new(Input.Left);
        return state.MulModWithNativeModulo(Input.Right, Input.Modulus);
    }

    public readonly record struct NativeModuloInput(ulong Left, ulong Right, ulong Modulus, string Name)
    {
        public override string ToString() => Name;
    }
}
