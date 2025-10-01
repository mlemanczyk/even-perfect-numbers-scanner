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

    /// <summary>
    /// Performs the modulo immediately after loading the left operand; runs in 7.28 ns on tiny inputs and 38–58 ns on dense or
    /// mixed workloads, keeping it well ahead of the deferred path.
    /// </summary>
    /// <remarks>
    /// Observed means: DenseOperands 38.127 ns (1.00×), MixedMagnitude 58.555 ns, NearModulus 54.691 ns, TinyOperands 7.278 ns.
    /// </remarks>
    [Benchmark(Baseline = true)]
    public ulong ImmediateModulo()
    {
        GpuUInt128 state = new(Input.Left % Input.Modulus);
        return state.MulMod(Input.Right, Input.Modulus);
    }

    /// <summary>
    /// Defers the modulo until after the multiply; only helps on tiny operands (18.15 ns) but is 4–8× slower on dense or
    /// near-modulus workloads at 256–299 ns.
    /// </summary>
    /// <remarks>
    /// Observed means: DenseOperands 298.926 ns (7.83×), MixedMagnitude 255.624 ns, NearModulus 293.280 ns, TinyOperands 18.149 ns.
    /// </remarks>
    [Benchmark]
    public ulong DeferredNativeModulo()
    {
        // TODO: Drop MulModWithNativeModulo from production once all callers use the immediate
        // reduction path; benchmarks show it is 4–8× slower on large 128-bit operands.
        GpuUInt128 state = new(Input.Left);
        return state.MulModWithNativeModulo(Input.Right, Input.Modulus);
    }

    public readonly record struct NativeModuloInput(ulong Left, ulong Right, ulong Modulus, string Name)
    {
        public override string ToString() => Name;
    }
}
