using System.Collections.Generic;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class ShiftLeftByNativeChunkBenchmarks
{
    private const ulong LargePrimeModulus = 0xFFFF_FFFF_FFFF_FFC5UL;
    private const ulong MediumPrimeModulus = 0xFFFF_FFFBUL;
    private const ulong SmallPrimeModulus = 97UL;
    private const ulong NativeChunkBits = 8UL;

    private static readonly ShiftLeftInput[] Inputs = new ShiftLeftInput[]
    {
        new(0xFFFF_FFFF_FFFF_FF03UL, LargePrimeModulus, "LargeResidueLargeModulus"),
        new(0x1234_5678_9ABC_DEF0UL, MediumPrimeModulus, "MixedResidueMediumModulus"),
        new(0x0000_0000_0000_00F3UL, SmallPrimeModulus, "CompactResidueSmallModulus"),
    };

    [ParamsSource(nameof(GetInputs))]
    public ShiftLeftInput Input { get; set; }

    public static IEnumerable<ShiftLeftInput> GetInputs() => Inputs;

    /// <summary>
    /// Windowed multiply that mirrors the production helper. Serves as the baseline for the eight-step ladder replacement.
    /// </summary>
    [Benchmark(Baseline = true)]
    public ulong WindowedMultiply()
    {
        return ShiftLeftWindowed(Input.Value, Input.Modulus);
    }

    /// <summary>
    /// Legacy eight-step ladder that performed a modulo after every shift. Retained for benchmarking to quantify the regression risk.
    /// </summary>
    [Benchmark]
    public ulong EightStepLadder()
    {
        return ShiftLeftLegacy(Input.Value, Input.Modulus);
    }

    private static ulong ShiftLeftWindowed(ulong value, ulong modulus)
    {
        ulong nativeChunkMultiplier = ULongExtensions.Pow2ModWindowedCpu(NativeChunkBits, modulus);
        return ULongExtensions.MulMod64(value, nativeChunkMultiplier, modulus);
    }

    private static ulong ShiftLeftLegacy(ulong value, ulong modulus)
    {
        for (int step = 0; step < (int)NativeChunkBits; step++)
        {
            value = (value << 1) % modulus;
        }

        return value;
    }

    public readonly record struct ShiftLeftInput(ulong Value, ulong Modulus, string Name)
    {
        public override string ToString()
        {
            return Name;
        }
    }
}
