using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
[MemoryDiagnoser]
public class HeuristicMultiplyShiftBenchmarks
{
    private static readonly MultiplyShiftInput[] Inputs =
    [
        new(0x3FFF_FFFF_FFFF_FFFFUL, 3UL, 3, "NearOverflowShift"),
        new(0x2000_0000_0000_0000UL, 7UL, 4, "HalfRange"),
        new(0x1234_5678_9ABC_DEF0UL, 5UL, 5, "MixedBits"),
        new(0x0000_0000_1000_0000UL, 11UL, 3, "Sparse"),
    ];

    [ParamsSource(nameof(GetInputs))]
    public MultiplyShiftInput Input { get; set; }

    public static IEnumerable<MultiplyShiftInput> GetInputs()
    {
        return Inputs;
    }

    [Benchmark(Baseline = true)]
    public ulong SafeMultiplyShift()
    {
        return ULongExtensions.MultiplyShiftRight(Input.Value, Input.Multiplier, Input.Shift);
    }

    [Benchmark]
    public ulong ShiftFirstMultiplyShift()
    {
        return ULongExtensions.MultiplyShiftRightShiftFirst(Input.Value, Input.Multiplier, Input.Shift);
    }

    [Benchmark]
    public ulong NaiveMultiplyShift()
    {
        return (Input.Value * Input.Multiplier) >> Input.Shift;
    }

    public readonly record struct MultiplyShiftInput(ulong Value, ulong Multiplier, int Shift, string Name)
    {
        public override string ToString()
        {
            return Name;
        }
    }
}
