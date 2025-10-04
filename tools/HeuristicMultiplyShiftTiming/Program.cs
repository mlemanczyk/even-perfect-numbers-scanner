using System.Diagnostics;
using PerfectNumbers.Core;

namespace HeuristicMultiplyShiftTiming;

internal static class Program
{
    private static readonly MultiplyShiftInput[] Inputs =
    [
        new(0x3FFF_FFFF_FFFF_FFFFUL, 3UL, 3, "NearOverflowShift"),
        new(0x2000_0000_0000_0000UL, 7UL, 4, "HalfRange"),
        new(0x1234_5678_9ABC_DEF0UL, 5UL, 5, "MixedBits"),
        new(0x0000_0000_1000_0000UL, 11UL, 3, "Sparse"),
    ];

    private const int Iterations = 200_000_000;

    public static void Main()
    {
        Console.WriteLine($"Timing each method for {Iterations:N0} iterations per input...\n");

        foreach (MultiplyShiftInput input in Inputs)
        {
            Console.WriteLine($"Input: {input.Name} (value=0x{input.Value:X16}, multiplier={input.Multiplier}, shift={input.Shift})");

            Measurement safe = MeasureSafe(input);
            Console.WriteLine($"  MultiplyShiftRight (UInt128 product): {safe.AverageNanoseconds:F3} ns | checksum=0x{safe.Checksum:X16}");

            Measurement shiftFirst = MeasureShiftFirst(input);
            Console.WriteLine($"  MultiplyShiftRightShiftFirst (shift-first): {shiftFirst.AverageNanoseconds:F3} ns | checksum=0x{shiftFirst.Checksum:X16}");

            Measurement naive = MeasureNaive(input);
            Console.WriteLine($"  Naive ((value * multiplier) >> shift): {naive.AverageNanoseconds:F3} ns | checksum=0x{naive.Checksum:X16}\n");
        }
    }

    private static Measurement MeasureSafe(MultiplyShiftInput input)
    {
        ulong checksum = 0UL;
        Stopwatch sw = Stopwatch.StartNew();
        ulong value = input.Value;

        for (int i = 0; i < Iterations; i++)
        {
            checksum += HeuristicArithmetic.MultiplyShiftRight(value, input.Multiplier, input.Shift);
            value += 1UL;
        }

        sw.Stop();
        return CreateMeasurement(sw, checksum);
    }

    private static Measurement MeasureShiftFirst(MultiplyShiftInput input)
    {
        ulong checksum = 0UL;
        Stopwatch sw = Stopwatch.StartNew();
        ulong value = input.Value;

        for (int i = 0; i < Iterations; i++)
        {
            checksum += HeuristicArithmetic.MultiplyShiftRightShiftFirst(value, input.Multiplier, input.Shift);
            value += 1UL;
        }

        sw.Stop();
        return CreateMeasurement(sw, checksum);
    }

    private static Measurement MeasureNaive(MultiplyShiftInput input)
    {
        ulong checksum = 0UL;
        Stopwatch sw = Stopwatch.StartNew();
        ulong value = input.Value;

        for (int i = 0; i < Iterations; i++)
        {
            checksum += (value * input.Multiplier) >> input.Shift;
            value += 1UL;
        }

        sw.Stop();
        return CreateMeasurement(sw, checksum);
    }

    private static Measurement CreateMeasurement(Stopwatch sw, ulong checksum)
    {
        double elapsedNanoseconds = sw.ElapsedTicks * (1_000_000_000.0 / Stopwatch.Frequency);
        double averageNanoseconds = elapsedNanoseconds / Iterations;
        return new Measurement(averageNanoseconds, checksum);
    }

    private readonly record struct MultiplyShiftInput(ulong Value, ulong Multiplier, int Shift, string Name);

    private readonly record struct Measurement(double AverageNanoseconds, ulong Checksum);
}
