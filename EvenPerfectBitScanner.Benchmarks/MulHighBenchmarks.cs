using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class MulHighBenchmarks
{
    private static readonly MulHighInput[] Inputs =
    [
        new(0x0000000100000000UL, 0x0000000100000000UL, "LowHighWord"),
        new(0x1234_5678_9ABC_DEF0UL, 0x0FED_CBA9_8765_4321UL, "Mixed"),
        new(0xFFFF_FFFF_FFFF_FFFFUL, 0xFFFF_FFFF_FFFF_FFFFUL, "AllBitsSet"),
        new(0x8000_0000_0000_0000UL, 0x8000_0000_0000_0000UL, "HighPowersOfTwo"),
        new(0x0000_0000_0001_0000UL, 0xFFFF_FFFF_0000_0000UL, "ShiftHeavy"),
        new(0x0000_0000_0000_0000UL, 0xFFFF_FFFF_FFFF_FFFFUL, "IncludesZero"),
    ];

    [ParamsSource(nameof(GetInputs))]
    public MulHighInput Input { get; set; }

    public static IEnumerable<MulHighInput> GetInputs() => Inputs;

    /// <summary>
    /// Baseline MulHigh implementation used across the scanner.
    /// Mean runtimes (ns) by input: AllBitsSet 0.868, HighPowersOfTwo 1.133,
    /// IncludesZero 0.820, LowHighWord 0.770, Mixed 1.325, ShiftHeavy 0.960.
    /// Remains competitive on mixed or high-shift workloads but is consistently
    /// slower than UInt128BuiltIn on both small (LowHighWord) and large inputs.
    /// </summary>
    [Benchmark(Baseline = true)]
    public ulong CurrentImplementation()
    {
        return ULongExtensions.MulHigh(Input.X, Input.Y);
    }

    /// <summary>
    /// Uses UInt128 to compute the high word directly.
    /// Mean runtimes (ns) by input: AllBitsSet 0.350, HighPowersOfTwo 0.338,
    /// IncludesZero 0.333, LowHighWord 0.348, Mixed 0.340, ShiftHeavy 0.378.
    /// Fastest variant across every dataset, with especially large wins on
    /// mixed and shift-heavy cases while staying under 0.4 ns for all inputs.
    /// </summary>
    [Benchmark]
    public ulong UInt128BuiltIn()
    {
        return MulHighWithUInt128(Input.X, Input.Y);
    }

    /// <summary>
    /// Emulates UInt128 multiplication using the GPU-friendly GpuUInt128 type.
    /// Mean runtimes (ns) by input: AllBitsSet 2.904, HighPowersOfTwo 2.537,
    /// IncludesZero 2.468, LowHighWord 1.970, Mixed 1.728, ShiftHeavy 2.548.
    /// Delivers functional parity with the built-in routine for GPU kernels,
    /// but trails both the CPU UInt128 path and the current baseline on every
    /// dataset, with its smallest gaps on mixed or low/high word patterns.
    /// </summary>
    [Benchmark]
    public ulong GpuUInt128Style()
    {
        return MulHighWithGpuUInt128(Input.X, Input.Y);
    }

    /// <summary>
    /// Reduces trailing zeros before multiplying smaller operands.
    /// Mean runtimes (ns) by input: AllBitsSet 0.856, HighPowersOfTwo 0.862,
    /// IncludesZero 1.122, LowHighWord 1.056, Mixed 1.692, ShiftHeavy 1.180.
    /// Performs best when both operands share large power-of-two factors, but
    /// incurs noticeable overhead on small/mixed values where reduction work
    /// dominates the final multiply.
    /// </summary>
    [Benchmark]
    public ulong ShiftThenMultiply()
    {
        return MulHighWithTrailingZeroReduction(Input.X, Input.Y);
    }

    /// <summary>
    /// Schoolbook multiply splitting inputs into 32-bit halves.
    /// Mean runtimes (ns) by input: AllBitsSet 0.940, HighPowersOfTwo 0.809,
    /// IncludesZero 1.237, LowHighWord 0.858, Mixed 0.893, ShiftHeavy 0.879.
    /// Strong option for high-power-of-two and shift-heavy inputs, but loses
    /// ground on zero-heavy workloads where carry handling amplifies cost.
    /// </summary>
    [Benchmark]
    public ulong SchoolbookVariant()
    {
        return MulHighSchoolbook(Input.X, Input.Y);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MulHighWithUInt128(ulong x, ulong y)
    {
        return (ulong)(((UInt128)x * y) >> 64);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MulHighWithGpuUInt128(ulong x, ulong y)
    {
        return ULongExtensions.MulHighGpuCompatible(x, y);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MulHighWithTrailingZeroReduction(ulong x, ulong y)
    {
        if ((x | y) == 0UL)
        {
            return 0UL;
        }

        int shiftX = BitOperations.TrailingZeroCount(x);
        int shiftY = BitOperations.TrailingZeroCount(y);
        x >>= shiftX;
        y >>= shiftY;

        UInt128 product = (UInt128)x * y;
        int totalShift = shiftX + shiftY;

        if (totalShift >= 64)
        {
            return (ulong)(product << (totalShift - 64));
        }

        return (ulong)(product >> (64 - totalShift));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MulHighSchoolbook(ulong x, ulong y)
    {
        ulong xLow = (uint)x;
        ulong xHigh = x >> 32;
        ulong yLow = (uint)y;
        ulong yHigh = y >> 32;

        ulong w1 = xLow * yHigh;
        ulong w2 = xHigh * yLow;
        ulong w3 = xLow * yLow;

        ulong result = (xHigh * yHigh) + (w1 >> 32) + (w2 >> 32);
        result += ((w3 >> 32) + (uint)w1 + (uint)w2) >> 32;
        return result;
    }

    public readonly record struct MulHighInput(ulong X, ulong Y, string Name)
    {
        public override string ToString()
        {
            return Name;
        }
    }
}

