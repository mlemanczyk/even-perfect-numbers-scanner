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
    /// Baseline MulHigh implementation used across the scanner; measured 0.73–0.78 ns with minimal variance between operand
    /// mixes, yet still slower than the UInt128 intrinsic.
    /// </summary>
    /// <remarks>
    /// Observed means: AllBitsSet 0.7577 ns, HighPowersOfTwo 0.7352 ns, IncludesZero 0.7497 ns,
    /// LowHighWord 0.7296 ns, Mixed 0.7365 ns, ShiftHeavy 0.7747 ns.
    /// </remarks>
    [Benchmark(Baseline = true)]
    public ulong CurrentImplementation()
    {
        // Routes through the UInt128-based helper adopted by CPU hot paths after benchmarking.
        return ULongExtensions.MulHighCpu(Input.X, Input.Y);
    }

    /// <summary>
    /// Uses UInt128 to compute the high word directly; fastest option at 0.054–0.067 ns across all mixes, beating the baseline
    /// by roughly 11–14×.
    /// </summary>
    /// <remarks>
    /// Observed means: AllBitsSet 0.0641 ns, HighPowersOfTwo 0.0674 ns, IncludesZero 0.0541 ns,
    /// LowHighWord 0.0576 ns, Mixed 0.0634 ns, ShiftHeavy 0.0591 ns.
    /// </remarks>
    [Benchmark]
    public ulong UInt128BuiltIn()
    {
        return MulHighWithUInt128(Input.X, Input.Y);
    }

    /// <summary>
    /// Emulates UInt128 multiplication using the GPU-friendly <see cref="GpuUInt128"/> type; lands between 0.854 ns and 0.869 ns,
    /// which is ~12–17% slower than the baseline but mirrors GPU arithmetic exactly.
    /// </summary>
    /// <remarks>
    /// Observed means: AllBitsSet 0.8643 ns, HighPowersOfTwo 0.8584 ns, IncludesZero 0.8538 ns,
    /// LowHighWord 0.8576 ns, Mixed 0.8693 ns, ShiftHeavy 0.8625 ns.
    /// </remarks>
    [Benchmark]
    public ulong GpuUInt128Style()
    {
        return MulHighWithGpuUInt128(Input.X, Input.Y);
    }

    /// <summary>
    /// Reduces trailing zeros before multiplying smaller operands; excels on high-power-of-two inputs (1.142 ns) but remains
    /// 20–55% slower than the baseline elsewhere.
    /// </summary>
    /// <remarks>
    /// Observed means: AllBitsSet 0.9894 ns, HighPowersOfTwo 1.1421 ns, IncludesZero 0.9270 ns,
    /// LowHighWord 0.9195 ns, Mixed 1.2042 ns, ShiftHeavy 1.1909 ns.
    /// </remarks>
    [Benchmark]
    public ulong ShiftThenMultiply()
    {
        return MulHighWithTrailingZeroReduction(Input.X, Input.Y);
    }

    /// <summary>
    /// Schoolbook multiply splitting inputs into 32-bit halves; tracks the baseline closely at 0.739–0.748 ns with its best
    /// showing on mixed inputs.
    /// </summary>
    /// <remarks>
    /// Observed means: AllBitsSet 0.7414 ns, HighPowersOfTwo 0.7459 ns, IncludesZero 0.7442 ns,
    /// LowHighWord 0.7441 ns, Mixed 0.7395 ns, ShiftHeavy 0.7485 ns.
    /// </remarks>
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
        return ULongExtensions.MulHighGpu(x, y);
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

