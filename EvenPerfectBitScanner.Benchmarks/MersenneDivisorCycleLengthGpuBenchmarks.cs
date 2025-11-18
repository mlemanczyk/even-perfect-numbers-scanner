using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace EvenPerfectBitScanner.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 3)]
[MemoryDiagnoser]
public class MersenneDivisorCycleLengthGpuBenchmarks
{
    private const int Iterations = 64;
    private static readonly ulong[] StepOffsets = [0UL, 2UL, 4UL, 6UL, 8UL, 10UL, 12UL, 14UL];
    private const int StepMask = 7;

    private static readonly ulong[] Divisors =
    [
        17UL,
        31UL,
        8191UL,
        131071UL,
        524287UL,
        2097151UL,
        8388607UL,
    ];

    [ParamsSource(nameof(GetDivisors))]
    public ulong Divisor { get; set; }

    public IEnumerable<ulong> GetDivisors() => Divisors;

    /// <summary>
    /// Baseline single-step GPU loop; spans from 6.74 ns at divisor 17 to 4.95 ms at divisor 8,388,607 and serves as the ratio
    /// reference for the other variants.
    /// </summary>
    [Benchmark(Baseline = true, OperationsPerInvoke = Iterations)]
    public ulong CurrentImplementation()
    {
        ulong checksum = 0UL;
        ulong baseDivisor = Divisor | 1UL;

        for (int i = 0; i < Iterations; i++)
        {
            ulong candidate = ((baseDivisor + StepOffsets[i & StepMask]) | 1UL);
            checksum ^= CalculateCycleLengthGpu_Current(candidate);
        }

        return checksum;
    }

    /// <summary>
    /// Do-while style loop; nearly matches the baseline (5.88 ns at 17, 9.41 ns at 31) and trims ~5% off the largest divisor with
    /// 4.95 ms at 8,388,607.
    /// </summary>
    [Benchmark(OperationsPerInvoke = Iterations)]
    public ulong DoWhileImplementation()
    {
        ulong checksum = 0UL;
        ulong baseDivisor = Divisor | 1UL;

        for (int i = 0; i < Iterations; i++)
        {
            ulong candidate = ((baseDivisor + StepOffsets[i & StepMask]) | 1UL);
            checksum ^= CalculateCycleLengthGpu_DoWhile(candidate);
        }

        return checksum;
    }

    /// <summary>
    /// Double-subtract variant retained for comparison; ranges from 6.30 ns at divisor 17 to 4.92 ms at 8,388,607, generally matching
    /// or slightly trailing the baseline.
    /// </summary>
    [Benchmark(OperationsPerInvoke = Iterations)]
    public ulong DoubleSubtractImplementation()
    {
        ulong checksum = 0UL;
        ulong baseDivisor = Divisor | 1UL;

        for (int i = 0; i < Iterations; i++)
        {
            ulong candidate = ((baseDivisor + StepOffsets[i & StepMask]) | 1UL);
            checksum ^= CalculateCycleLengthGpu_DoubleSubtract(candidate);
        }

        return checksum;
    }

    /// <summary>
    /// UInt128 modulo helper used for comparison; incurs 42.36 ns at divisor 17 and stretches to 6.17 ms at 8,388,607, making it the
    /// slowest option across the board.
    /// </summary>
    [Benchmark(OperationsPerInvoke = Iterations)]
    public ulong UInt128ModuloImplementation()
    {
        ulong checksum = 0UL;
        ulong baseDivisor = Divisor | 1UL;

        for (int i = 0; i < Iterations; i++)
        {
            ulong candidate = ((baseDivisor + StepOffsets[i & StepMask]) | 1UL);
            checksum ^= CalculateCycleLengthGpu_UInt128Modulo(candidate);
        }

        return checksum;
    }

    /// <summary>
    /// Two-step unrolled loop; improves small divisors to 6.79 ns (17) and large ones to 4.78 ms (8,388,607), shaving ~3–5% off the
    /// baseline at the top end.
    /// </summary>
    [Benchmark(OperationsPerInvoke = Iterations)]
    public ulong UnrolledPairImplementation()
    {
        ulong checksum = 0UL;
        ulong baseDivisor = Divisor | 1UL;

        for (int i = 0; i < Iterations; i++)
        {
            ulong candidate = ((baseDivisor + StepOffsets[i & StepMask]) | 1UL);
            checksum ^= CalculateCycleLengthGpu_UnrolledPair(candidate);
        }

        return checksum;
    }

    /// <summary>
    /// Four-step unrolled loop; posts 5.83 ns at divisor 17 and 4.75 ms at 8,388,607, outperforming the pair loop especially on large
    /// divisors (52.18 μs at 131,071 vs. 52.65 μs).
    /// </summary>
    [Benchmark(OperationsPerInvoke = Iterations)]
    public ulong UnrolledQuadImplementation()
    {
        ulong checksum = 0UL;
        ulong baseDivisor = Divisor | 1UL;

        for (int i = 0; i < Iterations; i++)
        {
            ulong candidate = ((baseDivisor + StepOffsets[i & StepMask]) | 1UL);
            checksum ^= CalculateCycleLengthGpu_UnrolledQuad(candidate);
        }

        return checksum;
    }

    /// <summary>
    /// Eight-step unrolled loop; 5.45 ns at divisor 17, 582 ns at 8,191, 50.45 μs at 131,071, and 4.67 ms at 8,388,607—slightly behind the
    /// hex loop on the largest case but faster on mid-range divisors.
    /// </summary>
    [Benchmark(OperationsPerInvoke = Iterations)]
    public ulong UnrolledOctImplementation()
    {
        ulong checksum = 0UL;
        ulong baseDivisor = Divisor | 1UL;

        for (int i = 0; i < Iterations; i++)
        {
            ulong candidate = ((baseDivisor + StepOffsets[i & StepMask]) | 1UL);
            checksum ^= CalculateCycleLengthGpu_UnrolledOct(candidate);
        }

        return checksum;
    }

    /// <summary>
    /// Sixteen-step unrolled loop (hex); leads the pack at the largest divisor with 4.59 ms and remains competitive elsewhere (4.83 ns at 17,
    /// 5.13 ns at 31, 50.52 μs at 131,071).
    /// </summary>
    [Benchmark(OperationsPerInvoke = Iterations)]
    public ulong UnrolledHexImplementation()
    {
        ulong checksum = 0UL;
        ulong baseDivisor = Divisor | 1UL;

        for (int i = 0; i < Iterations; i++)
        {
            ulong candidate = ((baseDivisor + StepOffsets[i & StepMask]) | 1UL);
            checksum ^= CalculateCycleLengthGpu_UnrolledHex(candidate);
        }

        return checksum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong CalculateCycleLengthGpu_Current(ulong divisor)
    {
        // EvenPerfectBitScanner never routes power-of-two divisors here; keep the guard disabled outside tests and benchmarks.
        // if ((divisor & (divisor - 1UL)) == 0UL)
        // {
        //     return 1UL;
        // }

        ulong order = 1UL;
        ulong pow = 2UL;

        while (pow != 1UL)
        {
            pow <<= 1;
            if (pow >= divisor)
            {
                pow -= divisor;
            }

            order++;
        }

        return order;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong CalculateCycleLengthGpu_DoWhile(ulong divisor)
    {
        // EvenPerfectBitScanner never routes power-of-two divisors here; keep the guard disabled outside tests and benchmarks.
        // if ((divisor & (divisor - 1UL)) == 0UL)
        // {
        //     return 1UL;
        // }

        ulong order = 1UL;
        ulong pow = 2UL;

        while (true)
        {
            pow <<= 1;
            if (pow >= divisor)
            {
                pow -= divisor;
            }

            order++;

            if (pow == 1UL)
            {
                return order;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong CalculateCycleLengthGpu_DoubleSubtract(ulong divisor)
    {
        // EvenPerfectBitScanner never routes power-of-two divisors here; keep the guard disabled outside tests and benchmarks.
        // if ((divisor & (divisor - 1UL)) == 0UL)
        // {
        //     return 1UL;
        // }

        ulong order = 1UL;
        ulong pow = 2UL;

        while (pow != 1UL)
        {
            pow += pow;
            if (pow >= divisor)
            {
                pow -= divisor;
                if (pow >= divisor)
                {
                    pow -= divisor;
                }
            }

            order++;
        }

        return order;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong CalculateCycleLengthGpu_UInt128Modulo(ulong divisor)
    {
        // EvenPerfectBitScanner never routes power-of-two divisors here; keep the guard disabled outside tests and benchmarks.
        // if ((divisor & (divisor - 1UL)) == 0UL)
        // {
        //     return 1UL;
        // }

        ulong order = 1UL;
        ulong pow = 2UL;

        while (pow != 1UL)
        {
            System.UInt128 doubled = (System.UInt128)pow << 1;
            pow = (ulong)(doubled % divisor);

            order++;
        }

        return order;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong CalculateCycleLengthGpu_UnrolledPair(ulong divisor)
    {
        // EvenPerfectBitScanner never routes power-of-two divisors here; keep the guard disabled outside tests and benchmarks.
        // if ((divisor & (divisor - 1UL)) == 0UL)
        // {
        //     return 1UL;
        // }

        ulong order = 1UL;
        ulong pow = 2UL;

        while (true)
        {
            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong CalculateCycleLengthGpu_UnrolledQuad(ulong divisor)
    {
        // EvenPerfectBitScanner never routes power-of-two divisors here; keep the guard disabled outside tests and benchmarks.
        // if ((divisor & (divisor - 1UL)) == 0UL)
        // {
        //     return 1UL;
        // }

        ulong order = 1UL;
        ulong pow = 2UL;

        while (true)
        {
            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong CalculateCycleLengthGpu_UnrolledOct(ulong divisor)
    {
        // EvenPerfectBitScanner never routes power-of-two divisors here; keep the guard disabled outside tests and benchmarks.
        // if ((divisor & (divisor - 1UL)) == 0UL)
        // {
        //     return 1UL;
        // }

        ulong order = 1UL;
        ulong pow = 2UL;

        while (true)
        {
            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong CalculateCycleLengthGpu_UnrolledHex(ulong divisor)
    {
        // EvenPerfectBitScanner never routes power-of-two divisors here; keep the guard disabled outside tests and benchmarks.
        // if ((divisor & (divisor - 1UL)) == 0UL)
        // {
        //     return 1UL;
        // }

        ulong order = 1UL;
        ulong pow = 2UL;

        while (true)
        {
            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }

            if (AdvanceMersenneDivisorCycleStep(ref pow, divisor, ref order))
            {
                return order;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool AdvanceMersenneDivisorCycleStep(ref ulong pow, ulong divisor, ref ulong order)
    {
        pow += pow;
        if (pow >= divisor)
        {
            pow -= divisor;
        }

        order++;
        return pow == 1UL;
    }

}
