using System.Collections.Generic;
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
    /// Baseline single-step GPU loop kept as a reference; every unrolled variant we measured (pair, quad, oct, hex) outran it from divisors 17 through 8,388,607.
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
    /// Do-while style loop; fastest among the legacy variants once divisors reach 131,071+, though still behind the unrolled loops.
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
    /// Double-subtract variant retained for comparison; recent runs (17–8,388,607) show it trailing both the baseline and unrolled loops.
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
    /// UInt128 modulo helper used for comparison; consistently the slowest due to wide arithmetic overhead.
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
    /// Two-step unrolled loop that now trails the deeper octo/hex variants (e.g., 100.7 µs at 131,071 and 9.50 ms at 8,388,607) but still clears the baseline and legacy loops by a wide margin.
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
    /// Four-step unrolled loop that used to lead across the board; recent measurements show it ceding the crown to the octo/hex versions, though it remains ~5% faster than the pair loop at divisor 131,071.
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
    /// Eight-step unrolled loop that posted the best time at divisor 131,071 (~99.8 µs) and stayed within ~0.6% of the hex variant at 8,388,607.
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
    /// Sixteen-step unrolled loop that becomes the clear leader on the largest input we tested (8,388,607 → 9.25 ms) while sitting about 2% behind the octo version at divisor 131,071.
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
        if ((divisor & (divisor - 1UL)) == 0UL)
        {
            return 1UL;
        }

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
        if ((divisor & (divisor - 1UL)) == 0UL)
        {
            return 1UL;
        }

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
        if ((divisor & (divisor - 1UL)) == 0UL)
        {
            return 1UL;
        }

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
        if ((divisor & (divisor - 1UL)) == 0UL)
        {
            return 1UL;
        }

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
        if ((divisor & (divisor - 1UL)) == 0UL)
        {
            return 1UL;
        }

        ulong order = 1UL;
        ulong pow = 2UL;

        while (true)
        {
            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong CalculateCycleLengthGpu_UnrolledQuad(ulong divisor)
    {
        if ((divisor & (divisor - 1UL)) == 0UL)
        {
            return 1UL;
        }

        ulong order = 1UL;
        ulong pow = 2UL;

        while (true)
        {
            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong CalculateCycleLengthGpu_UnrolledOct(ulong divisor)
    {
        if ((divisor & (divisor - 1UL)) == 0UL)
        {
            return 1UL;
        }

        ulong order = 1UL;
        ulong pow = 2UL;

        while (true)
        {
            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong CalculateCycleLengthGpu_UnrolledHex(ulong divisor)
    {
        if ((divisor & (divisor - 1UL)) == 0UL)
        {
            return 1UL;
        }

        ulong order = 1UL;
        ulong pow = 2UL;

        while (true)
        {
            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

            if (Step(ref pow, divisor, ref order))
            {
                return order;
            }

        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool Step(ref ulong pow, ulong divisor, ref ulong order)
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
