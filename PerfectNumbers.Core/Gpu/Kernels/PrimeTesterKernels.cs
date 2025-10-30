using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

internal enum HeuristicGpuDivisorTableKind : byte
{
    GroupA = 0,
    GroupBEnding1 = 1,
    GroupBEnding7 = 7,
    GroupBEnding9 = 9,
    Combined = 255,
}

internal readonly struct HeuristicGpuDivisorTables
{
    public HeuristicGpuDivisorTables(
        ArrayView1D<ulong, Stride1D.Dense> combinedDivisorsEnding1,
        ArrayView1D<ulong, Stride1D.Dense> combinedDivisorSquaresEnding1,
        ArrayView1D<ulong, Stride1D.Dense> combinedDivisorsEnding3,
        ArrayView1D<ulong, Stride1D.Dense> combinedDivisorSquaresEnding3,
        ArrayView1D<ulong, Stride1D.Dense> combinedDivisorsEnding7,
        ArrayView1D<ulong, Stride1D.Dense> combinedDivisorSquaresEnding7,
        ArrayView1D<ulong, Stride1D.Dense> combinedDivisorsEnding9,
        ArrayView1D<ulong, Stride1D.Dense> combinedDivisorSquaresEnding9,
        ArrayView1D<ulong, Stride1D.Dense> groupADivisors,
        ArrayView1D<ulong, Stride1D.Dense> groupADivisorSquares,
        ArrayView1D<ulong, Stride1D.Dense> groupBDivisorsEnding1,
        ArrayView1D<ulong, Stride1D.Dense> groupBDivisorSquaresEnding1,
        ArrayView1D<ulong, Stride1D.Dense> groupBDivisorsEnding7,
        ArrayView1D<ulong, Stride1D.Dense> groupBDivisorSquaresEnding7,
        ArrayView1D<ulong, Stride1D.Dense> groupBDivisorsEnding9,
        ArrayView1D<ulong, Stride1D.Dense> groupBDivisorSquaresEnding9)
    {
        CombinedDivisorsEnding1 = combinedDivisorsEnding1;
        CombinedDivisorSquaresEnding1 = combinedDivisorSquaresEnding1;
        CombinedDivisorsEnding3 = combinedDivisorsEnding3;
        CombinedDivisorSquaresEnding3 = combinedDivisorSquaresEnding3;
        CombinedDivisorsEnding7 = combinedDivisorsEnding7;
        CombinedDivisorSquaresEnding7 = combinedDivisorSquaresEnding7;
        CombinedDivisorsEnding9 = combinedDivisorsEnding9;
        CombinedDivisorSquaresEnding9 = combinedDivisorSquaresEnding9;
        GroupADivisors = groupADivisors;
        GroupADivisorSquares = groupADivisorSquares;
        GroupBDivisorsEnding1 = groupBDivisorsEnding1;
        GroupBDivisorSquaresEnding1 = groupBDivisorSquaresEnding1;
        GroupBDivisorsEnding7 = groupBDivisorsEnding7;
        GroupBDivisorSquaresEnding7 = groupBDivisorSquaresEnding7;
        GroupBDivisorsEnding9 = groupBDivisorsEnding9;
        GroupBDivisorSquaresEnding9 = groupBDivisorSquaresEnding9;
    }

    public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorsEnding1;
    public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorSquaresEnding1;
    public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorsEnding3;
    public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorSquaresEnding3;
    public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorsEnding7;
    public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorSquaresEnding7;
    public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorsEnding9;
    public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorSquaresEnding9;
    public readonly ArrayView1D<ulong, Stride1D.Dense> GroupADivisors;
    public readonly ArrayView1D<ulong, Stride1D.Dense> GroupADivisorSquares;
    public readonly ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorsEnding1;
    public readonly ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorSquaresEnding1;
    public readonly ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorsEnding7;
    public readonly ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorSquaresEnding7;
    public readonly ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorsEnding9;
    public readonly ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorSquaresEnding9;
}

internal static class PrimeTesterKernels
{
    // GPU kernel: small-prime sieve only. Returns 1 if passes sieve (probable prime), 0 otherwise.
    public static void SmallPrimeSieveKernel(
        Index1D index,
        ArrayView<ulong> numbers,
        ArrayView<uint> smallPrimes,
        ArrayView<uint> smallPrimesLastOne,
        ArrayView<uint> smallPrimesLastSeven,
        ArrayView<uint> smallPrimesLastThree,
        ArrayView<uint> smallPrimesLastNine,
        ArrayView<ulong> smallPrimesPow2,
        ArrayView<ulong> smallPrimesPow2LastOne,
        ArrayView<ulong> smallPrimesPow2LastSeven,
        ArrayView<ulong> smallPrimesPow2LastThree,
        ArrayView<ulong> smallPrimesPow2LastNine,
        ArrayView<byte> results)
    {
        ulong n = numbers[index];
        byte result = 1;

        ArrayView<uint> primes = smallPrimes;
        ArrayView<ulong> primeSquares = smallPrimesPow2;
        ulong lastDigit = n % 10UL;
        switch (lastDigit)
        {
            case 1UL:
                primes = smallPrimesLastOne;
                primeSquares = smallPrimesPow2LastOne;
                break;
            case 3UL:
                primes = smallPrimesLastThree;
                primeSquares = smallPrimesPow2LastThree;
                break;
            case 7UL:
                primes = smallPrimesLastSeven;
                primeSquares = smallPrimesPow2LastSeven;
                break;
            case 9UL:
                primes = smallPrimesLastNine;
                primeSquares = smallPrimesPow2LastNine;
                break;
        }

        int length = (int)primes.Length;
        for (int i = 0; i < length; i++)
        {
            ulong primeSquare = primeSquares[i];
            if (primeSquare > n)
            {
                break;
            }

            ulong prime = primes[i];
            if (prime == 0)
            {
                break;
            }

            if (n % prime == 0UL)
            {
                result = 0;
                break;
            }
        }

        results[index] = result;
    }

    public static void HeuristicTrialDivisionKernel(
        Index1D index,
        HeuristicGpuDivisorTables tables,
        ArrayView<int> resultFlag,
        ulong n,
        ulong maxDivisorSquare,
        HeuristicGpuDivisorTableKind tableKind,
        byte nMod10)
    {
        ArrayView1D<ulong, Stride1D.Dense> divisors;
        ArrayView1D<ulong, Stride1D.Dense> divisorSquares;

        switch (tableKind)
        {
            case HeuristicGpuDivisorTableKind.GroupA:
                divisors = tables.GroupADivisors;
                divisorSquares = tables.GroupADivisorSquares;
                break;
            case HeuristicGpuDivisorTableKind.GroupBEnding1:
                divisors = tables.GroupBDivisorsEnding1;
                divisorSquares = tables.GroupBDivisorSquaresEnding1;
                break;
            case HeuristicGpuDivisorTableKind.GroupBEnding7:
                divisors = tables.GroupBDivisorsEnding7;
                divisorSquares = tables.GroupBDivisorSquaresEnding7;
                break;
            case HeuristicGpuDivisorTableKind.GroupBEnding9:
                divisors = tables.GroupBDivisorsEnding9;
                divisorSquares = tables.GroupBDivisorSquaresEnding9;
                break;
            case HeuristicGpuDivisorTableKind.Combined:
                switch (nMod10)
                {
                    case 1:
                        divisors = tables.CombinedDivisorsEnding1;
                        divisorSquares = tables.CombinedDivisorSquaresEnding1;
                        break;
                    case 3:
                        divisors = tables.CombinedDivisorsEnding3;
                        divisorSquares = tables.CombinedDivisorSquaresEnding3;
                        break;
                    case 7:
                        divisors = tables.CombinedDivisorsEnding7;
                        divisorSquares = tables.CombinedDivisorSquaresEnding7;
                        break;
                    case 9:
                        divisors = tables.CombinedDivisorsEnding9;
                        divisorSquares = tables.CombinedDivisorSquaresEnding9;
                        break;
                    default:
                        return;
                }

                break;
            default:
                return;
        }

        ulong divisorSquare = divisorSquares[index];
        if (divisorSquare > maxDivisorSquare)
        {
            return;
        }

        ulong divisor = divisors[index];
        if (n % divisor == 0UL)
        {
            resultFlag[0] = 1;
        }
    }

    public static void SharesFactorKernel(Index1D index, ArrayView<ulong> numbers, ArrayView<byte> results)
    {
        ulong n = numbers[index];
        ulong m = 63UL - (ulong)XMath.LeadingZeroCount(n);
        ulong gcd = BinaryGcdGpu(n, m);
        results[index] = gcd == 1UL ? (byte)0 : (byte)1;
    }

    private static ulong BinaryGcdGpu(ulong u, ulong v)
    {
        // TODO: Replace this inline GPU binary GCD with the kernel extracted from
        // GpuUInt128BinaryGcdBenchmarks via GpuKernelPool so device callers reuse the
        // fully unrolled ladder instead of this branchy fallback.
        bool uIsZero = u == 0UL;
        bool vIsZero = v == 0UL;
        bool eitherZero = uIsZero | vIsZero;
        ulong zeroResult = uIsZero ? v : u;

        ulong combined = u | v | 1UL;
        int shift = XMath.TrailingZeroCount(combined);

        ulong normalizedU = u >> XMath.TrailingZeroCount(u | 1UL);
        ulong normalizedV = v >> XMath.TrailingZeroCount(v | 1UL);

        ulong currentU = normalizedU;
        ulong currentV = normalizedV;
        bool loopCondition = currentV != 0UL;

        while (loopCondition)
        {
            currentV >>= XMath.TrailingZeroCount(currentV);
            ulong minValue = XMath.Min(currentU, currentV);
            ulong maxValue = XMath.Max(currentU, currentV);
            currentU = minValue;
            currentV = maxValue - minValue;
            loopCondition = currentV != 0UL;
        }

        ulong gcd = currentU << shift;
        return eitherZero ? zeroResult : gcd;
    }
}
