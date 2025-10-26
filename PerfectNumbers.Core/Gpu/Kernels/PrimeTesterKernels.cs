using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Algorithms;

namespace PerfectNumbers.Core.Gpu;

internal static class PrimeTesterKernels
{
    // GPU kernel: small-prime sieve only. Returns 1 if passes sieve (probable prime), 0 otherwise.
    public static void SmallPrimeSieveKernel(
        Index1D index,
        ArrayView<ulong> numbers,
        ArrayView<uint> smallPrimes,
        ArrayView<ulong> smallPrimeSquares,
        ArrayView<byte> results)
    {
        ulong n = numbers[index];
        byte result = 1;

        // EvenPerfectBitScanner filters candidates so production GPU launches only see:
        //  - n >= 31 (the scanner starts at 136,279,841 and test mode seeds 31)
        //  - n odd (candidate generators and ModResidueTracker eliminate even exponents)
        //  - n not divisible by 5 (residue prefiltering and AddPrimes transform cover these composites)
        // Tests should honor the same constraints so the host wrapper can remain branchless.

        long length = smallPrimes.Length;
        for (int i = 0; i < length; i++)
        {
            ulong prime = smallPrimes[i];
            ulong primeSquare = smallPrimeSquares[i];
            if (primeSquare > n)
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

    public static void HeuristicTrialDivisionKernel(Index1D index, ArrayView<ulong> divisors, ulong n, ArrayView<byte> results)
    {
        ulong divisor = divisors[index];
        if (divisor <= 1UL)
        {
            results[index] = 0;
            return;
        }

        results[index] = n % divisor == 0UL ? (byte)1 : (byte)0;
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
