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

        int length = XMath.Min((int)primes.Length, (int)primeSquares.Length);
        for (int i = 0; i < length; i++)
        {
            ulong prime = primes[i];
            ulong primeSquare = primeSquares[i];
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
