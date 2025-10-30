using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

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

        int length = (int)primes.Length;
        if (length <= 0)
        {
            results[index] = result;
            return;
        }

        int tileSize = Group.Dimension.X;
        if (tileSize <= 0)
        {
            tileSize = 1;
        }

        var sharedPrimes = SharedMemory.Allocate<uint>(tileSize);
        var sharedPrimeSquares = SharedMemory.Allocate<ulong>(tileSize);
        bool completed = false;

        int offset = 0;
        while (offset < length)
        {
            int chunkLength = length - offset;
            if (chunkLength > tileSize)
            {
                chunkLength = tileSize;
            }

            if (!completed && Group.Index.X < chunkLength)
            {
                int loadIndex = offset + Group.Index.X;
                sharedPrimes[Group.Index.X] = primes[loadIndex];
                sharedPrimeSquares[Group.Index.X] = primeSquares[loadIndex];
            }

            Group.Barrier();

            if (!completed)
            {
                for (int i = 0; i < chunkLength; i++)
                {
                    ulong primeSquare = sharedPrimeSquares[i];
                    if (primeSquare > n)
                    {
                        completed = true;
                        break;
                    }

                    ulong prime = sharedPrimes[i];
                    if (n % prime == 0UL)
                    {
                        result = 0;
                        completed = true;
                        break;
                    }
                }
            }

            Group.Barrier();
            offset += chunkLength;
        }

        results[index] = result;
    }

    public static void HeuristicTrialDivisionKernel(
        Index1D index,
        ArrayView1D<ulong, Stride1D.Dense> divisors,
        ArrayView1D<ulong, Stride1D.Dense> divisorSquares,
        ulong n,
        ulong maxDivisorSquare,
        ArrayView<int> resultFlag)
    {
        int idx = index;
        int length = (int)divisors.Length;
        if ((uint)idx >= (uint)length)
        {
            return;
        }

        ulong divisorSquare = divisorSquares[idx];
        if (divisorSquare <= 1UL || divisorSquare > maxDivisorSquare)
        {
            return;
        }

        ulong divisor = divisors[idx];
        if (divisor <= 1UL)
        {
            return;
        }

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
