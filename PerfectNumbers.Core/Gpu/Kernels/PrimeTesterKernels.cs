using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public enum HeuristicGpuDivisorTableKind : byte
{
    GroupA = 0,
    GroupBEnding1 = 1,
    GroupBEnding7 = 7,
    GroupBEnding9 = 9,
    Combined = 255,
}

internal static class PrimeTesterKernels
{
    // GPU kernel: small-prime sieve only. Returns 1 if passes sieve (probable prime), 0 otherwise.
    public static void SmallPrimeSieveKernel(
		Index1D index,
		ArrayView<ulong> numbers,
		ArrayView<uint> smallPrimesLastOne,
		ArrayView<uint> smallPrimesLastSeven,
		ArrayView<uint> smallPrimesLastThree,
		ArrayView<uint> smallPrimesLastNine,
		ArrayView<ulong> smallPrimesPow2LastOne,
		ArrayView<ulong> smallPrimesPow2LastSeven,
		ArrayView<ulong> smallPrimesPow2LastThree,
		ArrayView<ulong> smallPrimesPow2LastNine,
		ArrayView<byte> results)
    {
        ulong n = numbers[index];
        byte result = 1;

        ArrayView<uint> primes;
        ArrayView<ulong> primeSquares;
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
			default:
				return;
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
        ArrayView<int> resultFlag,
        ulong n,
        ulong maxDivisorSquare,
        HeuristicGpuDivisorTableKind tableKind,
        HeuristicGpuDivisorTables tables)
    {
        byte nMod10 = (byte)(n % 10UL);
        ArrayView1D<ulong, Stride1D.Dense> divisors = tables.SelectDivisors(tableKind, nMod10);
        ArrayView1D<ulong, Stride1D.Dense> divisorSquares = tables.SelectDivisorSquares(tableKind, nMod10);

        int divisorLength = (int)divisorSquares.Length;
        int threadIndex = index;
        if (threadIndex >= divisorLength)
        {
            return;
        }

        ulong divisorSquare = divisorSquares[threadIndex];
        if (divisorSquare > maxDivisorSquare)
        {
            return;
        }

        ulong divisor = divisors[threadIndex];
        divisor = n % divisor;
        if (divisor == 0UL)
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
        bool uIsZero = u == 0UL;
        bool vIsZero = v == 0UL;
        bool eitherZero = uIsZero | vIsZero;
        ulong zeroResult = uIsZero ? v : u;

        ulong combined = (u | v) | 1UL;
        int shift = XMath.TrailingZeroCount(combined);
        ulong currentU = u >> XMath.TrailingZeroCount(u | 1UL);
        ulong currentV = v >> XMath.TrailingZeroCount(v | 1UL);

        while (true)
        {
            currentV >>= XMath.TrailingZeroCount(currentV);
            bool swap = currentU > currentV;
            ulong minValue = ConditionalSelect(currentU, currentV, swap);
            ulong maxValue = ConditionalSelect(currentV, currentU, swap);
            currentV = maxValue - minValue;
            currentU = minValue;
            if (currentV == 0UL)
            {
                ulong gcd = currentU << shift;
                return eitherZero ? zeroResult : gcd;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong ConditionalSelect(ulong left, ulong right, bool useRight)
    {
        ulong mask = useRight ? ulong.MaxValue : 0UL;
        return left ^ ((left ^ right) & mask);
    }

}
