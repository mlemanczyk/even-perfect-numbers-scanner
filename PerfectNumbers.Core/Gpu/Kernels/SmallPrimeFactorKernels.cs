using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

internal static class SmallPrimeFactorKernels
{
    public static void SmallPrimeFactorKernelScan(
        Index1D index,
        ulong value,
        uint limit,
        ArrayView1D<uint, Stride1D.Dense> primes,
        ArrayView1D<ulong, Stride1D.Dense> squares,
        int primeCount,
        ArrayView1D<ulong, Stride1D.Dense> primeSlots,
        ArrayView1D<int, Stride1D.Dense> exponentSlots,
        ArrayView1D<int, Stride1D.Dense> countSlot,
        ArrayView1D<ulong, Stride1D.Dense> remainingSlot)
    {
        if (index != 0)
        {
            return;
        }

        uint effectiveLimit = limit == 0 ? uint.MaxValue : limit;
        ulong remaining = value;
        int count = 0;
        int capacity = (int)primeSlots.Length;

        for (int i = 0; i < primeCount && remaining > 1UL && count < capacity; i++)
        {
            uint primeCandidate = primes[i];
            if (primeCandidate == 0U || primeCandidate > effectiveLimit)
            {
                break;
            }

            ulong primeSquare = squares[i];
            if (primeSquare != 0UL && primeSquare > remaining)
            {
                break;
            }

            ulong primeValue = primeCandidate;
            if ((remaining % primeValue) != 0UL)
            {
                continue;
            }

            int exponent = 0;
            do
            {
                remaining /= primeValue;
                exponent++;
            }
            while ((remaining % primeValue) == 0UL);

            primeSlots[count] = primeValue;
            exponentSlots[count] = exponent;
            count++;
        }

        countSlot[0] = count;
        remainingSlot[0] = remaining;
    }
}
