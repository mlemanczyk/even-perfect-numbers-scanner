using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Algorithms;

namespace PerfectNumbers.Core.Gpu;

internal static class PrimeTesterKernels
{
    // GPU kernel: small-prime sieve only. Returns 1 if passes sieve (probable prime), 0 otherwise.
    public static void SmallPrimeSieveKernel(Index1D index, ArrayView<ulong> numbers, ArrayView<uint> smallPrimes, ArrayView<byte> results)
    {
        ulong n = numbers[index];
        if (n <= 3UL)
        {
            results[index] = (byte)(n >= 2UL ? 1 : 0);
            return;
        }

        if ((n & 1UL) == 0UL || (n > 5UL && (n % 5UL) == 0UL))
        {
            // TODO: Replace the `% 5` branch with the GPU Mod5 helper once the sieve kernel can
            // reuse the benchmarked bitmask reduction instead of modulo on every candidate.
            results[index] = 0;
            return;
        }

        if (n.Mod10() == 1UL)
        {
            // Early reject special GCD heuristic with floor(log2 n)
            ulong m = 63UL - (ulong)XMath.LeadingZeroCount(n);
            if (BinaryGcdGpu(n, m) != 1UL)
            {
                results[index] = 0;
                return;
            }
        }

        long len = smallPrimes.Length;
        for (int i = 0; i < len; i++)
        {
            ulong p = smallPrimes[i];
            if (p * p > n)
            {
                break;
            }

            if ((n % p) == 0UL)
            {
                // TODO: Route this modulo through the shared divisor-cycle cache once exposed to GPU kernels
                // so batched sieves avoid per-prime `%` operations that profiling showed expensive.
                results[index] = 0;
                return;
            }
        }

        results[index] = 1;
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
        if (u == 0UL)
        {
            return v;
        }

        if (v == 0UL)
        {
            return u;
        }

        int shift = XMath.TrailingZeroCount(u | v);
        u >>= XMath.TrailingZeroCount(u);

        do
        {
            v >>= XMath.TrailingZeroCount(v);
            if (u > v)
            {
                (u, v) = (v, u);
            }

            v -= u;
        }
        while (v != 0UL);

        return u << shift;
    }
}
