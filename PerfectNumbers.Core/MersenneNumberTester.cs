using System.Runtime.CompilerServices;
using System.Collections.Concurrent;
using System.Buffers;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Cpu;
using PerfectNumbers.Core.Gpu;
using UInt128 = System.UInt128;

namespace PerfectNumbers.Core;

public enum GpuKernelType
{
	Incremental,
	Pow2Mod,
}

public sealed class MersenneNumberTester(
    bool useIncremental = true,
    bool useOrderCache = false,
    GpuKernelType kernelType = GpuKernelType.Incremental,
    bool useModuloWorkaround = false,
    bool useOrder = false,
    bool useGpuLucas = true,
    bool useGpuScan = true,
    bool useGpuOrder = true,
        bool useResidue = true,
    ulong maxK = 5_000_000UL)
{
    private readonly bool _useResidue = useResidue;
    private readonly bool _useIncremental = useIncremental && !useResidue;
        private readonly ulong _maxK = maxK;
	private readonly GpuKernelType _kernelType = kernelType;
    private readonly bool _useModuloWorkaround = useModuloWorkaround;
    private readonly bool _useGpuLucas = useGpuLucas;
    private readonly bool _useGpuScan = useGpuScan;     // device for pow2mod/incremental scanning
    private readonly bool _useGpuOrder = useGpuOrder;   // device for order computations
    private static readonly ConcurrentDictionary<UInt128, ulong> OrderCache = new();
    // TODO: Swap this ConcurrentDictionary for the pooled dictionary variant highlighted in
    // Pow2MontgomeryModBenchmarks once order warmups reuse deterministic divisor-cycle snapshots; the pooled approach removed
    // the locking overhead when scanning p >= 138M in those benchmarks.
	private readonly MersenneNumberIncrementalGpuTester _incrementalGpuTester = new(kernelType, useGpuOrder);
	private readonly MersenneNumberIncrementalCpuTester _incrementalCpuTester = new(kernelType);
	private readonly MersenneNumberOrderGpuTester _orderGpuTester = new(kernelType, useGpuOrder);
	private readonly MersenneNumberOrderCpuTester _orderCpuTester = new(kernelType);
	private readonly MersenneNumberLucasLehmerGpuTester _lucasLehmerGpuTester = new();
	private readonly MersenneNumberLucasLehmerCpuTester _lucasLehmerCpuTester = new();
	private readonly MersenneNumberResidueGpuTester? _residueGpuTester = useResidue ? new(useGpuOrder) : null;
	private readonly MersenneNumberResidueCpuTester? _residueCpuTester = useResidue ? new() : null;

	// TODO: Ensure Program passes useResidue correctly and that residue-vs-incremental-vs-LL selection respects CLI flags.
	// Also consider injecting a shared cycles cache (MersenneDivisorCycles.Shared) to both CPU and GPU testers.

    public void WarmUpOrders(ulong exponent, ulong limit = 5_000_000UL)
	{
		bool cacheEnabled = useOrderCache;
		if (!cacheEnabled)
		{
			return;
		}

                UInt128 twoP = (UInt128)exponent << 1; // 2 * p
                bool lastIsSeven = (exponent & 3UL) == 3UL;
                // CPU path: compute orders directly and cache
                if (!_useGpuOrder)
                {
                        UInt128 k = UInt128.One;
			UInt128 q;
			ulong remainder;
			for (; k <= limit; k++)
                        {
                                q = twoP * k + UInt128.One;
                                remainder = q.Mod10();
                                // TODO: Replace these repeated Mod10/Mod8/Mod3/Mod5 calls with a residue automaton walk so
                                // order warmups reuse the optimized cycle stepping validated in the residue benchmarks.
                                bool shouldCheck = lastIsSeven
                                        ? remainder == 7UL || remainder == 9UL || remainder == 3UL
                                        : remainder == 1UL || remainder == 3UL || remainder == 9UL;

                                // Early modular filters for odd order and small-prime rejections
				if (shouldCheck)
				{
                                    ulong mod8 = q.Mod8();
                                        if (mod8 != 1UL && mod8 != 7UL)
                                        {
                                                continue;
                                        }
                                    else if (q.Mod3() == 0UL || q.Mod5() == 0UL)
                                        {
						continue;
					}
				}

				if (!shouldCheck || !cacheEnabled || (cacheEnabled && OrderCache.ContainsKey(q)))
				{
					continue;
				}

                                // TODO: Swap this fallback to the shared windowed order helper once CalculateOrder migrates
                                // to the ProcessEightBitWindows pipeline so warm-ups stop invoking the slower legacy powmod.
                                ulong ord = q.CalculateOrder();
                                OrderCache[q] = ord;
                        }

			return;
		}

		UInt128[] qsBuffer = ArrayPool<UInt128>.Shared.Rent((int)limit);
		int idx = 0;
		UInt128 k2 = 1UL;
		UInt128 q2;
		ulong remainder2;
		for (; k2 <= limit; k2++)
		{
                        q2 = twoP * k2 + 1UL;
                        remainder2 = q2.Mod10();
                        // TODO: Reuse the same residue-automaton fast path here so the GPU warmup staging avoids `%` and
                        // branches the benchmarks showed slower than the tracked stepping helper.
                        bool shouldCheck2 = lastIsSeven
                                ? remainder2 == 7UL || remainder2 == 9UL || remainder2 == 3UL
                                : remainder2 == 1UL || remainder2 == 3UL || remainder2 == 9UL;

                        // Early modular filters for odd order and small-prime rejections
			if (shouldCheck2)
			{
                            ulong mod8 = q2.Mod8();
				if (mod8 != 1UL && mod8 != 7UL)
				{
					shouldCheck2 = false;
				}
                            else if (q2.Mod3() == 0UL || q2.Mod5() == 0UL)
				{
					shouldCheck2 = false;
				}
			}

			if (!shouldCheck2 || (cacheEnabled && OrderCache.ContainsKey(q2)))
			{
				continue;
			}

			qsBuffer[idx++] = q2;
		}

		if (idx == 0)
		{
			ArrayPool<UInt128>.Shared.Return(qsBuffer);
			return;
		}

		UInt128[] qs = qsBuffer;

                var gpuLease = GpuKernelPool.GetKernel(_useGpuOrder);
                var execution = gpuLease.EnterExecutionScope();
                var accelerator = gpuLease.Accelerator;
                var stream = gpuLease.Stream;
                var orderKernel = gpuLease.OrderKernel!;

                try
                {
                        // Guard long-running kernels by chunking the warm-up across batches.
                        // Reuse the same ScanBatchSize knob used for scanning.
                        int batchSize = GpuConstants.ScanBatchSize;
                        ulong divMul = (ulong)((((UInt128)1 << 64) - 1UL) / exponent) + 1UL;
                        int offset = 0;
                        while (offset < idx)
                        {
                                int count = Math.Min(batchSize, idx - offset);
                                var qBuffer = accelerator.Allocate1D<PerfectNumbers.Core.Gpu.GpuUInt128>(count);
                                var orderBuffer = accelerator.Allocate1D<ulong>(count);
                                // Convert to device-friendly GpuUInt128 on the fly
                                var tmp = System.Buffers.ArrayPool<PerfectNumbers.Core.Gpu.GpuUInt128>.Shared.Rent(count);
                                for (int i = 0; i < count; i++)
                                {
                                        tmp[i] = (PerfectNumbers.Core.Gpu.GpuUInt128)qs[offset + i];
                                }
                                // Copy only the populated elements to the device buffer to avoid overrun on pooled arrays.
                                qBuffer.View.CopyFromCPU(ref tmp[0], count);
                                System.Buffers.ArrayPool<PerfectNumbers.Core.Gpu.GpuUInt128>.Shared.Return(tmp);
                                orderKernel(stream, count, exponent, divMul, qBuffer.View, orderBuffer.View);
                                stream.Synchronize();

                                ulong[] orders = orderBuffer.GetAsArray1D();
                                for (int i = 0; i < count; i++)
                                {
                                        ulong order = orders[i];
                                        if (order == 0UL)
                                        {
                                                // TODO: Replace this CalculateOrder call with the upcoming windowed helper so
                                                // GPU warm-ups reuse the faster pow2 ladder measured in the CPU order benchmarks.
                                                order = qs[offset + i].CalculateOrder();
                                        }

                                        if (useOrderCache)
                                        {
                                                OrderCache[qs[offset + i]] = order;
                                        }
                                }

                                offset += count;
                                orderBuffer.Dispose();
                                qBuffer.Dispose();
                        }
                }
                finally
                {
                        execution.Dispose();
                        gpuLease.Dispose();
                        ArrayPool<UInt128>.Shared.Return(qs);
                }
        }

    public bool IsMersennePrime(ulong exponent)
    {
        // Safe early rejections using known orders for tiny primes:
        // 7 | M_p iff 3 | p. Avoid rejecting p == 3 where M_3 == 7 is prime.
        if ((exponent % 3UL) == 0UL && exponent != 3UL)
        {
            // TODO: Replace this `% 3` check with ULongExtensions.Mod3 to align the early rejection with
            // the benchmarked bitmask helper instead of generic modulo for CPU workloads.
            return false;
        }

        if ((exponent & 3UL) == 1UL && exponent.SharesFactorWithExponentMinusOne())
        {
            return false;
        }

        // Skip even exponents: for even p > 2, M_p is composite; here p is large.
        if ((exponent & 1UL) == 0UL)
        {
            return false;
        }

		bool prePrime = true;
        UInt128 twoP = (UInt128)exponent << 1; // 2 * p
        // UInt128 maxDivisor = new(ulong.MaxValue, ulong.MaxValue);
        UInt128 maxK = UInt128Numbers.Two.Pow(exponent) / 2 + 1;
        bool lastIsSeven = (exponent & 3UL) == 3UL;

        // In residue mode the residue scan is the final primality check.
        if (_useResidue)
        {
            if (_maxK < maxK)
            {
                maxK = _maxK;
            }

            if (_useGpuScan)
            {
                _residueGpuTester!.Scan(exponent, twoP, lastIsSeven, maxK, ref prePrime);
            }
            else
            {
                _residueCpuTester!.Scan(exponent, twoP, lastIsSeven, maxK, ref prePrime);
            }

            return prePrime;
        }

		if (!_useIncremental)
		{
			// Optional prefilter before Lucas–Lehmer using the same GPU/CPU scan path.
			// Scan a limited range of k to quickly reject obvious cases.
			UInt128 twoPPre = (UInt128)exponent << 1;
			UInt128 maxKPre = (UInt128)PreLucasPreScanK;
			bool lastIsSevenPre = (exponent & 3UL) == 3UL;
			if (_useGpuScan)
			{
				_incrementalGpuTester.Scan(exponent, twoPPre, lastIsSevenPre, maxKPre, ref prePrime);
			}
			else
			{
				_incrementalCpuTester.Scan(exponent, twoPPre, lastIsSevenPre, maxKPre, ref prePrime);
			}

			if (!prePrime)
			{
				return false;
			}

			return _useGpuLucas
				? _lucasLehmerGpuTester.IsPrime(exponent, runOnGpu: true)
				: _lucasLehmerCpuTester.IsPrime(exponent);
		}

		bool isPrime = true;

		if (useOrder)
		{
			if (_useGpuOrder)
			{
				_orderGpuTester.Scan(exponent, twoP, lastIsSeven, maxK, ref isPrime);
			}
			else
			{
				_orderCpuTester.Scan(exponent, twoP, lastIsSeven, maxK, ref isPrime);
			}
		}
		else
		{
			if (_useGpuScan)
			{
				_incrementalGpuTester.Scan(exponent, twoP, lastIsSeven, maxK, ref isPrime);
			}
			else
			{
				_incrementalCpuTester.Scan(exponent, twoP, lastIsSeven, maxK, ref isPrime);
			}
		}

        return isPrime;
    }

	// Limit of k to scan in the pre-Lucas prescan
	public const int PreLucasPreScanK = 5_000_000;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool DividesMersenne(ulong exponent, ulong prime)
	{
		return 2UL.ModPow64(exponent, prime) == 1UL;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void LegacyOrderKernel(Index1D index, ulong exponent, ArrayView<UInt128> qs, ArrayView<ulong> orders)
        {
                UInt128 q = qs[index];
                // TODO: Swap this legacy kernel over to the ProcessEightBitWindows helper so GPU order scans share the
                // eight-bit window pow2 implementation that beat the classic PowMod path in the GpuPow2Mod benchmarks.
                UInt128 pow = exponent.PowMod(q);
                orders[index] = pow == 1UL ? exponent : 0UL;
        }


}
