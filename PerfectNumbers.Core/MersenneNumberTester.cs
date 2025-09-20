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
    UInt128? maxK = null,
        UInt128? residueDivisorSets = null)
{
    private readonly bool _useResidue = useResidue;
    private readonly bool _useIncremental = useIncremental && !useResidue;
        private readonly UInt128 _maxK = maxK ?? (UInt128)5_000_000UL;
        private readonly UInt128 _residueDivisorSetCount = residueDivisorSets ?? (UInt128)PerfectNumberConstants.ExtraDivisorCycleSearchLimit;
        private readonly GpuKernelType _kernelType = kernelType;
    private readonly bool _useModuloWorkaround = useModuloWorkaround;
    private readonly bool _useGpuLucas = useGpuLucas;
    private readonly bool _useGpuScan = useGpuScan;     // device for pow2mod/incremental scanning
    private readonly bool _useGpuOrder = useGpuOrder;   // device for order computations
    private static readonly ConcurrentDictionary<UInt128, ulong> OrderCache = new();
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static UInt128 DivideRoundUp(UInt128 value, UInt128 divisor)
        {
                if (divisor == UInt128.Zero)
                {
                        return UInt128.Zero;
                }

                UInt128 result = value / divisor;
                if (result * divisor == value)
                {
                        return result;
                }

                return result + UInt128.One;
        }

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
                var accelerator = gpuLease.Accelerator;
        var orderKernel = gpuLease.OrderKernel!;

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
            orderKernel(count, exponent, divMul, qBuffer.View, orderBuffer.View);
            accelerator.Synchronize();

                        ulong[] orders = orderBuffer.GetAsArray1D();
                        for (int i = 0; i < count; i++)
                        {
                                ulong order = orders[i];
                                if (order == 0UL)
                                {
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

                gpuLease.Dispose();
                ArrayPool<UInt128>.Shared.Return(qs);
	}

    public bool IsMersennePrime(ulong exponent) => IsMersennePrime(exponent, out _);

    public bool IsMersennePrime(ulong exponent, out bool divisorsExhausted)
    {
        divisorsExhausted = false;

        // Safe early rejections using known orders for tiny primes:
        // 7 | M_p iff 3 | p. Avoid rejecting p == 3 where M_3 == 7 is prime.
        if ((exponent % 3UL) == 0UL && exponent != 3UL)
        {
            divisorsExhausted = true;
            return false;
        }

        if ((exponent & 3UL) == 1UL && exponent.SharesFactorWithExponentMinusOne())
        {
            divisorsExhausted = true;
            return false;
        }

        // Skip even exponents: for even p > 2, M_p is composite; here p is large.
        if ((exponent & 1UL) == 0UL)
        {
            divisorsExhausted = true;
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

            if (_maxK == UInt128.Zero || _residueDivisorSetCount == UInt128.Zero)
            {
                return prePrime;
            }

            UInt128 totalLimit = _maxK;
            if (totalLimit > maxK)
            {
                totalLimit = maxK;
            }

            if (totalLimit == UInt128.Zero)
            {
                return true;
            }

            UInt128 requestedSets = _residueDivisorSetCount;
            if (_useGpuScan)
            {
                UInt128 maxUsefulSets = DivideRoundUp(totalLimit, (UInt128)GpuConstants.ScanBatchSize);
                if (maxUsefulSets == UInt128.Zero)
                {
                    maxUsefulSets = UInt128.One;
                }

                if (requestedSets == UInt128.Zero || requestedSets > maxUsefulSets)
                {
                    requestedSets = maxUsefulSets;
                }
            }

            if (requestedSets == UInt128.Zero)
            {
                requestedSets = UInt128.One;
            }

            UInt128 perSetLimit = DivideRoundUp(totalLimit, requestedSets);
            if (perSetLimit == UInt128.Zero)
            {
                perSetLimit = UInt128.One;
            }

            UInt128 setsNeeded = DivideRoundUp(totalLimit, perSetLimit);
            UInt128 effectiveSets = requestedSets;
            if (setsNeeded < effectiveSets)
            {
                effectiveSets = setsNeeded;
            }

            bool scanCompleted = false;

            if (_useGpuScan)
            {
                _residueGpuTester!.Scan(exponent, twoP, lastIsSeven, perSetLimit, effectiveSets, totalLimit, ref prePrime, ref scanCompleted);
            }
            else
            {
                _residueCpuTester!.Scan(exponent, twoP, lastIsSeven, perSetLimit, effectiveSets, totalLimit, ref prePrime, ref scanCompleted);
            }

            if (!prePrime)
            {
                divisorsExhausted = true;
                return false;
            }

            divisorsExhausted = scanCompleted;
            return true;
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
                                divisorsExhausted = true;
                                return false;
                        }

                        bool lucasPrime = _useGpuLucas
                                ? _lucasLehmerGpuTester.IsPrime(exponent, runOnGpu: true)
                                : _lucasLehmerCpuTester.IsPrime(exponent);
                        divisorsExhausted = true;
                        return lucasPrime;
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

        divisorsExhausted = true;
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
        UInt128 pow = exponent.PowMod(q);
        orders[index] = pow == 1UL ? exponent : 0UL;
    }


}
