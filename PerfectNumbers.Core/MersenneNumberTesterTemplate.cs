using System.Runtime.CompilerServices;
using System.Collections.Concurrent;
using System.Buffers;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Cpu;
using PerfectNumbers.Core.Gpu;
using UInt128 = System.UInt128;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core;

public enum GpuKernelType
{
	Incremental,
	Pow2Mod,
}

public enum CalculationMethod
{
	ByDivisor,
	Divisor,
	Incremental,
	LucasLehmer,
	Pow2Mod,
	Residue,
}

[DeviceDependentTemplate(typeof(ComputationDevice))]
public sealed class MersenneNumberTesterTemplate(
	bool useIncremental = true,
	bool useOrderCache = false,
	GpuKernelType kernelType = GpuKernelType.Incremental,
	bool useOrder = false,
	bool useGpuLucas = true,
	bool useGpuScan = true,
	bool useGpuOrder = true,
	bool useResidue = true,
	ulong maxK = 5_000_000UL,
	ComputationDevice orderDevice = ComputationDevice.Gpu)
{
	private readonly bool _useResidue = useResidue;
	private readonly bool _useIncremental = useIncremental && !useResidue;
	private readonly ulong _maxK = maxK;
	private readonly Func<PrimeOrderCalculatorAccelerator, ulong, bool> _sharesFactorWithExponentMinusOne = orderDevice switch
	{
		ComputationDevice.Hybrid => static (gpu, exponent) => exponent.SharesFactorWithExponentMinusOneHybrid(gpu),
		_ when useGpuScan => static (gpu, exponent) => exponent.SharesFactorWithExponentMinusOneGpu(gpu),
		_ => static (_, exponent) => exponent.SharesFactorWithExponentMinusOneCpu(),
	};

	private static readonly ConcurrentDictionary<UInt128, ulong> _orderCache = [];
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

	private static readonly Accelerator[] _accelerators = AcceleratorPool.Shared.Accelerators;

	public void WarmUpOrders(ulong exponent, ulong limit = 5_000_000UL)
	{
		if (!useOrderCache)
		{
			return;
		}

		UInt128 twoP = (UInt128)exponent << 1; // 2 * p
		bool lastIsSeven = (exponent & 3UL) == 3UL;
#if DEVICE_GPU || DEVICE_HYBRID
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
#endif
		if (!useGpuOrder)
		{
			UInt128 k = UInt128.One;
			for (; k <= limit; k++)
			{
				UInt128 q = twoP * k + UInt128.One;
				ulong remainder = q.Mod10();
				bool shouldCheck = lastIsSeven
						? remainder == 7UL || remainder == 9UL || remainder == 3UL
						: remainder == 1UL || remainder == 3UL || remainder == 9UL;

				if (shouldCheck)
				{
					ulong mod8 = q.Mod8();
					if (mod8 != 1UL && mod8 != 7UL)
					{
						continue;
					}

					if (q.Mod3() == 0UL || q.Mod5() == 0UL)
					{
						continue;
					}
				}

				if (!shouldCheck || _orderCache.ContainsKey(q))
				{
					continue;
				}

				_orderCache[q] =
#if DEVICE_GPU
					q.CalculateOrderGpu(gpu);
#elif DEVICE_HYBRID
					q.CalculateOrderHybrid(gpu);
#else
					q.CalculateOrderCpu();
#endif
			}
#if DEVICE_GPU || DEVICE_HYBRID
			PrimeOrderCalculatorAccelerator.Return(gpu);
#endif
			return;
		}

		ArrayPool<UInt128> uInt128Pool = ThreadStaticPools.UInt128Pool;
		UInt128[] qsBuffer = uInt128Pool.Rent((int)limit);
		int idx = 0;
		UInt128 k2 = 1UL;
		for (; k2 <= limit; k2++)
		{
			UInt128 q2 = twoP * k2 + 1UL;
			ulong remainder2 = q2.Mod10();
			bool shouldCheck2 = lastIsSeven
					? remainder2 == 7UL || remainder2 == 9UL || remainder2 == 3UL
					: remainder2 == 1UL || remainder2 == 3UL || remainder2 == 9UL;

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

			if (!shouldCheck2 || _orderCache.ContainsKey(q2))
			{
				continue;
			}

			qsBuffer[idx++] = q2;
		}

		if (idx == 0)
		{
#if DEVICE_GPU || DEVICE_HYBRID
			PrimeOrderCalculatorAccelerator.Return(gpu);
#endif
			ThreadStaticPools.UInt128Pool.Return(qsBuffer);
			return;
		}

		UInt128[] qs = qsBuffer;

		var acceleratorIndex = AcceleratorPool.Shared.Rent();
		var accelerator = _accelerators[acceleratorIndex];
		var stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		var orderKernel = GpuKernelPool.GetOrAddKernels(acceleratorIndex, stream, KernelType.OrderKernelScan).Order!;

		int batchSize = GpuConstants.ScanBatchSize;
		ulong divMul = (ulong)((((UInt128)1 << 64) - 1UL) / exponent) + 1UL;
		int offset = 0;
		var gpuUInt128Pool = ThreadStaticPools.GpuUInt128Pool;
		MemoryBuffer1D<GpuUInt128, Stride1D.Dense>? qBuffer;
		MemoryBuffer1D<ulong, Stride1D.Dense>? orderBuffer;
		
		while (offset < idx)
		{
			int count = Math.Min(batchSize, idx - offset);
			qBuffer = accelerator.Allocate1D<GpuUInt128>(count);
			orderBuffer = accelerator.Allocate1D<ulong>(count);
			
			var tmp = gpuUInt128Pool.Rent(count);
			for (int i = 0; i < count; i++)
			{
				tmp[i] = (GpuUInt128)qs[offset + i];
			}
			
			qBuffer.View.CopyFromCPU(stream, ref tmp[0], count);
			gpuUInt128Pool.Return(tmp);
			orderKernel(stream, count, exponent, divMul, qBuffer.View, orderBuffer.View);
			stream.Synchronize();

			ulong[] orders = orderBuffer.GetAsArray1D();
			for (int i = 0; i < count; i++)
			{
				ulong order = orders[i];
				if (order == 0UL)
				{
					order =
#if DEVICE_GPU
						qs[offset + i].CalculateOrderGpu(gpu);
#elif DEVICE_HYBRID
						qs[offset + i].CalculateOrderHybrid(gpu);
#else
						qs[offset + i].CalculateOrderCpu();
#endif
				}

				if (useOrderCache)
				{
					_orderCache[qs[offset + i]] = order;
				}
			}

			offset += count;
			orderBuffer.Dispose();
			qBuffer.Dispose();
		}

		AcceleratorStreamPool.Return(acceleratorIndex,stream);
#if DEVICE_GPU || DEVICE_HYBRID
		PrimeOrderCalculatorAccelerator.Return(gpu);
#endif
		uInt128Pool.Return(qs);
	}

	public bool IsMersennePrime(PrimeOrderCalculatorAccelerator gpu, ulong exponent)
	{
		// Safe early rejections using known orders for tiny primes:
		// 7 | M_p iff 3 | p. Avoid rejecting p == 3 where M_3 == 7 is prime.
		if ((exponent % 3UL) == 0UL && exponent != 3UL)
		{
			// TODO: Replace this `% 3` check with ULongExtensions.Mod3 to align the early rejection with
			// the benchmarked bitmask helper instead of generic modulo for CPU workloads.
			return false;
		}

		if ((exponent & 3UL) == 1UL && _sharesFactorWithExponentMinusOne(gpu, exponent))
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
		LastDigit lastDigit = (exponent & 3UL) == 3UL ? LastDigit.Seven : LastDigit.One;

		// In residue mode the residue scan is the final primality check.
		if (_useResidue)
		{
			if (_maxK < maxK)
			{
				maxK = _maxK;
			}

			if (useGpuScan)
			{
				_residueGpuTester!.Scan(gpu, exponent, twoP, lastDigit, maxK, ref prePrime);
			}
			else
			{
				_residueCpuTester!.Scan(exponent, twoP, lastDigit, maxK, ref prePrime);
			}

			return prePrime;
		}

		if (!_useIncremental)
		{
			// Optional prefilter before Lucas–Lehmer using the same GPU/CPU scan path.
			// Scan a limited range of k to quickly reject obvious cases.
			UInt128 twoPPre = (UInt128)exponent << 1;
			UInt128 maxKPre = (UInt128)PreLucasPreScanK;
			LastDigit lastDigitPre = (exponent & 3UL) == 3UL ? LastDigit.Seven : LastDigit.One;
			if (useGpuScan)
			{
				_incrementalGpuTester.Scan(exponent, twoPPre, lastDigitPre, maxKPre, ref prePrime);
			}
			else
			{
				_incrementalCpuTester.Scan(gpu, exponent, twoPPre, lastDigitPre, maxKPre, ref prePrime);
			}

			if (!prePrime)
			{
				return false;
			}

			return useGpuLucas
				? _lucasLehmerGpuTester.IsPrime(gpu, exponent, runOnGpu: true)
				: _lucasLehmerCpuTester.IsPrime(gpu, exponent);
		}

		bool isPrime = true;

		if (useOrder)
		{
			if (useGpuOrder)
			{
				_orderGpuTester.Scan(exponent, twoP, lastDigit, maxK, ref isPrime);
			}
			else
			{
				_orderCpuTester.Scan(gpu, exponent, twoP, lastDigit, maxK, ref isPrime);
			}
		}
		else
		{
			if (useGpuScan)
			{
				_incrementalGpuTester.Scan(exponent, twoP, lastDigit, maxK, ref isPrime);
			}
			else
			{
				_incrementalCpuTester.Scan(gpu, exponent, twoP, lastDigit, maxK, ref isPrime);
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
		UInt128 pow = exponent.PowMod(q);
		orders[index] = pow == 1UL ? exponent : 0UL;
	}
}
