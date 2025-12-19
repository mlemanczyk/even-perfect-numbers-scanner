using System;
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

public enum CacheStatus
{
    Disabled,
    Enabled,
}

public enum OrderDevice
{
    Cpu,
    Hybrid,
    Gpu,
}

[EnumDependentTemplate(typeof(CalculationMethod))]
[EnumDependentTemplate(typeof(CacheStatus))]
[EnumDependentTemplate(typeof(OrderDevice))]
[DeviceDependentTemplate(typeof(ComputationDevice))]
public sealed class MersenneNumberTesterTemplate(
    GpuKernelType kernelType = GpuKernelType.Incremental,
    ulong maxK = 5_000_000UL)
{
#if CacheStatus_Enabled
    private static readonly ConcurrentDictionary<UInt128, ulong> _orderCache = [];
#endif

#if CalculationMethod_Residue
#if DEVICE_GPU || DEVICE_HYBRID
#if OrderDevice_Gpu || OrderDevice_Hybrid
    private readonly MersenneNumberResidueGpuTester _residueTester = new(true);
#else
    private readonly MersenneNumberResidueGpuTester _residueTester = new(false);
#endif
#else
    private readonly MersenneNumberResidueCpuTester _residueTester = new();
#endif
    private readonly ulong _maxK = maxK;
#elif CalculationMethod_LucasLehmer
#if DEVICE_GPU || DEVICE_HYBRID
#if OrderDevice_Cpu
    private readonly MersenneNumberIncrementalCpuTester _incrementalTester = new(kernelType);
#else
    private readonly MersenneNumberIncrementalGpuTester _incrementalTester = new(kernelType, true);
#endif
    private readonly MersenneNumberLucasLehmerGpuTester _lucasLehmerTester = new();
#else
    private readonly MersenneNumberIncrementalCpuTester _incrementalTester = new(kernelType);
    private readonly MersenneNumberLucasLehmerCpuTester _lucasLehmerTester = new();
#endif
#elif CalculationMethod_Divisor || CalculationMethod_ByDivisor
#if DEVICE_GPU || DEVICE_HYBRID
#if OrderDevice_Cpu
    private readonly MersenneNumberOrderCpuTester _orderTester = new(kernelType);
#else
    private readonly MersenneNumberOrderGpuTester _orderTester = new(kernelType, true);
#endif
#else
    private readonly MersenneNumberOrderCpuTester _orderTester = new(kernelType);
#endif
#else
#if DEVICE_GPU || DEVICE_HYBRID
#if OrderDevice_Cpu
    private readonly MersenneNumberIncrementalCpuTester _incrementalTester = new(kernelType);
#else
    private readonly MersenneNumberIncrementalGpuTester _incrementalTester = new(kernelType, true);
#endif
#else
    private readonly MersenneNumberIncrementalCpuTester _incrementalTester = new(kernelType);
#endif
#endif

    private static readonly Accelerator[] _accelerators = AcceleratorPool.Shared.Accelerators;

    public void WarmUpOrders(ulong exponent, ulong limit = 5_000_000UL)
    {
#if !CacheStatus_Enabled
        return;
#else
        UInt128 twoP = (UInt128)exponent << 1; // 2 * p
        bool lastIsSeven = (exponent & 3UL) == 3UL;

#if OrderDevice_Cpu
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

            _orderCache[q] = q.CalculateOrderCpu();
        }
        return;
#else
#if DEVICE_GPU || DEVICE_HYBRID
        var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
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
            PrimeOrderCalculatorAccelerator.Return(gpu);
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
                    order = qs[offset + i].CalculateOrderGpu(gpu);
                }

                _orderCache[qs[offset + i]] = order;
            }

            offset += count;
            orderBuffer.Dispose();
            qBuffer.Dispose();
        }

        AcceleratorStreamPool.Return(acceleratorIndex,stream);
        PrimeOrderCalculatorAccelerator.Return(gpu);
        uInt128Pool.Return(qs);
#else
        UInt128 kFallback = UInt128.One;
        for (; kFallback <= limit; kFallback++)
        {
            UInt128 qFallback = twoP * kFallback + UInt128.One;
            ulong remainderFallback = qFallback.Mod10();
            bool shouldCheckFallback = lastIsSeven
                    ? remainderFallback == 7UL || remainderFallback == 9UL || remainderFallback == 3UL
                    : remainderFallback == 1UL || remainderFallback == 3UL || remainderFallback == 9UL;

            if (shouldCheckFallback)
            {
                ulong mod8Fallback = qFallback.Mod8();
                if (mod8Fallback != 1UL && mod8Fallback != 7UL)
                {
                    continue;
                }

                if (qFallback.Mod3() == 0UL || qFallback.Mod5() == 0UL)
                {
                    continue;
                }
            }

            if (!shouldCheckFallback || _orderCache.ContainsKey(qFallback))
            {
                continue;
            }

            _orderCache[qFallback] = qFallback.CalculateOrderCpu();
        }
#endif
#endif
#endif
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

#if DEVICE_HYBRID
        if ((exponent & 3UL) == 1UL && exponent.SharesFactorWithExponentMinusOneHybrid(gpu))
        {
            return false;
        }
#elif DEVICE_GPU
        if ((exponent & 3UL) == 1UL && exponent.SharesFactorWithExponentMinusOneGpu(gpu))
        {
            return false;
        }
#else
        if ((exponent & 3UL) == 1UL && exponent.SharesFactorWithExponentMinusOneCpu())
        {
            return false;
        }
#endif

        // Skip even exponents: for even p > 2, M_p is composite; here p is large.
        if ((exponent & 1UL) == 0UL)
        {
            return false;
        }

#if CalculationMethod_Divisor || CalculationMethod_ByDivisor
        // When using order/divisor scanning the theoretical maxK is huge; cache or GPU selection already fixed at compile time.
#endif

        UInt128 twoP = (UInt128)exponent << 1; // 2 * p
        UInt128 maxK = UInt128Numbers.Two.Pow(exponent) / 2 + 1;
        LastDigit lastDigit = (exponent & 3UL) == 3UL ? LastDigit.Seven : LastDigit.One;

#if CalculationMethod_Residue
        bool prePrime = true;
        if (_maxK < maxK)
        {
            maxK = _maxK;
        }

#if DEVICE_GPU || DEVICE_HYBRID
        _residueTester.Scan(gpu, exponent, twoP, lastDigit, maxK, ref prePrime);
#else
        _residueTester.Scan(exponent, twoP, lastDigit, maxK, ref prePrime);
#endif

        return prePrime;
#elif CalculationMethod_LucasLehmer
        bool prePrime = true;
        UInt128 twoPPre = (UInt128)exponent << 1;
        UInt128 maxKPre = (UInt128)PreLucasPreScanK;
        LastDigit lastDigitPre = (exponent & 3UL) == 3UL ? LastDigit.Seven : LastDigit.One;
#if DEVICE_GPU || DEVICE_HYBRID
#if OrderDevice_Cpu
        _incrementalTester.Scan(gpu, exponent, twoPPre, lastDigitPre, maxKPre, ref prePrime);
#else
        _incrementalTester.Scan(exponent, twoPPre, lastDigitPre, maxKPre, ref prePrime);
#endif
        if (!prePrime)
        {
            return false;
        }

        return _lucasLehmerTester.IsPrime(gpu, exponent, runOnGpu: true);
#else
        _incrementalTester.Scan(gpu, exponent, twoPPre, lastDigitPre, maxKPre, ref prePrime);
        if (!prePrime)
        {
            return false;
        }

        return _lucasLehmerTester.IsPrime(gpu, exponent);
#endif
#elif CalculationMethod_Divisor || CalculationMethod_ByDivisor
        bool isPrime = true;
#if DEVICE_GPU || DEVICE_HYBRID
#if OrderDevice_Cpu
        _orderTester.Scan(gpu, exponent, twoP, lastDigit, maxK, ref isPrime);
#else
        _orderTester.Scan(exponent, twoP, lastDigit, maxK, ref isPrime);
#endif
#else
        _orderTester.Scan(gpu, exponent, twoP, lastDigit, maxK, ref isPrime);
#endif
        return isPrime;
#else
        bool isPrime = true;
#if DEVICE_GPU || DEVICE_HYBRID
#if OrderDevice_Cpu
        _incrementalTester.Scan(gpu, exponent, twoP, lastDigit, maxK, ref isPrime);
#else
        _incrementalTester.Scan(exponent, twoP, lastDigit, maxK, ref isPrime);
#endif
#else
        _incrementalTester.Scan(gpu, exponent, twoP, lastDigit, maxK, ref isPrime);
#endif
        return isPrime;
#endif
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