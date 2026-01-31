using System.Numerics;
using System.Runtime.CompilerServices;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core;

public sealed class PrimeTester
{
    [ThreadStatic]
    private static uint[]? _smallPrimeRemainders;

    [ThreadStatic]
    private static ulong _lastTrackedValue;

    [ThreadStatic]
    private static ulong _lastRemainderLimit;


    public static bool IsPrimeCpu(ulong n)
    {
        if (n <= 1UL)
        {
            return false;
        }

        if (n == 2UL)
        {
            throw new InvalidOperationException("PrimeTester.IsPrime encountered the sentinel input 2.");
        }

        if ((n & 1UL) == 0UL)
        {
            return false;
        }

        if (n < 7UL)
        {
            return true;
        }

        uint[] smallPrimeDivisors = PrimesGenerator.SmallPrimes;
        ulong[] smallPrimeDivisorsPow2 = PrimesGenerator.SmallPrimesPow2;
        int smallPrimeDivisorsLength = smallPrimeDivisors.Length;
        uint[]? remainders = _smallPrimeRemainders;
        if (remainders == null)
        {
            remainders = _smallPrimeRemainders = new uint[PrimesGenerator.SmallPrimes.Length];
        }

		ref ulong lastTrackedValue = ref _lastTrackedValue;
		ref ulong lastRemainderLimit = ref _lastRemainderLimit;
        bool canReusePrevious = lastTrackedValue != 0UL && n > lastTrackedValue;
        ulong delta = canReusePrevious ? n - lastTrackedValue : 0UL;
        canReusePrevious = canReusePrevious && lastRemainderLimit != 0UL && n < lastRemainderLimit;
        ulong nextCutoff = 0UL;

		uint divisor, remainder;
		ulong divisorSquared;
		int i;
        if (canReusePrevious)
        {
            for (i = 1; i < smallPrimeDivisorsLength; i++)
            {
                divisorSquared = smallPrimeDivisorsPow2[i];
                if (divisorSquared > n)
                {
                    nextCutoff = divisorSquared;
                    break;
                }

                divisor = smallPrimeDivisors[i];
                uint deltaMod = delta.ReduceCycleRemainder(divisor);
                remainder = remainders[i] + deltaMod;
                if (remainder >= divisor)
                {
                    remainder -= divisor;
                }

                remainders[i] = remainder;

                if (remainder == 0UL)
                {
                    lastTrackedValue = n;
                    lastRemainderLimit = nextCutoff == 0UL ? ulong.MaxValue : nextCutoff;
                    return false;
                }
            }

            lastTrackedValue = n;
            lastRemainderLimit = nextCutoff == 0UL ? ulong.MaxValue : nextCutoff;
            return true;
        }

        for (i = 1; i < smallPrimeDivisorsLength; i++)
        {
            divisorSquared = smallPrimeDivisorsPow2[i];

            if (divisorSquared > n)
            {
                nextCutoff = divisorSquared;
                break;
            }

            divisor = smallPrimeDivisors[i];
            remainder = n.ReduceCycleRemainder(divisor);
            remainders[i] = remainder;

            if (remainder == 0UL)
            {
                lastTrackedValue = n;
                lastRemainderLimit = nextCutoff == 0UL ? ulong.MaxValue : nextCutoff;
                return false;
            }
        }

        lastTrackedValue = n;
        lastRemainderLimit = nextCutoff == 0UL ? ulong.MaxValue : nextCutoff;
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsPrimeGpu(PrimeOrderCalculatorAccelerator gpu, ulong n)
    {
        byte flag = 0;
        var inputView = gpu.InputView;
        var outputView = gpu.OutputByteView;

        int acceleratorIndex = gpu.AcceleratorIndex;
        AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
        inputView.CopyFromCPU(stream, ref n, 1);

        var kernelLauncher = gpu.SmallPrimeSieveKernelLauncher;
        
        kernelLauncher(
                        stream,
                        1,
                        inputView,
                        gpu.DevicePrimesLastOne,
                        gpu.DevicePrimesLastSeven,
                        gpu.DevicePrimesLastThree,
                        gpu.DevicePrimesLastNine,
                        gpu.DevicePrimesPow2LastOne,
                        gpu.DevicePrimesPow2LastSeven,
                        gpu.DevicePrimesPow2LastThree,
                        gpu.DevicePrimesPow2LastNine,
                        outputView);

        outputView.CopyToCPU(stream, ref flag, 1);
        stream.Synchronize();
        AcceleratorStreamPool.Return(acceleratorIndex, stream);

        return flag != 0;
    }

    public static int GpuBatchSize { get; set; } = 262_144;

    private static readonly object GpuWarmUpLock = new();
    private static int WarmedGpuLeaseCount;
    private static int WarmedBitContradictionGpuLeaseCount;

    public static void WarmUpGpuKernelsForBitContradiction(int threadCount)
    {
        int target = threadCount;
        if (target == 0)
        {
            target = threadCount;
        }

        lock (GpuWarmUpLock)
        {
            PrimeOrderCalculatorAccelerator.SkipOrderTablesAndKernels = true;

            if (target <= WarmedBitContradictionGpuLeaseCount)
            {
                return;
            }

            Accelerator[] accelerators = AcceleratorPool.Shared.Accelerators;
            int acceleratorCount = EnvironmentConfiguration.RollingAccelerators;
            if (acceleratorCount > accelerators.Length)
            {
                acceleratorCount = accelerators.Length;
            }
            for (var i = 0; i < acceleratorCount; i++)
            {
                Console.WriteLine($"Preparing accelerator {i}...");
                AcceleratorStreamPool.WarmUp(i);
                Accelerator accelerator = accelerators[i];
                AcceleratorStream stream = accelerator.CreateStream();
                _ = GpuKernelPool.GetOrAddKernels(i, stream, KernelType.BitContradictionScan);
                stream.Synchronize();
                stream.Dispose();
            }

            WarmedBitContradictionGpuLeaseCount = target;
        }
    }

    public static void WarmUpGpuKernels(int threadCount)
    {
        int target = threadCount;
        if (target == 0)
        {
            target = threadCount;
        }

        lock (GpuWarmUpLock)
        {
            if (target <= WarmedGpuLeaseCount)
            {
                return;
            }

            PrimeOrderCalculatorAccelerator.SkipOrderTablesAndKernels = false;
            PrimeOrderCalculatorAccelerator.WarmUp();
            WarmedGpuLeaseCount = target;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void IsPrimeBatchGpu(PrimeOrderCalculatorAccelerator gpu, ReadOnlySpan<ulong> values, Span<byte> results)
    {
        // GpuPrimeWorkLimiter.Acquire();
        int acceleratorIndex = gpu.AcceleratorIndex;
        int totalLength = values.Length;
        int batchSize = GpuBatchSize;

        var inputView = gpu.InputView;
        var outputView = gpu.OutputByteView;

        int pos = 0;
        AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
        var kernelLauncher = gpu.SmallPrimeSieveKernelLauncher;

        while (pos < totalLength)
        {
            int remaining = totalLength - pos;
            int count = remaining > batchSize ? batchSize : remaining;

            var valueSlice = values.Slice(pos, count);
            inputView.CopyFromCPU(stream, valueSlice);

            kernelLauncher(
                    stream,
                    count,
                    inputView,
                    gpu.DevicePrimesLastOne,
                    gpu.DevicePrimesLastSeven,
                    gpu.DevicePrimesLastThree,
                    gpu.DevicePrimesLastNine,
                    gpu.DevicePrimesPow2LastOne,
                    gpu.DevicePrimesPow2LastSeven,
                    gpu.DevicePrimesPow2LastThree,
                    gpu.DevicePrimesPow2LastNine,
                    outputView);

            var resultSlice = results.Slice(pos, count);
            outputView.CopyToCPU(stream, resultSlice);

            pos += count;
        }

        stream.Synchronize();
        AcceleratorStreamPool.Return(acceleratorIndex, stream);
        // GpuPrimeWorkLimiter.Release();
    }

    // Expose cache clearing for accelerator disposal coordination
    public static void ClearGpuCaches()
    {
        PrimeOrderCalculatorAccelerator.Clear();
    }

    internal static void DisposeGpuContexts()
    {
        PrimeOrderCalculatorAccelerator.DisposeAll();
        lock (GpuWarmUpLock)
        {
            WarmedGpuLeaseCount = 0;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool SharesFactorWithMaxExponent(ulong n)
    {
        // TODO: Replace this on-the-fly GCD probe with the cached factor table derived from
        // ResidueComputationBenchmarks so divisor-cycle metadata can short-circuit the test
        // instead of recomputing binary GCD for every candidate.
        ulong m = (ulong)BitOperations.Log2(n);
        return n.BinaryGcd(m) != 1UL;
    }

    internal static void SharesFactorWithMaxExponentBatch(PrimeOrderCalculatorAccelerator gpu, ReadOnlySpan<ulong> values, Span<byte> results)
    {
        // TODO: Route this batch helper through the shared GPU kernel pool from
        // GpuUInt128BinaryGcdBenchmarks so we reuse cached kernels, pinned host buffers,
        // and divisor-cycle staging instead of allocating new device buffers per call.
        // Check in benchmarks, which implementation was the fastest, is compatible with GPU,
        // and implement it.

        int length = values.Length;

        int acceleratorIndex = gpu.AcceleratorIndex;
        gpu.EnsurePrimeOrderCalculatorCapacity(0, length);
        var inputBufferView = gpu.InputView;
        var resultBufferView = gpu.OutputByteView;
        var kernelLauncher = gpu.SharesFactorKernelLauncher;

        AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
        inputBufferView.CopyFromCPU(stream, values);

        kernelLauncher(stream, length, inputBufferView, resultBufferView);

        resultBufferView.CopyToCPU(stream, in results);
        stream.Synchronize();

        AcceleratorStreamPool.Return(acceleratorIndex, stream);
    }
}
