using System.Buffers;
using System.Runtime.InteropServices;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

/// <summary>
/// Uses GPU kernels to incrementally scan Mersenne candidates for prime exponents p >= 31.
/// </summary>
public class MersenneNumberIncrementalGpuTester(GpuKernelType kernelType, bool useGpuOrder)
{
    private readonly GpuKernelType _kernelType = kernelType;
    private readonly bool _useGpuOrder = useGpuOrder;

    public void Scan(ulong exponent, UInt128 twoP, LastDigit lastDigit, UInt128 maxK, ref bool isPrime)
    {
        throw new NotImplementedException($"GPU incremental scanning requires the device cycle heuristics implementation (kernel: {_kernelType}, GPU order: {_useGpuOrder}).");

        var gpuLease = GpuKernelPool.GetKernel(_useGpuOrder);
        var execution = gpuLease.EnterExecutionScope();
        var accelerator = gpuLease.Accelerator;
        var stream = gpuLease.Stream;
        int batchSize = GpuConstants.ScanBatchSize; // large batch improves GPU occupancy
        UInt128 kStart = 1UL;
        ulong divMul = (ulong)((((UInt128)1 << 64) - UInt128.One) / exponent) + 1UL;
        byte last = lastDigit == LastDigit.Seven ? (byte)1 : (byte)0; // ILGPU kernels do not support bool parameters

        var pow2Kernel = gpuLease.Pow2ModKernel;
        var incKernel = gpuLease.IncrementalKernel;
        ulong step10 = (exponent.Mod10() << 1).Mod10();
        ulong step8 = ((exponent & 7UL) << 1) & 7UL;
        ulong step3 = ((exponent % 3UL) << 1) % 3UL;
        ulong step5 = ((exponent % 5UL) << 1) % 5UL;
        GpuUInt128 twoPGpu = (GpuUInt128)twoP;
        var smallCyclesView = GpuKernelPool.EnsureSmallCyclesOnDevice(accelerator);
        ResiduePrimeViews primeViews = default;
        if (_kernelType == GpuKernelType.Pow2Mod)
        {
            primeViews = GpuKernelPool.EnsureSmallPrimesOnDevice(accelerator);
        }

        var orderBuffer = accelerator.Allocate1D<ulong>(batchSize);
        // Avoid giant stack allocations that can trigger StackOverflow when batchSize is large.
        // Rent a reusable array from the shared pool instead.
        ulong[] orderArray = ArrayPool<ulong>.Shared.Rent(batchSize);
        UInt128 remaining;
        int currentSize;
        int i;
        UInt128 q = UInt128.Zero;
        try
        {
            while (kStart <= maxK && Volatile.Read(ref isPrime))
            {
                remaining = maxK - kStart + UInt128.One;
                currentSize = remaining > (UInt128)batchSize ? batchSize : (int)remaining;
                Span<ulong> orders = orderArray.AsSpan(0, currentSize);
                // Precompute residue automaton bases for this batch
                UInt128 q0 = twoP * kStart + UInt128.One;
                if (_kernelType == GpuKernelType.Pow2Mod)
                {
                    q0.Mod10_8_5_3(out ulong q0m10, out ulong q0m8, out ulong q0m5, out ulong q0m3);
                    var kernelArgs = new ResidueAutomatonArgs(q0m10, step10, q0m8, step8, q0m3, step3, q0m5, step5);
                    pow2Kernel(
                        stream,
                        currentSize,
                        exponent,
                        twoPGpu,
                        (GpuUInt128)kStart,
                        last,
                        divMul,
                        kernelArgs,
                        orderBuffer.View,
                        smallCyclesView,
                        primeViews.LastOne,
                        primeViews.LastSeven,
                        primeViews.LastOnePow2,
                        primeViews.LastSevenPow2);
                }
                else
                {
                    q0.Mod10_8_5_3(out ulong q0m10, out ulong q0m8, out ulong q0m5, out ulong q0m3);
                    incKernel(
                        stream,
                        currentSize,
                        exponent,
                        twoPGpu,
                        (GpuUInt128)kStart,
                        last,
                        divMul,
                        q0m10,
                        q0m8,
                        q0m3,
                        q0m5,
                        orderBuffer.View,
                        smallCyclesView);
                }

                stream.Synchronize();
                orderBuffer.View.CopyToCPU(ref MemoryMarshal.GetReference(orders), currentSize);
                if (_kernelType == GpuKernelType.Pow2Mod)
                {
                    for (i = 0; i < currentSize; i++)
                    {
                        if (orders[i] != 0UL)
                        {
                            Volatile.Write(ref isPrime, false);
                            break;
                        }
                    }
                }
                else
                {
                    q = twoP * kStart + UInt128.One;
                    for (i = 0; i < currentSize && Volatile.Read(ref isPrime); i++, q += twoP)
                    {
                        if (orders[i] != 0UL && q.IsPrimeCandidate())
                        {
                            Volatile.Write(ref isPrime, false);
                        }
                    }
                }

                kStart += (UInt128)currentSize;
            }
        }
        finally
        {
            ArrayPool<ulong>.Shared.Return(orderArray);
            orderBuffer.Dispose();
            execution.Dispose();
            gpuLease.Dispose();
        }
    }
}
