using System;
using System.Buffers;
using System.Runtime.InteropServices;
using System.Threading;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public class MersenneNumberResidueGpuTester(bool useGpuOrder)
{
    private readonly bool _useGpuOrder = useGpuOrder;
    // TODO: Integrate MersenneDivisorCycles.Shared to consult cycle lengths for small q (<= 4M) and fast-reject
    // candidates without launching heavy kernels. Consider a device-visible constant buffer per-accelerator
    // via GpuKernelPool to avoid host round-trips.
    // TODO: When cycles are missing for larger q values, compute only the single required cycle on the device
    // requested by the caller, skip queuing additional block generation, and keep the snapshot cache untouched.

    // GPU residue variant: check 2^p % q == 1 for q = 2*p*k + 1.
    public void Scan(ulong exponent, UInt128 twoP, bool lastIsSeven, UInt128 maxK, ref bool isPrime)
    {
        var gpuLease = GpuKernelPool.GetKernel(_useGpuOrder);
        var accelerator = gpuLease.Accelerator;
        var stream = gpuLease.Stream;
        ResiduePrimeViews primeViews = GpuKernelPool.EnsureSmallPrimesOnDevice(accelerator);
        int batchSize = GpuConstants.ScanBatchSize;
        UInt128 kStart = 1UL;
        UInt128 limit = maxK + UInt128.One;
        byte last = lastIsSeven ? (byte)1 : (byte)0;
        var kernel = gpuLease.Pow2ModWindowedKernel;
        if (exponent > uint.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(exponent), "Pow2ModWindowedKernel stores order sentinels in 32-bit buffers.");
        }
        twoP.Mod10_8_5_3(out ulong step10, out ulong step8, out ulong step5, out ulong step3);
        step10 = step10.Mod10();

        GpuUInt128 twoPGpu = (GpuUInt128)twoP;

        var orderBuffer = accelerator.Allocate1D<uint>(batchSize);
        // Device memory returned by Allocate1D is not guaranteed to be zeroed.
        // Ensure the buffer starts with a known state so that lanes skipped by the
        // kernel (for example due to early rejections) do not leave stale garbage
        // values that would be interpreted as composite witnesses when copied back
        // to the host. This mirrors the explicit zeroing performed by other GPU
        // testers such as the order kernels.
        orderBuffer.MemSetToZero();
        uint[] orderArray = ArrayPool<uint>.Shared.Rent(batchSize);
        UInt128 batchSize128 = (UInt128)batchSize;
        UInt128 fullBatchStep = twoP * batchSize128;
        UInt128 q = twoP * kStart + UInt128.One;
        // TODO: Replace the direct UInt128 multiply/add updates below with the residue-cycle stepping helper so that q advances reuse cached cycle increments rather than recomputing twoP multiples on every iteration.

        try
        {
            while (kStart < limit && Volatile.Read(ref isPrime))
            {
                UInt128 remaining = limit - kStart;
                int currentSize = remaining > batchSize128 ? batchSize : (int)remaining;
                Span<uint> orders = orderArray.AsSpan(0, currentSize);
                ref uint ordersRef = ref MemoryMarshal.GetReference(orders);

                q.Mod10_8_5_3(out ulong q0m10, out ulong q0m8, out ulong q0m5, out ulong q0m3);
                var kernelArgs = new ResidueAutomatonArgs(q0m10, step10, q0m8, step8, q0m3, step3, q0m5, step5);

                kernel(stream, currentSize, exponent, twoPGpu, (GpuUInt128)kStart, last, 0UL,
                        kernelArgs, orderBuffer.View, primeViews.LastOne, primeViews.LastSeven, primeViews.LastOnePow2, primeViews.LastSevenPow2);

                accelerator.Synchronize();
                orderBuffer.View.CopyToCPU(ref ordersRef, currentSize);
                if (!Volatile.Read(ref isPrime))
                {
                    break;
                }

                bool compositeFound = false;
                for (int i = 0; i < currentSize; i++)
                {
                    if (orders[i] != 0U)
                    {
                        Volatile.Write(ref isPrime, false);
                        compositeFound = true;
                        break;
                    }
                }

                if (compositeFound)
                {
                    break;
                }

                UInt128 processed = (UInt128)currentSize;
                kStart += processed;
                if (kStart < limit)
                {
                    if (processed == batchSize128)
                    {
                        q += fullBatchStep;
                    }
                    else
                    {
                        q += twoP * processed; // TODO: Swap this multiplication for the shared cycle stepping helper once residue cycles are mandatory so we can reuse the cached remainder ladder instead of performing UInt128 multiply operations.
                    }
                }
            }
        }
        finally
        {
            ArrayPool<uint>.Shared.Return(orderArray);
            orderBuffer.Dispose();
            gpuLease.Dispose();
        }
    }
}
