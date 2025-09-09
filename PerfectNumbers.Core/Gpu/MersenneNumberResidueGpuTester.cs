using System.Buffers;
using System.Runtime.InteropServices;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public class MersenneNumberResidueGpuTester(bool useGpuOrder)
{
    private readonly bool _useGpuOrder = useGpuOrder;
    // TODO: Integrate MersenneDivisorCycles.Shared to consult cycle lengths for small q (<= 4M) and fast-reject
    // candidates without launching heavy kernels. Consider a device-visible constant buffer per-accelerator
    // via GpuKernelPool to avoid host round-trips.

	// GPU residue variant: check 2^p % q == 1 for q = 2*p*k + 1.
    public void Scan(ulong exponent, UInt128 twoP, bool lastIsSeven, UInt128 maxK, ref bool isPrime)
    {
        var gpuLease = GpuKernelPool.GetKernel(_useGpuOrder);
        var accelerator = gpuLease.Accelerator;
        // Ensure device has small cycles table for in-kernel lookup
        var smallCyclesView = GpuKernelPool.EnsureSmallCyclesOnDevice(accelerator);
        int batchSize = GpuConstants.ScanBatchSize;
                UInt128 kStart = 1UL;
                byte last = lastIsSeven ? (byte)1 : (byte)0;
                var kernel = gpuLease.Pow2ModKernel;

                var orderBuffer = accelerator.Allocate1D<ulong>(batchSize);
                ulong[] orderArray = ArrayPool<ulong>.Shared.Rent(batchSize);
                UInt128 remaining;
                int currentSize;
                int i;
                UInt128 batchSize128 = (UInt128)batchSize,
                                q = UInt128.Zero;
                Span<ulong> orders = orderArray.AsSpan(0, batchSize);
        try
        {
            while (kStart < maxK && Volatile.Read(ref isPrime))
            {
                                remaining = maxK - kStart;
                                if (remaining > batchSize128)
                                {
                                        currentSize = batchSize;
                                }
                                else
                                {
                                        currentSize = (int)remaining;
                                        orders = orderArray.AsSpan(0, currentSize);
                                }

                                q = twoP * kStart + UInt128.One;
                                q.Mod10_8_5_3(out ulong q0m10, out ulong q0m8, out ulong q0m5, out ulong q0m3);
                                ulong step10 = ((exponent % 10UL) << 1) % 10UL;
                                ulong step8 = ((exponent & 7UL) << 1) & 7UL;
                                ulong step3 = ((exponent % 3UL) << 1) % 3UL;
                                ulong step5 = ((exponent % 5UL) << 1) % 5UL;

                kernel(currentSize, exponent, (GpuUInt128)twoP, (GpuUInt128)kStart, last, 0UL,
                    q0m10, q0m8, q0m3, q0m5, orderBuffer.View, smallCyclesView);

                                accelerator.Synchronize();
                                orderBuffer.View.CopyToCPU(ref MemoryMarshal.GetReference(orders), currentSize);
                                if (!Volatile.Read(ref isPrime))
                                {
                                        break;
                                }

                for (i = 0; i < currentSize; i++, q += twoP)
                {
                    if (orders[i] != 0UL && q.IsPrimeCandidate())
                    {
                        Volatile.Write(ref isPrime, false);
                        break;
                    }
                }

                                kStart += batchSize128;
                        }
                }
                finally
                {
                        ArrayPool<ulong>.Shared.Return(orderArray);
                }

                orderBuffer.Dispose();
                gpuLease.Dispose();

    }
}
