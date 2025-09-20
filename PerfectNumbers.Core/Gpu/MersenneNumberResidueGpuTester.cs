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
        public void Scan(
                        ulong exponent,
                        UInt128 twoP,
                        bool lastIsSeven,
                        UInt128 perSetLimit,
                        UInt128 setCount,
                        UInt128 overallLimit,
                        ref bool isPrime,
                        ref bool divisorsExhausted)
        {
                if (setCount == UInt128.Zero || perSetLimit == UInt128.Zero || overallLimit == UInt128.Zero)
                {
                        return;
                }

		var gpuLease = GpuKernelPool.GetKernel(_useGpuOrder);
		var accelerator = gpuLease.Accelerator;
		// Ensure device has small cycles and primes tables for in-kernel lookup
		var smallCyclesView = GpuKernelPool.EnsureSmallCyclesOnDevice(accelerator);
		ResiduePrimeViews primeViews = GpuKernelPool.EnsureSmallPrimesOnDevice(accelerator);
		int batchSize = GpuConstants.ScanBatchSize;
		byte last = lastIsSeven ? (byte)1 : (byte)0;
		var kernel = gpuLease.Pow2ModKernel;
		ulong step10 = ((exponent % 10UL) << 1) % 10UL;
		ulong step8 = ((exponent & 7UL) << 1) & 7UL;
		ulong step3 = ((exponent % 3UL) << 1) % 3UL;
		ulong step5 = ((exponent % 5UL) << 1) % 5UL;
		GpuUInt128 twoPGpu = (GpuUInt128)twoP;

                var orderBuffer = accelerator.Allocate1D<ulong>(batchSize);
                ulong[] orderArray = ArrayPool<ulong>.Shared.Rent(batchSize);
                UInt128 batchSize128 = (UInt128)batchSize;
                UInt128 limitInclusive = overallLimit == UInt128.MaxValue
                        ? overallLimit
                        : checked(overallLimit + UInt128.One);
                Span<ulong> orders = orderArray.AsSpan(0, batchSize);
                try
                {
                        for (ulong setIndex = 0; setIndex < setCount && Volatile.Read(ref isPrime); setIndex++)
                        {
                                UInt128 setOffset = checked(perSetLimit * (UInt128)setIndex);
                                UInt128 setStart = checked(setOffset + UInt128.One);
                                if (setStart >= limitInclusive)
                                {
                                        break;
                                }

                                UInt128 setLimitExclusive = checked(setOffset + perSetLimit);
                                setLimitExclusive = setLimitExclusive >= limitInclusive
                                        ? limitInclusive
                                        : checked(setLimitExclusive + UInt128.One);
                                if (setLimitExclusive > limitInclusive)
                                {
                                        setLimitExclusive = limitInclusive;
                                }

                                UInt128 kStart = setStart;
                                while (kStart < setLimitExclusive && Volatile.Read(ref isPrime))
                                {
                                        UInt128 remaining = checked(setLimitExclusive - kStart);
                                        int currentSize = remaining > batchSize128 ? batchSize : (int)remaining;
                                        if (currentSize <= 0)
                                        {
                                                break;
                                        }
                                        orders = orderArray.AsSpan(0, currentSize);

                                        UInt128 q = checked(twoP * kStart);
                                        q = checked(q + UInt128.One);
                                        q.Mod10_8_5_3(out ulong q0m10, out ulong q0m8, out ulong q0m5, out ulong q0m3);
                                        var ra = new ResidueAutomatonArgs(q0m10, step10, q0m8, step8, q0m3, step3, q0m5, step5);

                                        kernel(currentSize, exponent, twoPGpu, (GpuUInt128)kStart, last, 0UL,
                                                ra, orderBuffer.View, smallCyclesView, primeViews.LastOne, primeViews.LastSeven, primeViews.LastOnePow2, primeViews.LastSevenPow2);

                                        accelerator.Synchronize();
                                        orderBuffer.View.CopyToCPU(ref MemoryMarshal.GetReference(orders), currentSize);
                                        if (!Volatile.Read(ref isPrime))
                                        {
                                                break;
                                        }

                                        for (int i = 0; i < currentSize; i++)
                                        {
                                                if (orders[i] != 0UL)
                                                {
                                                        Volatile.Write(ref isPrime, false);
                                                        break;
                                                }
                                        }

                                        kStart = checked(kStart + batchSize128);
                                }
                        }
                }
                finally
                {
                        ArrayPool<ulong>.Shared.Return(orderArray);
                        orderBuffer.Dispose();
                        gpuLease.Dispose();
                }

		divisorsExhausted = true;
	}
}
