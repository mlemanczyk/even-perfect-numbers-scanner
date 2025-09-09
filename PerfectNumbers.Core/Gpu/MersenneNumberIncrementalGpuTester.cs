using System.Buffers;
using System.Runtime.InteropServices;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public class MersenneNumberIncrementalGpuTester(GpuKernelType kernelType, bool useGpuOrder)
{
	private readonly GpuKernelType _kernelType = kernelType;
	private readonly bool _useGpuOrder = useGpuOrder;

	public void Scan(ulong exponent, UInt128 twoP, bool lastIsSeven, UInt128 maxK, ref bool isPrime)
	{
		using var gpuLease = GpuKernelPool.GetKernel(_useGpuOrder);
		var accelerator = gpuLease.Accelerator;
		int batchSize = GpuConstants.ScanBatchSize; // large batch improves GPU occupancy
		UInt128 kStart = 1UL;
		ulong divMul = (ulong)((((UInt128)1 << 64) - UInt128.One) / exponent) + 1UL;
		byte last = lastIsSeven ? (byte)1 : (byte)0; // ILGPU kernels do not support bool parameters

        var kernel = (_kernelType == GpuKernelType.Pow2Mod)
            ? gpuLease.Pow2ModKernel
            : gpuLease.IncrementalKernel;

		using var orderBuffer = accelerator.Allocate1D<ulong>(batchSize);
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
				ulong q0m10 = q0.Mod10();
				ulong q0m8 = q0.Mod8();
				ulong q0m3 = q0.Mod3();
				ulong q0m5 = q0.Mod5();
				ulong step10 = ((exponent % 10UL) << 1) % 10UL;
				ulong step8 = ((exponent & 7UL) << 1) & 7UL;
				ulong step3 = ((exponent % 3UL) << 1) % 3UL;
				ulong step5 = ((exponent % 5UL) << 1) % 5UL;

                var smallCyclesView = GpuKernelPool.EnsureSmallCyclesOnDevice(accelerator);
                kernel(currentSize, exponent, (GpuUInt128)twoP, (GpuUInt128)kStart, last, divMul,
                    q0m10, q0m8, q0m3, q0m5, orderBuffer.View, smallCyclesView);

				accelerator.Synchronize();
				orderBuffer.View.CopyToCPU(ref MemoryMarshal.GetReference(orders), currentSize);
				q = twoP * kStart + UInt128.One;
				for (i = 0; i < currentSize && Volatile.Read(ref isPrime); i++, q += twoP)
				{
					if (orders[i] != 0UL)
					{
						if (q.IsPrimeCandidate())
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
		}
	}
}
