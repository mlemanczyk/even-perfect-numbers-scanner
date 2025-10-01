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
		var gpuLease = GpuKernelPool.GetKernel(_useGpuOrder);
		var execution = gpuLease.EnterExecutionScope();
		var accelerator = gpuLease.Accelerator;
		var stream = gpuLease.Stream;
		int batchSize = GpuConstants.ScanBatchSize; // large batch improves GPU occupancy
		UInt128 kStart = 1UL;
		ulong divMul = (ulong)((((UInt128)1 << 64) - UInt128.One) / exponent) + 1UL;
		byte last = lastIsSeven ? (byte)1 : (byte)0; // ILGPU kernels do not support bool parameters

                var pow2Kernel = gpuLease.Pow2ModKernel; // TODO: Route this binding to the ProcessEightBitWindows kernel once the scalar helper lands so GPU incremental scans inherit the windowed speedup from GpuPow2ModBenchmarks.
		var incKernel = gpuLease.IncrementalKernel;
		exponent.Mod10_8_5_3Steps(out ulong step10, out ulong step8, out ulong step5, out ulong step3);
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
					pow2Kernel(stream, currentSize, exponent, twoPGpu, (GpuUInt128)kStart, last, divMul,
							kernelArgs, orderBuffer.View, smallCyclesView, primeViews.LastOne, primeViews.LastSeven,
							primeViews.LastOnePow2, primeViews.LastSevenPow2);
				}
				else
				{
					q0.Mod10_8_5_3(out ulong q0m10, out ulong q0m8, out ulong q0m5, out ulong q0m3);
					incKernel(stream, currentSize, exponent, twoPGpu, (GpuUInt128)kStart, last, divMul,
							q0m10, q0m8, q0m3, q0m5, orderBuffer.View, smallCyclesView);
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
