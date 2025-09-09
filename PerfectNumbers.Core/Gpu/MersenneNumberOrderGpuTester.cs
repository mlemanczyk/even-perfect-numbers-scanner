using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public class MersenneNumberOrderGpuTester(GpuKernelType kernelType, bool useGpuOrder)
{
	private readonly GpuKernelType _kernelType = kernelType;
	private readonly bool _useGpuOrder = useGpuOrder;

	public void Scan(ulong exponent, UInt128 twoP, bool lastIsSeven, UInt128 maxK, ref bool isPrime)
	{
		using var gpuLease = GpuKernelPool.GetKernel(_useGpuOrder);
		var accelerator = gpuLease.Accelerator;
		int batchSize = GpuConstants.ScanBatchSize; // large batch improves GPU occupancy and avoids TDR
		UInt128 kStart = UInt128.One;
		ulong divMul = (ulong)((((UInt128)1 << 64) - UInt128.One) / exponent) + 1UL;
		byte last = lastIsSeven ? (byte)1 : (byte)0; // ILGPU kernels do not support bool parameters

        var kernel = _kernelType switch
        {
            GpuKernelType.Pow2Mod => gpuLease.Pow2ModOrderKernel,
            _ => gpuLease.IncrementalOrderKernel,
        };

		using var foundBuffer = accelerator.Allocate1D<int>(1);
		UInt128 remaining;
		int currentSize;
		int found = 0;
		while (kStart <= maxK && Volatile.Read(ref isPrime))
		{
			remaining = maxK - kStart + 1UL;
			currentSize = remaining > (UInt128)batchSize ? batchSize : (int)remaining;
			foundBuffer.MemSetToZero();
			// Precompute residue automaton bases for this batch
			UInt128 q0 = twoP * kStart + 1UL;
			ulong q0m10 = q0.Mod10();
			ulong q0m8 = q0.Mod8();
			ulong q0m3 = q0.Mod3();
			ulong q0m5 = q0.Mod5();
			ulong step10 = ((exponent % 10UL) << 1) % 10UL;
			ulong step8 = ((exponent & 7UL) << 1) & 7UL;
			ulong step3 = ((exponent % 3UL) << 1) % 3UL;
			ulong step5 = ((exponent % 5UL) << 1) % 5UL;
			var ra = new ResidueAutomatonArgs(q0m10, step10, q0m8, step8, q0m3, step3, q0m5, step5);


			// Ensure device has small cycles table for early rejections in order kernels
            var smallCyclesView = GpuKernelPool.EnsureSmallCyclesOnDevice(accelerator);
            kernel(currentSize, exponent, (GpuUInt128)twoP, (GpuUInt128)kStart, last, divMul,
                ra, foundBuffer.View, smallCyclesView);

			accelerator.Synchronize();
			foundBuffer.View.CopyToCPU(ref found, 1);
			if (found != 0)
			{
				Volatile.Write(ref isPrime, false);
				break;
			}

			kStart += (UInt128)currentSize;
		}
	}
}
