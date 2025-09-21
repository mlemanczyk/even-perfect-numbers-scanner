using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public class MersenneNumberOrderGpuTester(GpuKernelType kernelType, bool useGpuOrder)
{
	private readonly GpuKernelType _kernelType = kernelType;
	private readonly bool _useGpuOrder = useGpuOrder;

	public void Scan(ulong exponent, UInt128 twoP, bool lastIsSeven, UInt128 maxK, ref bool isPrime)
	{
		var gpuLease = GpuKernelPool.GetKernel(_useGpuOrder);
		var execution = gpuLease.EnterExecutionScope();
		var accelerator = gpuLease.Accelerator;
		var stream = gpuLease.Stream;
		int batchSize = GpuConstants.ScanBatchSize; // large batch improves GPU occupancy and avoids TDR
		UInt128 kStart = UInt128.One;
		ulong divMul = (ulong)((((UInt128)1 << 64) - UInt128.One) / exponent) + 1UL;
		byte last = lastIsSeven ? (byte)1 : (byte)0; // ILGPU kernels do not support bool parameters

		var kernel = _kernelType switch
		{
			GpuKernelType.Pow2Mod => gpuLease.Pow2ModOrderKernel,
			_ => gpuLease.IncrementalOrderKernel,
		};

		var foundBuffer = accelerator.Allocate1D<int>(1);
		UInt128 exponent128 = exponent;
		exponent128.Mod10_8_5_3(out ulong exponentMod10, out ulong exponentMod8, out ulong exponentMod5, out ulong exponentMod3);
		ulong step10 = (exponentMod10 << 1).Mod10();
		ulong step8 = (exponentMod8 << 1) & 7UL;
		ulong step3 = (exponentMod3 << 1) % 3UL;
		ulong step5 = (exponentMod5 << 1) % 5UL;
		try
		{
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
				q0.Mod10_8_5_3(out ulong q0m10, out ulong q0m8, out ulong q0m5, out ulong q0m3);
				var kernelArgs = new ResidueAutomatonArgs(q0m10, step10, q0m8, step8, q0m3, step3, q0m5, step5);


				// Ensure device has small cycles table for early rejections in order kernels
				var smallCyclesView = GpuKernelPool.EnsureSmallCyclesOnDevice(accelerator);
				kernel(stream, currentSize, exponent, (GpuUInt128)twoP, (GpuUInt128)kStart, last, divMul,
						kernelArgs, foundBuffer.View, smallCyclesView);

				stream.Synchronize();
				foundBuffer.View.CopyToCPU(ref found, 1);
				if (found != 0)
				{
					Volatile.Write(ref isPrime, false);
					break;
				}

				kStart += (UInt128)currentSize;
			}
		}
		finally
		{
			foundBuffer.Dispose();
			execution.Dispose();
			gpuLease.Dispose();
		}

	}
}
