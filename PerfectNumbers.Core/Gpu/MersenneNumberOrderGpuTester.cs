using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public class MersenneNumberOrderGpuTester(GpuKernelType kernelType, bool useGpuOrder)
{
	private readonly GpuKernelType _kernelType = kernelType;
	private readonly bool _useGpuOrder = useGpuOrder;

	public void Scan(ulong exponent, UInt128 twoP, LastDigit lastDigit, UInt128 maxK, ref bool isPrime)
	{
		var gpuLease = GpuKernelPool.GetKernel(_useGpuOrder);
		var execution = gpuLease.EnterExecutionScope();
		var accelerator = gpuLease.Accelerator;
		var stream = gpuLease.Stream;
		int batchSize = GpuConstants.ScanBatchSize; // large batch improves GPU occupancy and avoids TDR
		UInt128 kStart = UInt128.One;
		ulong divMul = (ulong)((((UInt128)1 << 64) - UInt128.One) / exponent) + 1UL;
		byte last = lastDigit == LastDigit.Seven ? (byte)1 : (byte)0; // ILGPU kernels do not support bool parameters

                var kernel = _kernelType switch
                {
                        GpuKernelType.Pow2Mod => gpuLease.Pow2ModOrderKernel,
                        _ => gpuLease.IncrementalOrderKernel,
                };

                var foundBuffer = accelerator.Allocate1D<int>(1); // TODO: Replace this allocation with the pooled device buffer from GpuOrderKernelBenchmarks so repeated scans reuse the pinned staging memory instead of allocating per run.
                exponent.Mod10_8_5_3Steps(out ulong step10, out ulong step8, out ulong step5, out ulong step3);
                try
                {
			UInt128 remaining;
			int currentSize;
			int found = 0;
			while (kStart <= maxK && Volatile.Read(ref isPrime))
			{
				remaining = maxK - kStart + 1UL;
				currentSize = remaining > (UInt128)batchSize ? batchSize : (int)remaining;
				// This clear is required because the result may not always be set. Remove it after the kernels are modified to always set the result.
				foundBuffer.MemSetToZero();
				// Precompute residue automaton bases for this batch
                                UInt128 q0 = twoP * kStart + 1UL;
                                // TODO: Replace these modulo recomputations with the divisor-cycle stepping deltas captured in
                                // MersenneDivisorCycleLengthGpuBenchmarks so each batch advances via cached residues from the
                                // single snapshot block instead of recalculating mod 10/8/5/3 in every iteration.
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
