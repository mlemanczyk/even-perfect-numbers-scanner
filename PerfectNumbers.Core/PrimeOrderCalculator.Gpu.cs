using System;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

internal static partial class PrimeOrderCalculator
{
	private static bool IsGpuPow2Allowed => s_pow2ModeInitialized && s_allowGpuPow2;

	private const int GpuSmallPrimeFactorSlots = 64;

	private static bool TryPopulateSmallPrimeFactorsGpu(ulong value, uint limit, Dictionary<ulong, int> counts, out int factorCount, out ulong remaining)
	{
		var primeBufferArray = ThreadStaticPools.UlongPool.Rent(GpuSmallPrimeFactorSlots);
		var exponentBufferArray = ThreadStaticPools.IntPool.Rent(GpuSmallPrimeFactorSlots);
		Span<ulong> primeBuffer = primeBufferArray.AsSpan(0, GpuSmallPrimeFactorSlots);
		Span<int> exponentBuffer = exponentBufferArray.AsSpan(0, GpuSmallPrimeFactorSlots);
		// primeBuffer.Clear();
		// exponentBuffer.Clear();
		remaining = value;

		try
		{
			var lease = GpuKernelPool.GetKernel(useGpuOrder: true);
			var execution = lease.EnterExecutionScope();
			Accelerator accelerator = lease.Accelerator;
			AcceleratorStream stream = lease.Stream;

			SmallPrimeFactorScratch scratch = GpuKernelPool.EnsureSmallPrimeFactorScratch(accelerator, GpuSmallPrimeFactorSlots);

			bool success = PrimeOrderGpuHeuristics.TryPartialFactor(
				accelerator,
				stream,
				scratch,
				value,
				limit,
				primeBuffer,
				exponentBuffer,
				out factorCount,
				out remaining,
				out _);

			execution.Dispose();
			lease.Dispose();

			if (!success)
			{
				return false;
			}

			for (int i = 0; i < factorCount; i++)
			{
				ulong primeValue = primeBuffer[i];
				int exponent = exponentBuffer[i];
				// This will never happen in production code
				// if (primeValue == 0UL || exponent == 0)
				// {
				//      continue;
				// }

				counts.Add(primeValue, exponent);
			}

			return true;
		}
		finally
		{
			ThreadStaticPools.UlongPool.Return(primeBufferArray);
			ThreadStaticPools.IntPool.Return(exponentBufferArray);
		}
	}
}
