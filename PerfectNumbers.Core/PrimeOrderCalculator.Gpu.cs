using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
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
		remaining = value;

		GpuPrimeWorkLimiter.Acquire();
		Accelerator accelerator = AcceleratorPool.Shared.Rent();
		var stream = accelerator.CreateStream();
		KernelContainer kernels = GpuKernelPool.GetOrAddKernels(accelerator, stream);

		ScratchBuffer scratch = GpuScratchBufferPool.Rent(accelerator, GpuSmallPrimeFactorSlots, 0);

		SmallPrimeFactorViews tables = GpuKernelPool.GetSmallPrimeFactorTables(kernels);

		if (kernels.SmallPrimeFactorsPrimes is null) throw new NullReferenceException($"{nameof(kernels.SmallPrimeFactorsPrimes)} is null");
		if (kernels.SmallPrimeFactorsSquares is null) throw new NullReferenceException($"{nameof(kernels.SmallPrimeFactorsSquares)} is null");

		var kernel = kernels.SmallPrimeFactor!;

		ArrayView1D<int, Stride1D.Dense> smallPrimeFactorCountSlotView = scratch.SmallPrimeFactorCountSlot.View;
		ArrayView1D<ulong, Stride1D.Dense> smallPrimeFactorRemainingSlotView = scratch.SmallPrimeFactorRemainingSlot.View;
		ArrayView1D<ulong, Stride1D.Dense> smallPrimeFactorPrimeSlotsView = scratch.SmallPrimeFactorPrimeSlots.View;
		ArrayView1D<int, Stride1D.Dense> smallPrimeFactorExponentSlotsView = scratch.SmallPrimeFactorExponentSlots.View;
		ArrayView1D<uint, Stride1D.Dense> smallPrimeFactorsPrimesView = kernels.SmallPrimeFactorsPrimes!.View;

		kernel(
				stream,
				1,
				value,
				limit,
				smallPrimeFactorsPrimesView,
				kernels.SmallPrimeFactorsSquares!.View,
				(int)smallPrimeFactorsPrimesView.Length,
				smallPrimeFactorPrimeSlotsView,
				smallPrimeFactorExponentSlotsView,
				smallPrimeFactorCountSlotView,
				smallPrimeFactorRemainingSlotView);


		// Span<int> factorCountTemp = stackalloc int[1];
		factorCount = 0;
		smallPrimeFactorCountSlotView.CopyToCPU(stream, ref factorCount, 1);
		stream.Synchronize();
		// factorCount = factorCountTemp[0];
		factorCount = Math.Min(factorCount, GpuSmallPrimeFactorSlots);

		if (factorCount > 0)
		{
			smallPrimeFactorPrimeSlotsView.CopyToCPU(stream, ref MemoryMarshal.GetReference(primeBuffer), factorCount);

			smallPrimeFactorExponentSlotsView.CopyToCPU(stream, ref MemoryMarshal.GetReference(exponentBuffer), factorCount);
		}

		// Span<ulong> remainingTemp = stackalloc ulong[1];

		smallPrimeFactorRemainingSlotView.CopyToCPU(stream, ref remaining, 1);
		// remaining = remainingTemp[0];
		stream.Synchronize();
		stream.Dispose();

		for (int i = 0; i < factorCount; i++)
		{
			ulong primeValue = primeBuffer[i];
			int exponent = exponentBuffer[i];
			counts.Add(primeValue, exponent);
		}

		GpuScratchBufferPool.Return(scratch);
		GpuPrimeWorkLimiter.Release();
		return true;
	}

	private static bool EvaluateSpecialMaxCandidatesGpu(
			Span<ulong> buffer,
			ReadOnlySpan<FactorEntry> factors,
			ulong phi,
			ulong prime,
			in MontgomeryDivisorData divisorData)
	{
		if (factors.Length == 0)
		{
			return true;
		}

		int factorCount = factors.Length;
		Span<ulong> factorSpan = buffer[..factorCount];
		for (int i = 0; i < factorCount; i++)
		{
			factorSpan[i] = factors[i].Value;
		}

		GpuPrimeWorkLimiter.Acquire();
		Accelerator accelerator = AcceleratorPool.Shared.Rent();
		var stream = accelerator.CreateStream();
		var kernels = GpuKernelPool.GetOrAddKernels(accelerator, stream);

		ScratchBuffer scratch = GpuScratchBufferPool.Rent(accelerator, 0, factorCount);

		ArrayView1D<ulong, Stride1D.Dense> specialMaxFactorsView = scratch.SpecialMaxFactors.View;
		specialMaxFactorsView.SubView(0, factorCount).CopyFromCPU(stream, ref MemoryMarshal.GetReference(factorSpan), factorCount);

		var kernel = kernels.SpecialMax!;

		ArrayView1D<ulong, Stride1D.Dense> specialMaxResultView = scratch.SpecialMaxResult.View;

		kernel(
				stream,
				1,
				phi,
				specialMaxFactorsView,
				factorCount,
				divisorData,
				scratch.SpecialMaxCandidates.View,
				specialMaxResultView);

		// Span<ulong> result = stackalloc ulong[1];
		ulong result = 0UL;
		// specialMaxResultView.CopyToCPU(stream, ref result, 1);
		specialMaxResultView.CopyToCPU(stream, ref result, 1);
		stream.Synchronize();
		stream.Dispose();

		// stream.Synchronize();
		// return result != 0;
		GpuScratchBufferPool.Return(scratch);
		GpuPrimeWorkLimiter.Release();
		return result != 0;
	}
}

