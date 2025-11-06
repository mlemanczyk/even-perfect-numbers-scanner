using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;
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

		var lease = GpuKernelPool.GetKernel();
		Accelerator accelerator = lease.Accelerator;
		var stream = lease.Stream;
		KernelContainer kernels = lease.Kernels;

		ScratchBuffer scratch = GpuScratchBufferPool.Rent(accelerator, GpuSmallPrimeFactorSlots, 0);

		SmallPrimeFactorTables tables = GpuKernelPool.EnsureSmallPrimeFactorTables(kernels, accelerator, stream);

		if (kernels.SmallPrimeFactorsPrimes is null) throw new NullReferenceException($"{nameof(kernels.SmallPrimeFactorsPrimes)} is null");
		if (kernels.SmallPrimeFactorsSquares is null) throw new NullReferenceException($"{nameof(kernels.SmallPrimeFactorsSquares)} is null");

		var kernel = lease.SmallPrimeFactorKernel;

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


		factorCount = 0;
		smallPrimeFactorCountSlotView.CopyToCPU(stream, ref factorCount, 1);
		stream.Synchronize();

		factorCount = Math.Min(factorCount, GpuSmallPrimeFactorSlots);

		if (factorCount > 0)
		{
			smallPrimeFactorPrimeSlotsView.CopyToCPU(stream, ref MemoryMarshal.GetReference(primeBuffer), factorCount);

			smallPrimeFactorExponentSlotsView.CopyToCPU(stream, ref MemoryMarshal.GetReference(exponentBuffer), factorCount);
		}

		smallPrimeFactorRemainingSlotView.CopyToCPU(stream, ref remaining, 1);
		stream.Synchronize();

		for (int i = 0; i < factorCount; i++)
		{
			ulong primeValue = primeBuffer[i];
			int exponent = exponentBuffer[i];
			counts.Add(primeValue, exponent);
		}

		GpuScratchBufferPool.Return(scratch);
		lease.Dispose();
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

		GpuKernelLease lease = GpuKernelPool.GetKernel();
		Accelerator accelerator = lease.Accelerator;

		var stream = lease.Stream;

		ScratchBuffer scratch = GpuScratchBufferPool.Rent(accelerator, 0, factorCount);

		ArrayView1D<ulong, Stride1D.Dense> specialMaxFactorsView = scratch.SpecialMaxFactors.View;
		specialMaxFactorsView.SubView(0, factorCount).CopyFromCPU(stream, ref MemoryMarshal.GetReference(factorSpan), factorCount);

		var kernel = lease.SpecialMaxKernel;

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

		Span<ulong> result = stackalloc ulong[1];
		// specialMaxResultView.CopyToCPU(stream, ref result, 1);
		specialMaxResultView.CopyToCPU(stream, in result);
		stream.Synchronize();

		// stream.Synchronize();
		// return result != 0;
		GpuScratchBufferPool.Return(scratch);
		lease.Dispose();
		return result[0] != 0;
	}
}

