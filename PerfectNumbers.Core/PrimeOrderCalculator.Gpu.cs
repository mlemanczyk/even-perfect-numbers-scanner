using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core;

internal static partial class PrimeOrderCalculator
{
	private static bool IsGpuPow2Allowed => s_pow2ModeInitialized && s_allowGpuPow2;

	private const int GpuSmallPrimeFactorSlots = 64;

	private static ulong CalculateByFactorizationGpu(PrimeOrderCalculatorAccelerator gpu, HeuristicCombinedPrimeTesterAccelerator gpu2, ulong prime, in MontgomeryDivisorData divisorData)
	{
		ulong phi = prime - 1UL;
		Dictionary<ulong, int> counts = gpu.Pow2ModEntriesToTestOnHost;
		counts.Clear();
		
		FactorCompletelyCpu(gpu2, phi, counts);
		if (counts.Count == 0)
		{
			return phi;
		}

		int entryCount = counts.Count;
		Span<KeyValuePair<ulong, int>> entries = stackalloc KeyValuePair<ulong, int>[entryCount];

		int stored = 0;
		foreach (var entry in counts)
		{
			entries[stored++] = entry;
		}
		
		entries.Sort(entries, static (a, b) => a.Key.CompareTo(b.Key));

		ulong order = phi;
		var acceleratorIndex = gpu.AcceleratorIndex;
		var kernel = gpu.CheckFactorsKernel;
		gpu.EnsureCapacity(entryCount, 1);

		// GpuPrimeWorkLimiter.Acquire();
		AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		gpu.Pow2ModEntriesToTestOnDevice.View.CopyFromCPU(stream, entries);

		var kernelLauncher = kernel.CreateLauncherDelegate<Action<AcceleratorStream, int, ulong, ArrayView1D<KeyValuePair<ulong, int>, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>>();
		
		kernelLauncher(stream, entryCount, phi, gpu.Pow2ModEntriesToTestOnDevice.View, divisorData, gpu.OutputUlong.View);

		gpu.OutputUlong.View.CopyToCPU(stream, ref order, 1);
		stream.Synchronize();
		AcceleratorStreamPool.Return(acceleratorIndex, stream);

		// GpuPrimeWorkLimiter.Release();
		return order;
	}

	private static bool TryPopulateSmallPrimeFactorsGpu(PrimeOrderCalculatorAccelerator gpu, ulong value, uint limit, Dictionary<ulong, int> counts, out int factorCount, out ulong remaining)
	{
		var primeBufferArray = ThreadStaticPools.UlongPool.Rent(GpuSmallPrimeFactorSlots);
		var exponentBufferArray = ThreadStaticPools.IntPool.Rent(GpuSmallPrimeFactorSlots);
		Span<ulong> primeBuffer = primeBufferArray.AsSpan(0, GpuSmallPrimeFactorSlots);
		Span<int> exponentBuffer = exponentBufferArray.AsSpan(0, GpuSmallPrimeFactorSlots);
		remaining = value;

		gpu.EnsureSmallPrimeFactorSlotsCapacity(GpuSmallPrimeFactorSlots);
		int acceleratorIndex = gpu.AcceleratorIndex;
		var accelerator = gpu.Accelerator;

		// GpuPrimeWorkLimiter.Acquire();
		var stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		KernelContainer kernels = GpuKernelPool.GetOrAddKernels(acceleratorIndex, stream, KernelType.SmallPrimeFactorKernelScan);
		SmallPrimeFactorViews tables = GpuKernelPool.GetSmallPrimeFactorTables(kernels);

		var kernel = kernels.SmallPrimeFactor!;
		var kernelLauncher = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, uint, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>>();

		ArrayView1D<int, Stride1D.Dense> smallPrimeFactorCountSlotView = gpu.SmallPrimeFactorCountSlot.View;
		ArrayView1D<ulong, Stride1D.Dense> smallPrimeFactorRemainingSlotView = gpu.SmallPrimeFactorRemainingSlot.View;
		ArrayView1D<ulong, Stride1D.Dense> smallPrimeFactorPrimeSlotsView = gpu.SmallPrimeFactorPrimeSlots.View;
		ArrayView1D<int, Stride1D.Dense> smallPrimeFactorExponentSlotsView = gpu.SmallPrimeFactorExponentSlots.View;
		ArrayView1D<uint, Stride1D.Dense> smallPrimeFactorsPrimesView = kernels.SmallPrimeFactorsPrimes!.View;

		kernelLauncher(
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
			primeBuffer = primeBuffer[..factorCount];
			smallPrimeFactorPrimeSlotsView.CopyToCPU(stream, primeBuffer);

			exponentBuffer = exponentBuffer[..factorCount];
			smallPrimeFactorExponentSlotsView.CopyToCPU(stream, exponentBuffer);
		}

		smallPrimeFactorRemainingSlotView.CopyToCPU(stream, ref remaining, 1);
		stream.Synchronize();

		AcceleratorStreamPool.Return(acceleratorIndex, stream);
		// GpuPrimeWorkLimiter.Release();

		for (int i = 0; i < factorCount; i++)
		{
			ulong primeValue = primeBuffer[i];
			int exponent = exponentBuffer[i];
			counts.Add(primeValue, exponent);
		}

		return true;
	}

	private static bool EvaluateSpecialMaxCandidatesGpu(
			PrimeOrderCalculatorAccelerator gpu,
			ReadOnlySpan<ulong> factors,
			ulong phi,
			ulong prime,
			in MontgomeryDivisorData divisorData)
	{
		int factorCount = factors.Length;
		gpu.EnsureSpecialMaxFactorsCapacity(factorCount);

		int acceleratorIndex = gpu.AcceleratorIndex;
		Accelerator accelerator = gpu.Accelerator;
		// GpuPrimeWorkLimiter.Acquire();
		var stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		var kernels = GpuKernelPool.GetOrAddKernels(acceleratorIndex, stream, KernelType.EvaluateSpecialMaxCandidatesKernel);

		ArrayView1D<ulong, Stride1D.Dense> specialMaxFactorsView = gpu.SpecialMaxFactors.View;
		specialMaxFactorsView.SubView(0, factorCount).CopyFromCPU(stream, ref MemoryMarshal.GetReference(factors), factorCount);

		var kernel = kernels.SpecialMax!;

		ArrayView1D<ulong, Stride1D.Dense> specialMaxResultView = gpu.SpecialMaxResult.View;

		kernel(
				stream,
				1,
				phi,
				specialMaxFactorsView,
				factorCount,
				divisorData,
				gpu.SpecialMaxCandidates.View,
				specialMaxResultView);

		ulong result = 0UL;
		specialMaxResultView.CopyToCPU(stream, ref result, 1);
		stream.Synchronize();
		
		AcceleratorStreamPool.Return(acceleratorIndex, stream);
		// GpuPrimeWorkLimiter.Release();
		return result != 0;
	}

	private static bool TryPollardRhoGpu(PrimeOrderCalculatorAccelerator gpu, ulong n, out ulong factor)
	{
		Span<ulong> randomStateSpan = stackalloc ulong[1]; 
		randomStateSpan[0] = ThreadStaticDeterministicRandomGpu.Exclusive.State;

		Span<byte> factoredSpan = stackalloc byte[1];
		Span<ulong> factorSpan = stackalloc ulong[1];
		
		int acceleratorIndex = gpu.AcceleratorIndex;
		AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		gpu.Input.View.CopyFromCPU(stream, randomStateSpan);

		var kernelLauncher = gpu.PollardRhoKernel.CreateLauncherDelegate<Action<AcceleratorStream, ulong, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>>();

		kernelLauncher(stream, n, 1, gpu.Input.View, gpu.OutputByte.View, gpu.OutputUlong.View);

		gpu.OutputByte.View.CopyToCPU(stream, factoredSpan);
		gpu.OutputUlong.View.CopyToCPU(stream, factorSpan);
		gpu.Input.View.CopyToCPU(stream, randomStateSpan);
		stream.Synchronize();

		AcceleratorStreamPool.Return(acceleratorIndex, stream);
		ThreadStaticDeterministicRandomGpu.Exclusive.SetState(randomStateSpan[0]);

		bool factored = factoredSpan[0] != 0;
		factor = factored ? factorSpan[0] : 0UL;
		return factored;
	}
	
}

