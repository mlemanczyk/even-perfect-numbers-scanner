using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core;

internal static partial class PrimeOrderCalculator
{
	private static bool IsGpuPow2Allowed => s_pow2ModeInitialized && s_allowGpuPow2;

	private const int GpuSmallPrimeFactorSlots = 64;

	private static ulong CalculateByFactorizationGpu(PrimeOrderCalculatorAccelerator gpu, ulong prime, in MontgomeryDivisorData divisorData)
	{
		ulong phi = prime - 1UL;
		Dictionary<ulong, int> counts = gpu.Pow2ModEntriesToTestOnHost;
		counts.Clear();
		
		FactorCompletelyCpu(gpu, phi, counts);
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
		var pow2ModEntriesToTestOnDeviceView = gpu.Pow2ModEntriesToTestOnDeviceView;

		AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		pow2ModEntriesToTestOnDeviceView.CopyFromCPU(stream, entries);

		kernel.Launch(stream, 1, entryCount, phi, pow2ModEntriesToTestOnDeviceView, divisorData.Modulus, divisorData.NPrime, divisorData.MontgomeryOne, divisorData.MontgomeryTwo, divisorData.MontgomeryTwoSquared, gpu.OutputUlongView);

		gpu.OutputUlongView.CopyToCPU(stream, ref order, 1);
		stream.Synchronize();

		AcceleratorStreamPool.Return(acceleratorIndex, stream);
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

		var kernelLauncher = gpu.SmallPrimeFactorKernelLauncher;
		ArrayView1D<int, Stride1D.Dense> smallPrimeFactorCountSlotView = gpu.OutputIntView2;
		ArrayView1D<ulong, Stride1D.Dense> smallPrimeFactorRemainingSlotView = gpu.InputView;
		ArrayView1D<ulong, Stride1D.Dense> smallPrimeFactorPrimeSlotsView = gpu.OutputUlongView;
		ArrayView1D<int, Stride1D.Dense> smallPrimeFactorExponentSlotsView = gpu.OutputIntView;
		ArrayView1D<ulong, Stride1D.Dense> smallPrimeFactorsSquaresView = gpu.SmallPrimeFactorSquares;
		ArrayView1D<uint, Stride1D.Dense> smallPrimeFactorsPrimesView = gpu.SmallPrimeFactorPrimes;

		// GpuPrimeWorkLimiter.Acquire();
		var stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		kernelLauncher(
				stream,
				1,
				value,
				limit,
				smallPrimeFactorsPrimesView,
				smallPrimeFactorsSquaresView,
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
		gpu.EnsureUlongInputOutputCapacity(factorCount);

		int acceleratorIndex = gpu.AcceleratorIndex;
		// GpuPrimeWorkLimiter.Acquire();
		ArrayView<ulong> specialMaxFactorsView = gpu.InputView;
		ArrayView<ulong> specialMaxResultView = gpu.OutputUlongView2;
		var kernelLauncher = gpu.SpecialMaxKernelLauncher;

		var stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		specialMaxFactorsView.SubView(0, factorCount).CopyFromCPU(stream, factors);

		kernelLauncher(
				stream,
				1,
				phi,
				specialMaxFactorsView,
				factorCount,
				divisorData.Modulus,
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
		var kernel = gpu.PollardRhoKernel;

		AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		gpu.InputView.CopyFromCPU(stream, randomStateSpan);

		kernel.Launch(stream, 1, n, 1, gpu.InputView, gpu.OutputByteView, gpu.OutputUlongView);

		gpu.OutputByteView.CopyToCPU(stream, factoredSpan);
		gpu.OutputUlongView.CopyToCPU(stream, factorSpan);
		gpu.InputView.CopyToCPU(stream, randomStateSpan);
		stream.Synchronize();

		AcceleratorStreamPool.Return(acceleratorIndex, stream);
		ThreadStaticDeterministicRandomGpu.Exclusive.SetState(randomStateSpan[0]);

		bool factored = factoredSpan[0] != 0;
		factor = factored ? factorSpan[0] : 0UL;
		return factored;
	}
	
}

