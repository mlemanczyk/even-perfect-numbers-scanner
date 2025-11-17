using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu.Kernels;

namespace PerfectNumbers.Core.Gpu.Accelerators;

public sealed class PrimeOrderCalculatorAccelerator
{
	#region Pool

	[ThreadStatic]
	private static Queue<PrimeOrderCalculatorAccelerator>? _pool;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static PrimeOrderCalculatorAccelerator Rent(int primeTesterCapacity)
	{
		var pool = _pool ??= [];
		if (!pool.TryDequeue(out var gpu))
		{
			int acceleratorIndex = AcceleratorPool.Shared.Rent();
			gpu = new(acceleratorIndex, PerfectNumberConstants.DefaultFactorsBuffer, primeTesterCapacity, PerfectNumberConstants.DefaultSmallPrimeFactorSlotCount, PerfectNumberConstants.DefaultSpecialMaxFactorCapacity);
		}
		else
		{
			gpu.EnsureCapacity(PerfectNumberConstants.DefaultFactorsBuffer, primeTesterCapacity);
		}

		return gpu;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static void Return(PrimeOrderCalculatorAccelerator gpu) => _pool!.Enqueue(gpu);

	internal static void Clear()
	{
		Queue<PrimeOrderCalculatorAccelerator>? pool = _pool;
		if (pool != null)
		{
			while (pool.TryDequeue(out var gpu))
			{
				gpu.Dispose();
			}
		}
	}

	internal static void DisposeAll()
	{
		Queue<PrimeOrderCalculatorAccelerator>? pool = _pool;
		if (pool != null)
		{
			while (pool.TryDequeue(out var gpu))
			{
				gpu.Dispose();
			}
		}
	}

	internal static void WarmUp()
	{
		Accelerator[] accelerators = AcceleratorPool.Shared.Accelerators;
		int acceleratorCount = accelerators.Length;
		for (var i = 0; i < acceleratorCount; i++)
		{
			Console.WriteLine($"Preparing accelerator {i}...");
			var accelerator = accelerators[i];
			AcceleratorStreamPool.WarmUp(i);
			// Don't take this from the pool as quick uploads of data to the accelerator consumes much of GPU's memory and throws.
			AcceleratorStream stream = accelerator.CreateStream();
			LastDigitGpuTables.WarmUp(i, stream);
			HeuristicCombinedPrimeTesterAccelerator.WarmUp(i, stream);
			// SharedHeuristicGpuTables.EnsureStaticTables(accelerator, stream);
			// _ = GpuKernelPool.GetOrAddKernels(accelerator, stream, KernelType.None);
			// KernelContainer kernels = GpuKernelPool.GetOrAddKernels(accelerator, stream);
			// GpuStaticTableInitializer.EnsureStaticTables(accelerator, kernels, stream);
			stream.Synchronize();
			stream.Dispose();
			accelerator.Synchronize();
		}

	}

	#endregion

	private static readonly Accelerator[] _accelerators = AcceleratorPool.Shared.Accelerators;

	public readonly Accelerator Accelerator;
	public readonly int AcceleratorIndex;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> Input;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> OutputUlong;
	public MemoryBuffer1D<ulong, Stride1D.Dense> PrimeTestInput;
	public MemoryBuffer1D<byte, Stride1D.Dense> OutputByte;
	public MemoryBuffer1D<int, Stride1D.Dense> HeuristicFlag;
	public MemoryBuffer1D<KeyValuePair<ulong, int>, Stride1D.Dense> Pow2ModEntriesToTestOnDevice;
	public Dictionary<ulong, int> Pow2ModEntriesToTestOnHost = [];
	public readonly MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastOne;
	public readonly MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastSeven;
	public readonly MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastThree;
	public readonly MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastNine;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastOne;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastSeven;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastThree;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastNine;
	public readonly MemoryBuffer1D<int, Stride1D.Dense> SmallPrimeFactorCountSlot;
	public MemoryBuffer1D<int, Stride1D.Dense> SmallPrimeFactorExponentSlots;
	public MemoryBuffer1D<ulong, Stride1D.Dense> SmallPrimeFactorPrimeSlots;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> SmallPrimeFactorRemainingSlot;
	public MemoryBuffer1D<ulong, Stride1D.Dense> SpecialMaxCandidates;
	public MemoryBuffer1D<ulong, Stride1D.Dense> SpecialMaxFactors;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> SpecialMaxResult;

	public readonly Kernel SmallPrimeSieveKernel;
	public readonly Kernel CheckFactorsKernel;
	public readonly Kernel ConvertToStandardKernel;
	public readonly Kernel KeepMontgomeryKernel;
	public readonly Kernel PollardRhoKernel;
	public readonly Kernel Pow2ModWideKernel;
	public Kernel SharesFactorKernel;


	public void EnsureCapacity(int factorsCount, int primeTesterCapacity)
	{
		if (factorsCount > Pow2ModEntriesToTestOnDevice.Length)
		{
			Accelerator accelerator = Accelerator;
			Pow2ModEntriesToTestOnDevice.Dispose();
			Pow2ModEntriesToTestOnDevice = accelerator.Allocate1D<KeyValuePair<ulong, int>>(factorsCount);
		}

		if (primeTesterCapacity > PrimeTestInput.Length)
		{
			PrimeTestInput.Dispose();
			OutputByte.Dispose();
			HeuristicFlag.Dispose();

			PrimeTestInput = Accelerator.Allocate1D<ulong>(primeTesterCapacity);
			OutputByte = Accelerator.Allocate1D<byte>(primeTesterCapacity);
			HeuristicFlag = Accelerator.Allocate1D<int>(primeTesterCapacity);
		}
	}

	public void EnsureSmallPrimeFactorSlotsCapacity(int newSize)
	{
		// Console.WriteLine($"Resizing GPU scratch buffer from pool ({buffer.SmallPrimeFactorPrimeSlots.Length} / {smallPrimeFactorSlotCount}), ({buffer.SpecialMaxFactors.Length}/{specialMaxFactorCapacity})");
		if (SmallPrimeFactorPrimeSlots.Length < newSize)
		{
			SmallPrimeFactorPrimeSlots.Dispose();
			SmallPrimeFactorExponentSlots.Dispose();

			var accelerator = Accelerator;
			SmallPrimeFactorPrimeSlots = accelerator.Allocate1D<ulong>(newSize);
			SmallPrimeFactorExponentSlots = accelerator.Allocate1D<int>(newSize);
		}
	}

	public void EnsureSpecialMaxFactorsCapacity(int newSize)
	{
		if (SpecialMaxFactors.Length < newSize)
		{
			SpecialMaxFactors.Dispose();
			SpecialMaxCandidates.Dispose();

			var accelerator = Accelerator;
			SpecialMaxFactors = accelerator.Allocate1D<ulong>(newSize);
			SpecialMaxCandidates = accelerator.Allocate1D<ulong>(newSize);
		}
	}

	public PrimeOrderCalculatorAccelerator(int acceleratorIndex, int factorsCount, int primeTesterCapacity, int smallPrimeFactorSlotCount, int specialMaxFactorCapacity)
	{
		var accelerator = _accelerators[acceleratorIndex];
		Accelerator = accelerator;
		AcceleratorIndex = acceleratorIndex;
		Input = accelerator.Allocate1D<ulong>(1);
		OutputUlong = accelerator.Allocate1D<ulong>(1);
		Pow2ModEntriesToTestOnDevice = accelerator.Allocate1D<KeyValuePair<ulong, int>>(factorsCount);
		PrimeTestInput = Accelerator.Allocate1D<ulong>(primeTesterCapacity);
		OutputByte = Accelerator.Allocate1D<byte>(primeTesterCapacity);
		HeuristicFlag = Accelerator.Allocate1D<int>(primeTesterCapacity);
		SmallPrimeFactorPrimeSlots = accelerator.Allocate1D<ulong>(smallPrimeFactorSlotCount);
		SmallPrimeFactorExponentSlots = accelerator.Allocate1D<int>(smallPrimeFactorSlotCount);
		SmallPrimeFactorCountSlot = accelerator.Allocate1D<int>(1);
		SmallPrimeFactorRemainingSlot = accelerator.Allocate1D<ulong>(1);

		SpecialMaxFactors = accelerator.Allocate1D<ulong>(specialMaxFactorCapacity);
		SpecialMaxCandidates = accelerator.Allocate1D<ulong>(specialMaxFactorCapacity);
		SpecialMaxResult = accelerator.Allocate1D<ulong>(1);

		var sharedTables = LastDigitGpuTables.EnsureStaticTables(acceleratorIndex);

		DevicePrimesLastOne = sharedTables.DevicePrimesLastOne;
		DevicePrimesLastSeven = sharedTables.DevicePrimesLastSeven;
		DevicePrimesLastThree = sharedTables.DevicePrimesLastThree;
		DevicePrimesLastNine = sharedTables.DevicePrimesLastNine;
		DevicePrimesPow2LastOne = sharedTables.DevicePrimesPow2LastOne;
		DevicePrimesPow2LastSeven = sharedTables.DevicePrimesPow2LastSeven;
		DevicePrimesPow2LastThree = sharedTables.DevicePrimesPow2LastThree;
		DevicePrimesPow2LastNine = sharedTables.DevicePrimesPow2LastNine;

		var checkFactorsKernel = accelerator.LoadStreamKernel<int, ulong, ArrayView1D<KeyValuePair<ulong, int>, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>(PrimeOrderGpuHeuristics.CheckFactorsKernel);
		CheckFactorsKernel = KernelUtil.GetKernel(checkFactorsKernel);

		var convertKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>(Pow2MontgomeryKernels.Pow2MontgomeryKernelConvertToStandard);
		ConvertToStandardKernel = KernelUtil.GetKernel(convertKernel);

		var keepKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>(Pow2MontgomeryKernels.Pow2MontgomeryKernelKeepMontgomery);
		KeepMontgomeryKernel = KernelUtil.GetKernel(keepKernel);

		SmallPrimeSieveKernel = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>(PrimeTesterKernels.SmallPrimeSieveKernel));

		PollardRhoKernel = KernelUtil.GetKernel(accelerator.LoadStreamKernel<ulong, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>(Pow2MontgomeryKernels.TryPollardRhoKernel));

		Pow2ModWideKernel = KernelUtil.GetKernel(
			accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<GpuUInt128, Stride1D.Dense>, GpuUInt128, ArrayView1D<GpuUInt128, Stride1D.Dense>>(PrimeOrderGpuHeuristics.Pow2ModKernelWide));

		SharesFactorKernel = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<byte>>(PrimeTesterKernels.SharesFactorKernel));
	}

	public void Dispose()
	{
		Input.Dispose();
		OutputUlong.Dispose();
		Pow2ModEntriesToTestOnDevice.Dispose();
		PrimeTestInput.Dispose();
		OutputByte.Dispose();
		HeuristicFlag.Dispose();
		SmallPrimeFactorPrimeSlots.Dispose();
		SmallPrimeFactorExponentSlots.Dispose();
		SmallPrimeFactorCountSlot.Dispose();
		SmallPrimeFactorRemainingSlot.Dispose();
		SpecialMaxFactors.Dispose();
		SpecialMaxCandidates.Dispose();
		SpecialMaxResult.Dispose();

		// These resources are shared between GPU leases
		// Stream.Dispose();
		// Accelerator.Dispose();
		// Context.Dispose();
	}
}
