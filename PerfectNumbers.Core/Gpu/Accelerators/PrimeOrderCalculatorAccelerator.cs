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
			// TODO: Review which tables are needed when final execution path is defined.
			LastDigitGpuTables.WarmUp(i, stream);
			HeuristicGpuCombinedTables.WarmUp(i, stream);
			// SmallPrimeFactorGpuTables.WarmUp(i, stream);
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
	public MemoryBuffer1D<ulong, Stride1D.Dense> Input;
	public MemoryBuffer1D<ulong, Stride1D.Dense> OutputUlong;
	public MemoryBuffer1D<byte, Stride1D.Dense> OutputByte;
	public MemoryBuffer1D<int, Stride1D.Dense> OutputInt;
	public MemoryBuffer1D<KeyValuePair<ulong, int>, Stride1D.Dense> Pow2ModEntriesToTestOnDevice;
	public Dictionary<ulong, int> Pow2ModEntriesToTestOnHost = [];

	#region Static Tables

	public readonly MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastOne;
	public readonly MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastSeven;
	public readonly MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastThree;
	public readonly MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastNine;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastOne;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastSeven;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastThree;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastNine;

	#endregion

	public readonly MemoryBuffer1D<int, Stride1D.Dense> SmallPrimeFactorCountSlot;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> SpecialMaxResult;

	public readonly HeuristicGpuCombinedDivisorTables DivisorTables;
	public readonly SmallPrimeFactorGpuTables SmallPrimeFactorTables;

	public readonly Kernel CheckFactorsKernel;
	public readonly Kernel ConvertToStandardKernel;
	public readonly Kernel HeuristicCombinedTrialDivisionKernel;
	public readonly Kernel KeepMontgomeryKernel;
	public readonly Kernel PollardRhoKernel;
	public readonly Kernel Pow2ModWideKernel;
	public readonly Kernel SharesFactorKernel;
	public readonly Kernel SmallPrimeFactorKernel;
	public readonly Kernel SmallPrimeSieveKernel;
	public readonly Kernel SpecialMaxKernel;

	public void EnsureCapacity(int factorsCount, int primeTesterCapacity)
	{
		if (factorsCount > Pow2ModEntriesToTestOnDevice.Length)
		{
			Accelerator accelerator = Accelerator;
			Pow2ModEntriesToTestOnDevice.Dispose();
			Pow2ModEntriesToTestOnDevice = accelerator.Allocate1D<KeyValuePair<ulong, int>>(factorsCount);
		}

		if (primeTesterCapacity > Input.Length)
		{
			Input.Dispose();
			Input = Accelerator.Allocate1D<ulong>(primeTesterCapacity);			
		}

		if (primeTesterCapacity > OutputByte.Length)
		{
			OutputByte.Dispose();
			OutputByte = Accelerator.Allocate1D<byte>(primeTesterCapacity);
		}

		if (primeTesterCapacity > OutputInt.Length)
		{
			OutputInt.Dispose();
			OutputInt = Accelerator.Allocate1D<int>(primeTesterCapacity);
		}
	}

	public void EnsureSmallPrimeFactorSlotsCapacity(int newSize)
	{
		// Console.WriteLine($"Resizing GPU scratch buffer from pool ({buffer.SmallPrimeFactorPrimeSlots.Length} / {smallPrimeFactorSlotCount}), ({buffer.SpecialMaxFactors.Length}/{specialMaxFactorCapacity})");
		if (OutputUlong.Length < newSize)
		{
			OutputUlong.Dispose();

			var accelerator = Accelerator;
			OutputUlong = accelerator.Allocate1D<ulong>(newSize);
		}

		if (OutputInt.Length < newSize)
		{
			OutputInt.Dispose();
			
			var accelerator = Accelerator;
			OutputInt = accelerator.Allocate1D<int>(newSize);
		}
	}

	public void EnsureSpecialMaxFactorsCapacity(int newSize)
	{
		if (Input.Length < newSize)
		{
			Input.Dispose();

			var accelerator = Accelerator;
			Input = accelerator.Allocate1D<ulong>(newSize);
		}

		if (OutputUlong.Length < newSize)
		{
			OutputUlong.Dispose();

			var accelerator = Accelerator;
			OutputUlong = accelerator.Allocate1D<ulong>(newSize);
		}
	}

	public PrimeOrderCalculatorAccelerator(int acceleratorIndex, int factorsCount, int primeTesterCapacity, int smallPrimeFactorSlotCount, int specialMaxFactorCapacity)
	{
		var accelerator = _accelerators[acceleratorIndex];
		Accelerator = accelerator;
		AcceleratorIndex = acceleratorIndex;
		Input = accelerator.Allocate1D<ulong>(Math.Max(specialMaxFactorCapacity, primeTesterCapacity));
		OutputUlong = accelerator.Allocate1D<ulong>(Math.Max(specialMaxFactorCapacity, smallPrimeFactorSlotCount));
		Pow2ModEntriesToTestOnDevice = accelerator.Allocate1D<KeyValuePair<ulong, int>>(factorsCount);
		OutputByte = Accelerator.Allocate1D<byte>(primeTesterCapacity);
		OutputInt = Accelerator.Allocate1D<int>(Math.Max(primeTesterCapacity, smallPrimeFactorSlotCount));
		SmallPrimeFactorCountSlot = accelerator.Allocate1D<int>(1);

		SpecialMaxResult = accelerator.Allocate1D<ulong>(1);

		LastDigitGpuTables lastDigitSharedTables = LastDigitGpuTables.GetStaticTables(acceleratorIndex);

		DevicePrimesLastOne = lastDigitSharedTables.DevicePrimesLastOne;
		DevicePrimesLastSeven = lastDigitSharedTables.DevicePrimesLastSeven;
		DevicePrimesLastThree = lastDigitSharedTables.DevicePrimesLastThree;
		DevicePrimesLastNine = lastDigitSharedTables.DevicePrimesLastNine;
		DevicePrimesPow2LastOne = lastDigitSharedTables.DevicePrimesPow2LastOne;
		DevicePrimesPow2LastSeven = lastDigitSharedTables.DevicePrimesPow2LastSeven;
		DevicePrimesPow2LastThree = lastDigitSharedTables.DevicePrimesPow2LastThree;
		DevicePrimesPow2LastNine = lastDigitSharedTables.DevicePrimesPow2LastNine;

		var heuristicSharedTables = HeuristicGpuCombinedTables.GetStaticTables(acceleratorIndex);
		DivisorTables = heuristicSharedTables.CreateHeuristicDivisorTables();
		SmallPrimeFactorTables = SmallPrimeFactorGpuTables.GetStaticTables(acceleratorIndex);

		var checkFactorsKernel = accelerator.LoadStreamKernel<int, ulong, ArrayView1D<KeyValuePair<ulong, int>, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>(PrimeOrderGpuHeuristics.CheckFactorsKernel);
		CheckFactorsKernel = KernelUtil.GetKernel(checkFactorsKernel);

		var convertKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>(Pow2MontgomeryKernels.Pow2MontgomeryKernelConvertToStandard);
		ConvertToStandardKernel = KernelUtil.GetKernel(convertKernel);

		HeuristicCombinedTrialDivisionKernel = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, byte, ArrayView<int>, ulong, ulong, HeuristicGpuCombinedDivisorTables>(PrimeTesterKernels.HeuristicTrialCombinedDivisionKernel));

		var keepKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>(Pow2MontgomeryKernels.Pow2MontgomeryKernelKeepMontgomery);
		KeepMontgomeryKernel = KernelUtil.GetKernel(keepKernel);

		SmallPrimeSieveKernel = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>(PrimeTesterKernels.SmallPrimeSieveKernel));

		PollardRhoKernel = KernelUtil.GetKernel(accelerator.LoadStreamKernel<ulong, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>(Pow2MontgomeryKernels.TryPollardRhoKernel));

		Pow2ModWideKernel = KernelUtil.GetKernel(
			accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<GpuUInt128, Stride1D.Dense>, GpuUInt128, ArrayView1D<GpuUInt128, Stride1D.Dense>>(PrimeOrderGpuHeuristics.Pow2ModKernelWide));

		SharesFactorKernel = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<byte>>(PrimeTesterKernels.SharesFactorKernel));

		SpecialMaxKernel = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ulong, ArrayView1D<ulong, Stride1D.Dense>, int, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>(PrimeOrderGpuHeuristics.EvaluateSpecialMaxCandidatesKernel));

		SmallPrimeFactorKernel = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ulong, uint, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>(SmallPrimeFactorKernels.SmallPrimeFactorKernelScan));
	}

	public void Dispose()
	{
		Input.Dispose();
		OutputUlong.Dispose();
		Pow2ModEntriesToTestOnDevice.Dispose();
		OutputByte.Dispose();
		OutputInt.Dispose();
		SmallPrimeFactorCountSlot.Dispose();
		SpecialMaxResult.Dispose();

		// These resources are shared between GPU leases
		// Stream.Dispose();
		// Accelerator.Dispose();
		// Context.Dispose();
	}
}
