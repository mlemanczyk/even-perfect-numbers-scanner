using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu.Accelerators;

public sealed class HeuristicCombinedPrimeTesterAccelerator
{
	#region Pool

	[ThreadStatic]
	private static Queue<HeuristicCombinedPrimeTesterAccelerator>? Pool;

	public static void WarmUp(int acceleratorIndex, AcceleratorStream stream)
	{
		Console.WriteLine($"Preparing heuristic accelerator {acceleratorIndex}...");
		HeuristicGpuCombinedTables.WarmUp(acceleratorIndex, stream);
		// _ = GpuKernelPool.GetOrAddKernels(accelerator, stream, KernelType.None);
		// KernelContainer kernels = GpuKernelPool.GetOrAddKernels(accelerator, stream);
		// GpuStaticTableInitializer.EnsureStaticTables(accelerator, kernels, stream);
		Console.WriteLine($"Heuristic accelerator {acceleratorIndex} is ready");
	}

	public static HeuristicCombinedPrimeTesterAccelerator Rent(int minBufferCapacity = 1)
	{
		var pool = Pool ??= new();
		if (!pool.TryDequeue(out var lease))
		{
			lease = new HeuristicCombinedPrimeTesterAccelerator(minBufferCapacity);
		}
		else
		{
			lease.EnsureCapacity(minBufferCapacity);
		}

		return lease;
	}

	public static void Return(HeuristicCombinedPrimeTesterAccelerator lease)
	{
		var pool = Pool ??= new();
		pool.Enqueue(lease);
	}

	public static void DisposeAll()
	{
		if (Pool != null)
		{
			while (Pool.TryDequeue(out var lease))
			{
				lease.DisposeResources();
			}
		}
	}

	public static void Clear()
	{
		if (Pool != null)
		{
			var retained = new List<HeuristicCombinedPrimeTesterAccelerator>();
			while (Pool.TryDequeue(out var lease))
			{
				lease.DisposeResources();
			}
		}
	}

	#endregion

	public readonly HeuristicGpuCombinedDivisorTables DivisorTables;
	internal readonly Context Context;
	public readonly Accelerator Accelerator;
	public readonly Kernel HeuristicCombinedTrialDivisionKernel;

	public MemoryBuffer1D<ulong, Stride1D.Dense>? Input = null;
	public MemoryBuffer1D<byte, Stride1D.Dense>? Output = null;
	public MemoryBuffer1D<int, Stride1D.Dense>? HeuristicFlag = null!;
	public int BufferCapacity;

	private HeuristicGpuCombinedTables _sharedTables;
	public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorsEnding1 => _sharedTables.HeuristicCombinedDivisorsEnding1;
	public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorSquaresEnding1 => _sharedTables.HeuristicCombinedDivisorSquaresEnding1;
	public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorsEnding3 => _sharedTables.HeuristicCombinedDivisorsEnding3;
	public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorSquaresEnding3 => _sharedTables.HeuristicCombinedDivisorSquaresEnding3;
	public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorsEnding7 => _sharedTables.HeuristicCombinedDivisorsEnding7;
	public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorSquaresEnding7 => _sharedTables.HeuristicCombinedDivisorSquaresEnding7;
	public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorsEnding9 => _sharedTables.HeuristicCombinedDivisorsEnding9;
	public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorSquaresEnding9 => _sharedTables.HeuristicCombinedDivisorSquaresEnding9;

	private static readonly Accelerator[] _accelerators = AcceleratorPool.Shared.Accelerators;
	internal readonly int AcceleratorIndex;

	public HeuristicCombinedPrimeTesterAccelerator(int minBufferCapacity)
	{
		var acceleratorIndex = AcceleratorPool.Shared.Rent();
		AcceleratorIndex = acceleratorIndex;
		Accelerator accelerator = _accelerators[acceleratorIndex];
		Context = accelerator.Context;

		Accelerator = accelerator;
		EnsureCapacity(minBufferCapacity);

		HeuristicCombinedTrialDivisionKernel = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, byte, ArrayView<int>, ulong, ulong, HeuristicGpuCombinedDivisorTables>(PrimeTesterKernels.HeuristicTrialCombinedDivisionKernel));

		AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		
		// GpuStaticTableInitializer.EnsureStaticTables(accelerator, kernels, stream);

		_sharedTables = HeuristicGpuCombinedTables.GetStaticTables(acceleratorIndex);
		DivisorTables = _sharedTables.CreateHeuristicDivisorTables();

		stream.Synchronize();
		AcceleratorStreamPool.Return(acceleratorIndex, stream); 
	}

	public void EnsureCapacity(int minCapacity)
	{
		if (minCapacity <= BufferCapacity)
		{
			return;
		}

		Input?.Dispose();
		Output?.Dispose();
		HeuristicFlag?.Dispose();

		Input = Accelerator.Allocate1D<ulong>(minCapacity);
		Output = Accelerator.Allocate1D<byte>(minCapacity);
		HeuristicFlag = Accelerator.Allocate1D<int>(minCapacity);

		BufferCapacity = minCapacity;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void Return() => Return(this);

	public void DisposeResources()
	{
		Input?.Dispose();
		Output?.Dispose();
		HeuristicFlag?.Dispose();
		// ReleaseSharedTables(_sharedTables);
		// These resources are shared between GPU leases
		// Accelerator.Dispose();
		// AcceleratorContext.Dispose();
	}
}
