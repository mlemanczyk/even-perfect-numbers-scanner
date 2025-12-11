using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu.Kernels;

namespace PerfectNumbers.Core.Gpu.Accelerators;

internal sealed class HeuristicGroupABPrimeTesterAccelerator
{
	#region Pool

	[ThreadStatic]
	private static Queue<HeuristicGroupABPrimeTesterAccelerator>? Pool;

	internal static void WarmUp()
	{
		Accelerator[]? accelerators = AcceleratorPool.Shared.Accelerators;
		int acceleratorCount = accelerators.Length;
		for (var i = 0; i < acceleratorCount; i++)
		{
			Console.WriteLine($"Preparing heuristic accelerator {i}...");
			var accelerator = accelerators[i];
			AcceleratorStream stream = accelerator.CreateStream();
			_ = HeuristicGroupABGpuTables.EnsureStaticTables(i, stream);
			// _ = GpuKernelPool.GetOrAddKernels(accelerator, stream, KernelType.None);
			// KernelContainer kernels = GpuKernelPool.GetOrAddKernels(accelerator, stream);
			// GpuStaticTableInitializer.EnsureStaticTables(accelerator, kernels, stream);
			stream.Synchronize();
			stream.Dispose();
			Console.WriteLine($"Heuristic accelerator {i} is ready");
		}
	}

	internal static HeuristicGroupABPrimeTesterAccelerator Rent(int minBufferCapacity = 1)
	{
		var pool = Pool ??= new();
		if (!pool.TryDequeue(out var lease))
		{
			lease = new HeuristicGroupABPrimeTesterAccelerator(minBufferCapacity);
		}
		else
		{
			lease.EnsureCapacity(minBufferCapacity);
		}

		return lease;
	}

	internal static void Return(HeuristicGroupABPrimeTesterAccelerator lease)
	{
		var pool = Pool ??= new();
		pool.Enqueue(lease);
	}

	internal static void DisposeAll()
	{
		if (Pool != null)
		{
			while (Pool.TryDequeue(out var lease))
			{
				lease.DisposeResources();
			}
		}
	}

	internal static void Clear()
	{
		if (Pool != null)
		{
			var retained = new List<HeuristicGroupABPrimeTesterAccelerator>();
			while (Pool.TryDequeue(out var lease))
			{
				lease.DisposeResources();
			}
		}
	}

	#endregion

	public readonly HeuristicGroupABGpuViews DivisorTables;
	internal readonly Context Context;
	public readonly Accelerator Accelerator;

	[ThreadStatic]
	public static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>>? SmallPrimeSieveKernel;

	internal static Action<AcceleratorStream, Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>> GetSmallPrimeSieveKernel(Accelerator accelerator)
	{
		var pool = SmallPrimeSieveKernel ??= [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}

		cached = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>(PrimeTesterKernels.SmallPrimeSieveKernel)).CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>>();
		pool[accelerator] = cached;
		return cached;
	}

	public readonly Kernel HeuristicTrialDivisionKernel;

	public MemoryBuffer1D<ulong, Stride1D.Dense>? Input = null;
	public MemoryBuffer1D<byte, Stride1D.Dense>? Output = null;
	public MemoryBuffer1D<int, Stride1D.Dense>? HeuristicFlag = null!;
	public int BufferCapacity;

	private HeuristicGroupABGpuTables _sharedTables;
	public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupADivisors => _sharedTables.HeuristicGroupADivisors;
	public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupADivisorSquares => _sharedTables.HeuristicGroupADivisorSquares;
	public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorsEnding1 => _sharedTables.HeuristicGroupBDivisorsEnding1;
	public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorSquaresEnding1 => _sharedTables.HeuristicGroupBDivisorSquaresEnding1;
	public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorsEnding7 => _sharedTables.HeuristicGroupBDivisorsEnding7;
	public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorSquaresEnding7 => _sharedTables.HeuristicGroupBDivisorSquaresEnding7;
	public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorsEnding9 => _sharedTables.HeuristicGroupBDivisorsEnding9;
	public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorSquaresEnding9 => _sharedTables.HeuristicGroupBDivisorSquaresEnding9;

	private static readonly Accelerator[] _accelerators = AcceleratorPool.Shared.Accelerators;
	internal readonly int AcceleratorIndex;

	internal HeuristicGroupABPrimeTesterAccelerator(int minBufferCapacity)
	{
		var acceleratorIndex = AcceleratorPool.Shared.Rent();
		AcceleratorIndex = acceleratorIndex;
		Accelerator accelerator = _accelerators[acceleratorIndex];
		Context = accelerator.Context;

		Accelerator = accelerator;
		EnsureCapacity(minBufferCapacity);

		HeuristicTrialDivisionKernel = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ulong, ulong, HeuristicGpuDivisorTableKind, HeuristicGroupABGpuViews>(PrimeTesterKernels.HeuristicTrialGroupABDivisionKernel));

		AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		
		// GpuStaticTableInitializer.EnsureStaticTables(accelerator, kernels, stream);

		_sharedTables = HeuristicGroupABGpuTables.EnsureStaticTables(acceleratorIndex, stream);
		DivisorTables = _sharedTables.CreateViews();

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

	public void Dispose()
	{
		Return(this);
	}

	internal void DisposeResources()
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
