using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu.Accelerators;

internal sealed class PrimeTesterByLastDigitAccelerator
{
	#region Pool

	[ThreadStatic]
	private static Queue<PrimeTesterByLastDigitAccelerator>? Pool;

	// internal static void EnsureStaticTables(Accelerator accelerator)
	// {
	// 	var tables = CreateSharedTables(accelerator);
	// }

	internal static void WarmUp()
	{
		Accelerator[]? accelerators = AcceleratorPool.Shared.Accelerators;
		int acceleratorCount = accelerators.Length;
		Span<AcceleratorStream> streams = new(new AcceleratorStream[acceleratorCount]);
		for (var i = 0; i < acceleratorCount; i++)
		{
			Console.WriteLine($"Preparing accelerator {i}...");
			var accelerator = accelerators[i];
			AcceleratorStream stream = accelerator.CreateStream();
			LastDigitGpuTables.EnsureStaticTables(accelerator, stream);
			// SharedHeuristicGpuTables.EnsureStaticTables(accelerator, stream);
			// _ = GpuKernelPool.GetOrAddKernels(accelerator, stream, KernelType.None);
			// KernelContainer kernels = GpuKernelPool.GetOrAddKernels(accelerator, stream);
			// GpuStaticTableInitializer.EnsureStaticTables(accelerator, kernels, stream);
			streams[i] = stream;
		}

		for (var i = 0; i < acceleratorCount; i++)
		{
			AcceleratorStream stream = streams[i];
			stream.Synchronize();
			stream.Dispose();
			Console.WriteLine($"Accelerator {i} is ready");
		}
	}

	internal static PrimeTesterByLastDigitAccelerator Rent(int minBufferCapacity = 1)
	{
		var pool = Pool ??= new();
		if (!pool.TryDequeue(out var lease))
		{
			lease = new PrimeTesterByLastDigitAccelerator(minBufferCapacity);
		}
		else
		{
			lease.EnsureCapacity(minBufferCapacity);
		}

		return lease;
	}

	internal static void Return(PrimeTesterByLastDigitAccelerator lease)
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
			var retained = new List<PrimeTesterByLastDigitAccelerator>();
			while (Pool.TryDequeue(out var lease))
			{
				lease.DisposeResources();
			}
		}
	}

	#endregion

	private readonly LastDigitGpuTables _sharedTables;
	internal readonly Context Context;
	public readonly Accelerator Accelerator;
	private AcceleratorStream? _stream;
	public AcceleratorStream Stream
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		get => _stream ?? Accelerator.CreateStream();
	}

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

	[ThreadStatic]
	public static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView<int>, ulong, ulong, HeuristicGpuDivisorTableKind, HeuristicGpuDivisorTables>>? HeuristicTrialDivisionKernel;


	public MemoryBuffer1D<ulong, Stride1D.Dense> Input = null!;
	public MemoryBuffer1D<byte, Stride1D.Dense> Output = null!;
	public MemoryBuffer1D<int, Stride1D.Dense> HeuristicFlag = null!;
	public int BufferCapacity;

	public MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastOne => _sharedTables.DevicePrimesLastOne;
	public MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastSeven => _sharedTables.DevicePrimesLastSeven;
	public MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastThree => _sharedTables.DevicePrimesLastThree;
	public MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastNine => _sharedTables.DevicePrimesLastNine;
	public MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastOne => _sharedTables.DevicePrimesPow2LastOne;
	public MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastSeven => _sharedTables.DevicePrimesPow2LastSeven;
	public MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastThree => _sharedTables.DevicePrimesPow2LastThree;
	public MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastNine => _sharedTables.DevicePrimesPow2LastNine;

	internal PrimeTesterByLastDigitAccelerator(int minBufferCapacity)
	{
		Accelerator accelerator = AcceleratorPool.Shared.Rent();
		Context = accelerator.Context;

		Accelerator = accelerator;
		AcceleratorStream stream = accelerator.CreateStream();

		var sharedTables = LastDigitGpuTables.EnsureStaticTables(accelerator, stream);
		_sharedTables = sharedTables;

		EnsureCapacity(minBufferCapacity);
		stream.Synchronize();
		stream.Dispose();
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

		// lock (Accelerator)
		{
			Input = Accelerator.Allocate1D<ulong>(minCapacity);
			Output = Accelerator.Allocate1D<byte>(minCapacity);
			HeuristicFlag = Accelerator.Allocate1D<int>(minCapacity);
		}

		BufferCapacity = minCapacity;
	}

	public void Dispose()
	{
		_stream?.Dispose();
		_stream = null;
		Return(this);
	}

	internal void DisposeResources()
	{
		Input?.Dispose();
		Output?.Dispose();
		HeuristicFlag?.Dispose();
		_stream?.Dispose();
		_stream = null;
		// ReleaseSharedTables(_sharedTables);
		// These resources are shared between GPU leases
		// Accelerator.Dispose();
		// AcceleratorContext.Dispose();
	}
}
