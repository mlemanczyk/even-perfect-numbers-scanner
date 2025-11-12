using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu.Kernels;

namespace PerfectNumbers.Core.Gpu.Accelerators;

public sealed class Pow2MontgomeryAccelerator
{
	#region Pool

	[ThreadStatic]
	private static Queue<Pow2MontgomeryAccelerator>? _pool;

	private static AcceleratorPool _accelerators = new(PerfectNumberConstants.RollingAccelerators);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static Pow2MontgomeryAccelerator Rent(int primeTesterCapacity)
	{
		var pool = _pool ??= [];
		if (!pool.TryDequeue(out var gpu))
		{
			Accelerator accelerator = _accelerators.Rent();
			gpu = new(accelerator, PerfectNumberConstants.DefaultFactorsBuffer, primeTesterCapacity);
		}
		else
		{
			// gpu.Stream = gpu.Accelerator.CreateStream();
			gpu.EnsureCapacity(PerfectNumberConstants.DefaultFactorsBuffer, primeTesterCapacity);
		}

		return gpu;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static void Return(Pow2MontgomeryAccelerator gpu)
	{
		// gpu.Stream!.Dispose();
		// gpu.Stream = null;
		_pool!.Enqueue(gpu);
	}

	internal static void Clear()
	{
		Queue<Pow2MontgomeryAccelerator>? pool = _pool;
		if (pool != null)
		{
			while (pool.TryDequeue(out var lease))
			{
				lease.Dispose();
			}
		}
	}

	internal static void DisposeAll()
	{
		Queue<Pow2MontgomeryAccelerator>? pool = _pool;
		if (pool != null)
		{
			while (pool.TryDequeue(out var lease))
			{
				lease.Dispose();
			}
		}
	}

	internal static void WarmUp()
	{
		Accelerator[]? accelerators = _accelerators.Accelerators;
		int acceleratorCount = accelerators.Length;
		for (var i = 0; i < acceleratorCount; i++)
		{
			Console.WriteLine($"Preparing accelerator {i}...");
			var accelerator = accelerators[i];
			AcceleratorStreamPool.WarmUp(accelerator);
			// AcceleratorStream stream = AcceleratorStreamPool.Rent(accelerator);
			AcceleratorStream stream = accelerator.CreateStream();
			LastDigitGpuTables.EnsureStaticTables(accelerator, stream);
			// SharedHeuristicGpuTables.EnsureStaticTables(accelerator, stream);
			// _ = GpuKernelPool.GetOrAddKernels(accelerator, stream, KernelType.None);
			// KernelContainer kernels = GpuKernelPool.GetOrAddKernels(accelerator, stream);
			// GpuStaticTableInitializer.EnsureStaticTables(accelerator, kernels, stream);
			stream.Synchronize();
			stream.Dispose();

			// AcceleratorStreamPool.Return(stream);
		}
	}

	#endregion

	public readonly Accelerator Accelerator;
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
	public readonly Kernel SmallPrimeSieveKernel;

	// public AcceleratorStream? Stream;

	public readonly Kernel CheckFactorsKernel;
	public readonly Kernel ConvertToStandardKernel;
	public readonly Kernel KeepMontgomeryKernel;
	public readonly Kernel PollardRhoKernel;

	[ThreadStatic]
	private static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>>? SmallPrimeSieveKernelPool;

	internal static Action<AcceleratorStream, Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>> GetSmallPrimeSieveKernel(Accelerator accelerator)
	{
		var pool = SmallPrimeSieveKernelPool ??= [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}

		cached = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>(PrimeTesterKernels.SmallPrimeSieveKernel)).CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>>();
		pool[accelerator] = cached;
		return cached;
	}

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
			PrimeTestInput?.Dispose();
			OutputByte?.Dispose();
			HeuristicFlag?.Dispose();

			// lock (Accelerator)
			{
				PrimeTestInput = Accelerator.Allocate1D<ulong>(primeTesterCapacity);
				OutputByte = Accelerator.Allocate1D<byte>(primeTesterCapacity);
				HeuristicFlag = Accelerator.Allocate1D<int>(primeTesterCapacity);
			}
		}
	}

	public Pow2MontgomeryAccelerator(Accelerator accelerator, int factorsCount, int primeTesterCapacity)
	{
		Accelerator = accelerator;
		AcceleratorStream stream = AcceleratorStreamPool.Rent(accelerator);
		// AcceleratorStream stream = accelerator.DefaultStream;
		// AcceleratorStream stream = accelerator.CreateStream();
		// Stream = stream;

		Input = accelerator.Allocate1D<ulong>(1);
		OutputUlong = accelerator.Allocate1D<ulong>(1);
		Pow2ModEntriesToTestOnDevice = accelerator.Allocate1D<KeyValuePair<ulong, int>>(factorsCount);
		PrimeTestInput = Accelerator.Allocate1D<ulong>(primeTesterCapacity);
		OutputByte = Accelerator.Allocate1D<byte>(primeTesterCapacity);
		HeuristicFlag = Accelerator.Allocate1D<int>(primeTesterCapacity);

		var sharedTables = LastDigitGpuTables.EnsureStaticTables(accelerator, stream);
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

		AcceleratorStreamPool.Return(stream);
	}

	public void Dispose()
	{
		Input.Dispose();
		OutputUlong.Dispose();
		// Stream?.Dispose();
		Pow2ModEntriesToTestOnDevice.Dispose();
		PrimeTestInput?.Dispose();
		OutputByte?.Dispose();
		HeuristicFlag?.Dispose();

		// Stream is disposed upon return to the queue, so we don't need to dispose it here.
		// Stream.Dispose();

		// These resources are shared between GPU leases
		// ReleaseSharedTables(_sharedTables);
		// Accelerator.Dispose();
		// Context.Dispose();
		// Accelerator.Dispose();
	}
}