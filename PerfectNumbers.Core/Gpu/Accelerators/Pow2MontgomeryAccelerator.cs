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
	public static Pow2MontgomeryAccelerator Rent()
	{
		var pool = _pool ??= [];
		if (!pool.TryDequeue(out var gpu))
		{
			Accelerator accelerator = _accelerators.Rent();
			gpu = new(accelerator, PerfectNumberConstants.DefaultFactorsBuffer);
		}
		else
		{
			gpu.Stream = gpu.Accelerator.CreateStream();
		}

		return gpu;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static void Return(Pow2MontgomeryAccelerator gpu)
	{
		gpu.Stream!.Dispose();
		gpu.Stream = null;
		_pool!.Enqueue(gpu);
	}

	#endregion

	public readonly Accelerator Accelerator;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> Input;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> Output;
	public MemoryBuffer1D<KeyValuePair<ulong, int>, Stride1D.Dense> Pow2ModEntriesToTestOnDevice;
	public Dictionary<ulong, int> Pow2ModEntriesToTestOnHost = [];
	public AcceleratorStream? Stream;

	public readonly Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>> ConvertToStandardKernel;
	public readonly Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>> KeepMontgomeryKernel;

	public readonly Kernel CheckFactorsKernel;

	public void EnsureCapacity(int factorsCount)
	{
		if (factorsCount > Pow2ModEntriesToTestOnDevice.Length)
		{
			Accelerator accelerator = Accelerator;
			Pow2ModEntriesToTestOnDevice.Dispose();
			Pow2ModEntriesToTestOnDevice = accelerator.Allocate1D<KeyValuePair<ulong, int>>(factorsCount);
		}
	}

	public Pow2MontgomeryAccelerator(Accelerator accelerator, int factorsCount)
	{
		Accelerator = accelerator;
		Stream = accelerator.CreateStream();

		Input = accelerator.Allocate1D<ulong>(1);
		Output = accelerator.Allocate1D<ulong>(1);
		Pow2ModEntriesToTestOnDevice = accelerator.Allocate1D<KeyValuePair<ulong, int>>(factorsCount);

		var keepKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>(Pow2MontgomeryKernels.Pow2MontgomeryKernelKeepMontgomery);
		var keepLauncher = KernelUtil.GetKernel(keepKernel);

		KeepMontgomeryKernel = keepLauncher.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>>();

		var convertKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>(Pow2MontgomeryKernels.Pow2MontgomeryKernelConvertToStandard);
		var convertLauncher = KernelUtil.GetKernel(convertKernel);

		ConvertToStandardKernel = convertLauncher.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>>();

		var checkFactorsKernel = accelerator.LoadStreamKernel<int, ulong, ArrayView1D<KeyValuePair<ulong, int>, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>(PrimeOrderGpuHeuristics.CheckFactorsKernel);
		CheckFactorsKernel = KernelUtil.GetKernel(checkFactorsKernel);
	}

	public void Dispose()
	{
		Input.Dispose();
		Output.Dispose();
		Stream?.Dispose();
		Pow2ModEntriesToTestOnDevice.Dispose();
		// Accelerator is shared with other threads
		// Accelerator.Dispose();
	}
}