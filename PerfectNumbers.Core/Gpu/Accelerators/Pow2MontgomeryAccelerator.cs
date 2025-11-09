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
			gpu = new(accelerator);
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
	public AcceleratorStream? Stream;

	public readonly Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>> ConvertToStandardKernel;
	public readonly Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>> KeepMontgomeryKernel;

	public Pow2MontgomeryAccelerator(Accelerator accelerator)
	{
		Accelerator = accelerator;
		Input = accelerator.Allocate1D<ulong>(1);
		Output = accelerator.Allocate1D<ulong>(1);
		Stream = accelerator.CreateStream();

		var keepKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>(Pow2MontgomeryKernels.Pow2MontgomeryKernelKeepMontgomery);
		var keepLauncher = KernelUtil.GetKernel(keepKernel);

		KeepMontgomeryKernel = keepLauncher.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>>();

		var convertKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>(Pow2MontgomeryKernels.Pow2MontgomeryKernelConvertToStandard);
		var convertLauncher = KernelUtil.GetKernel(convertKernel);

		ConvertToStandardKernel = convertLauncher.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>>();
	}

	public void Dispose()
	{
		Input.Dispose();
		Output.Dispose();
		Stream?.Dispose();
		// Accelerator is shared with other threads
		// Accelerator.Dispose();
	}
}
