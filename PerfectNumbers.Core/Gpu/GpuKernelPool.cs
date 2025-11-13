using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public class GpuKernelPool
{
	[ThreadStatic]
	private static Dictionary<Accelerator, KernelContainer>? _pool;

	public static T InitOnce<T>(ref T? target, Func<T> valueFactory) where T : class
	{
		if (target is { } value)
		{
			return value;
		}

		var newValue = valueFactory();
		return Interlocked.CompareExchange(ref target, newValue, null) ?? newValue;
	}

	internal static KernelContainer GetOrAddKernels(Accelerator accelerator, AcceleratorStream stream, KernelType kernelType)
	{
		var pool = _pool ??= [];
		if (!pool.TryGetValue(accelerator, out var kernels))
		{
			kernels = new KernelContainer();			
		}

		if (kernelType.HasFlag(KernelType.OrderKernelScan))
		{
			InitOrderKernelScan(accelerator, kernels);
		}

		if (kernelType.HasFlag(KernelType.Pow2ModKernelScan))
		{
			InitPow2ModKernelScan(accelerator, kernels);
		}

		if (kernelType.HasFlag(KernelType.IncrementalKernelScan))
		{
			InitIncrementalKernelScan(accelerator, kernels);
		}

		if (kernelType.HasFlag(KernelType.IncrementalOrderKernelScan))
		{
			IncrementalOrderKernelScan(accelerator, kernels);
		}

		if (kernelType.HasFlag(KernelType.Pow2ModOrderKernelScan))
		{
			InitPow2ModOrderKernelScan(accelerator, kernels);
		}

		if (kernelType.HasFlag(KernelType.EvaluateSpecialMaxCandidatesKernel))
		{
			InitEvaluateSpecialMaxCandidatesKernel(accelerator, kernels);
		}

		if (kernelType.HasFlag(KernelType.SmallPrimeFactorKernelScan))
		{
			InitSmallPrimeFactorKernelScan(accelerator, kernels);
		}

		PreloadStaticTables(accelerator, kernels, stream, kernelType);

		pool[accelerator] = kernels;
		return kernels;

		static void InitOrderKernelScan(Accelerator accelerator, KernelContainer kernels)
		{
			InitOnce(ref kernels.Order, () =>
			{
				var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ulong, ulong, ArrayView<GpuUInt128>, ArrayView<ulong>>(OrderKernels.OrderKernelScan);

				var kernel = KernelUtil.GetKernel(loaded);

				return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, ulong, ArrayView<GpuUInt128>, ArrayView<ulong>>>();
			});
		}

		static void InitPow2ModKernelScan(Accelerator accelerator, KernelContainer kernels)
		{
			InitOnce(ref kernels.Pow2Mod, () =>
			{
				var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>(Pow2ModKernels.Pow2ModKernelScan);

				var kernel = KernelUtil.GetKernel(loaded);

				return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>>();
			});
		}

		static void InitIncrementalKernelScan(Accelerator accelerator, KernelContainer kernels)
		{
			InitOnce(ref kernels.Incremental, () =>
			{
				var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ulong, ulong, ulong, ulong, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>>(IncrementalKernels.IncrementalKernelScan);

				var kernel = KernelUtil.GetKernel(loaded);

				return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ulong, ulong, ulong, ulong, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>>>();
			});
		}

		static void IncrementalOrderKernelScan(Accelerator accelerator, KernelContainer kernels)
		{
			InitOnce(ref kernels.IncrementalOrder, () =>
			{
				var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>>(IncrementalKernels.IncrementalOrderKernelScan);

				var kernel = KernelUtil.GetKernel(loaded);

				return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte,
				ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>>>();
			});
		}

		static void InitPow2ModOrderKernelScan(Accelerator accelerator, KernelContainer kernels)
		{
			InitOnce(ref kernels.Pow2ModOrder, () =>
			{
				var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>>(Pow2ModKernels.Pow2ModOrderKernelScan);

				var kernel = KernelUtil.GetKernel(loaded);

				return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>>>();
			});
		}

		static void InitEvaluateSpecialMaxCandidatesKernel(Accelerator accelerator, KernelContainer kernels)
		{
			InitOnce(ref kernels.SpecialMax, () =>
			{
				var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ulong, ArrayView1D<ulong, Stride1D.Dense>, int, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>(PrimeOrderGpuHeuristics.EvaluateSpecialMaxCandidatesKernel);

				var kernel = KernelUtil.GetKernel(loaded);

				return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, ArrayView1D<ulong, Stride1D.Dense>, int, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>>();
			});
		}

		static void InitSmallPrimeFactorKernelScan(Accelerator accelerator, KernelContainer kernels)
		{
			var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ulong, uint, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>(SmallPrimeFactorKernels.SmallPrimeFactorKernelScan);

			kernels.SmallPrimeFactor = KernelUtil.GetKernel(loaded);
		}
	}

	internal static void PreloadStaticTables(Accelerator accelerator, KernelContainer kernels, AcceleratorStream stream, KernelType kernelType)
	{
		if (kernelType.HasFlag(KernelType.SmallCycles))
		{
			EnsureSmallCyclesOnDevice(accelerator, kernels, stream);
		}

		if (kernelType.HasFlag(KernelType.SmallPrimes))
		{
			EnsureSmallPrimesOnDevice(accelerator, kernels, stream);
		}
		
		if (kernelType.HasFlag(KernelType.SmallPrimeFactorKernelScan))
		{
			EnsureSmallPrimeFactorTables(accelerator, kernels, stream);			
		}
	}

	public static ArrayView1D<ulong, Stride1D.Dense> GetSmallCyclesOnDevice(KernelContainer kernels) => kernels.SmallCycles!;

	// Ensures the small cycles table is uploaded to the device for the given accelerator.
	// Returns the ArrayView to pass into kernels (when kernels are extended to accept it).
	private static void EnsureSmallCyclesOnDevice(Accelerator accelerator, KernelContainer kernels, AcceleratorStream stream)
	{
		if (kernels.SmallCycles is { })
		{
			return;
		}

		var host = MersenneDivisorCycles.Shared.ExportSmallCyclesSnapshot();
		MemoryBuffer1D<ulong, Stride1D.Dense>? device;

		// lock (accelerator)
		{
			device = accelerator.Allocate1D<ulong>(host.Length);
		}

		device.View.CopyFromCPU(stream, host);
		kernels.SmallCycles = device;
	}

	public static ResiduePrimeViews GetSmallPrimesOnDevice(KernelContainer kernels) => new(
		kernels.SmallPrimesLastOne!,
		kernels.SmallPrimesLastSeven!,
		kernels.SmallPrimesPow2LastOne!,
		kernels.SmallPrimesPow2LastSeven!
	);

	private static void EnsureSmallPrimesOnDevice(Accelerator accelerator, KernelContainer kernels, AcceleratorStream stream)
	{
		if (kernels.SmallPrimesLastOne is { } &&
			kernels.SmallPrimesLastSeven is { } &&
			kernels.SmallPrimesPow2LastOne is { } &&
			kernels.SmallPrimesPow2LastSeven is { })
		{
			return;
		}

		if (kernels.SmallPrimesLastOne is { } &&
			kernels.SmallPrimesLastSeven is { } &&
			kernels.SmallPrimesPow2LastOne is { } &&
			kernels.SmallPrimesPow2LastSeven is { })
		{
			return;
		}

		var hostLastOne = PrimesGenerator.SmallPrimesLastOne;
		var hostLastSeven = PrimesGenerator.SmallPrimesLastSeven;
		var hostLastOnePow2 = PrimesGenerator.SmallPrimesPow2LastOne;
		var hostLastSevenPow2 = PrimesGenerator.SmallPrimesPow2LastSeven;

		MemoryBuffer1D<uint, Stride1D.Dense>? deviceLastOne;
		MemoryBuffer1D<uint, Stride1D.Dense>? deviceLastSeven;
		MemoryBuffer1D<ulong, Stride1D.Dense>? deviceLastOnePow2;
		MemoryBuffer1D<ulong, Stride1D.Dense>? deviceLastSevenPow2;

		// lock (accelerator)
		// {
		deviceLastOne = accelerator.Allocate1D<uint>(hostLastOne.Length);
		deviceLastSeven = accelerator.Allocate1D<uint>(hostLastSeven.Length);
		deviceLastOnePow2 = accelerator.Allocate1D<ulong>(hostLastOnePow2.Length);
		deviceLastSevenPow2 = accelerator.Allocate1D<ulong>(hostLastSevenPow2.Length);
		// }

		deviceLastOne.View.CopyFromCPU(stream, hostLastOne);
		deviceLastSeven.View.CopyFromCPU(stream, hostLastSeven);
		deviceLastOnePow2.View.CopyFromCPU(stream, hostLastOnePow2);
		deviceLastSevenPow2.View.CopyFromCPU(stream, hostLastSevenPow2);
		kernels.SmallPrimesLastOne = deviceLastOne;
		kernels.SmallPrimesLastSeven = deviceLastSeven;
		kernels.SmallPrimesPow2LastOne = deviceLastOnePow2;
		kernels.SmallPrimesPow2LastSeven = deviceLastSevenPow2;
	}

	public static SmallPrimeFactorViews GetSmallPrimeFactorTables(KernelContainer kernels) => new(
		kernels.SmallPrimeFactorsPrimes!,
		kernels.SmallPrimeFactorsSquares!
	);

	private static void EnsureSmallPrimeFactorTables(Accelerator accelerator, KernelContainer kernels, AcceleratorStream stream)
	{
		if (kernels.SmallPrimeFactorsPrimes is { } && kernels.SmallPrimeFactorsSquares is { })
		{
			return;
		}

		var hostPrimes = PrimesGenerator.SmallPrimes;
		var hostSquares = PrimesGenerator.SmallPrimesPow2;

		int length = hostPrimes.Length;
		MemoryBuffer1D<uint, Stride1D.Dense>? devicePrimes;
		MemoryBuffer1D<ulong, Stride1D.Dense>? deviceSquares;

		// lock (accelerator)
		{
			devicePrimes = accelerator.Allocate1D<uint>(length);
			deviceSquares = accelerator.Allocate1D<ulong>(length);
		}

		devicePrimes.View.CopyFromCPU(stream, hostPrimes);
		deviceSquares.View.CopyFromCPU(stream, hostSquares);

		kernels.SmallPrimeFactorsPrimes = devicePrimes;
		kernels.SmallPrimeFactorsSquares = deviceSquares;
	}

	/// <summary>
	/// Runs a GPU action with an acquired accelerator and stream.
	/// </summary>
	/// <param name="action">Action to run with (Accelerator, Stream).</param>
	public static void Run(Action<Accelerator, AcceleratorStream> action)
	{
		// GpuPrimeWorkLimiter.Acquire();
		var accelerator = SharedGpuContext.CreateAccelerator();
		var stream = accelerator.CreateStream();
		action(accelerator, stream);
		stream.Synchronize();
		stream.Dispose();
		accelerator.Dispose();
		// GpuPrimeWorkLimiter.Release();
	}
}
