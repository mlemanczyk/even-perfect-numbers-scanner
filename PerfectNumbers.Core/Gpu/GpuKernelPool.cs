using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public class GpuKernelPool
{
	public static readonly KernelContainer Kernels = CreateKernels();

	public static T InitOnce<T>(ref T? target, Func<T> valueFactory) where T : class
	{
		if (target is { } value)
		{
			return value;
		}

		var newValue = valueFactory();
		return Interlocked.CompareExchange(ref target, newValue, null) ?? newValue;
	}

	private static KernelContainer CreateKernels()
	{
		var kernels = new KernelContainer();
		var stream = SharedGpuContext.Accelerator.CreateStream();
		PreloadStaticTables(kernels, stream);
		stream.Synchronize();
		stream.Dispose();
		return kernels;
	}

	private static KernelContainer CreateKernels(AcceleratorStream stream)
	{
		KernelContainer kernels = new();
		PreloadStaticTables(kernels, stream);
		return Kernels;
	}

	internal static void PreloadStaticTables(KernelContainer kernels, AcceleratorStream stream)
	{
		_ = EnsureSmallCyclesOnDevice(kernels, stream);
		_ = EnsureSmallPrimesOnDevice(kernels, stream);
		_ = EnsureSmallPrimeFactorTables(kernels, stream);
	}

	public static ArrayView1D<ulong, Stride1D.Dense> GetSmallCyclesOnDevice() => Kernels.SmallCycles!;

	// Ensures the small cycles table is uploaded to the device for the given accelerator.
	// Returns the ArrayView to pass into kernels (when kernels are extended to accept it).
	private static ArrayView1D<ulong, Stride1D.Dense> EnsureSmallCyclesOnDevice(KernelContainer kernels, AcceleratorStream stream)
	{
		if (kernels.SmallCycles is { } buffer)
		{
			return buffer.View;
		}

		var host = MersenneDivisorCycles.Shared.ExportSmallCyclesSnapshot(); // TODO: Preload this device buffer during startup and keep it immutable so we can delete the lock above in favor of the preloaded snapshot.
		var device = SharedGpuContext.Accelerator.Allocate1D<ulong>(host.Length);
		device.View.CopyFromCPU(stream, host);
		kernels.SmallCycles = device;
		return device.View;
	}

	public static ResiduePrimeViews GetSmallPrimesOnDevice() => new(
		Kernels.SmallPrimesLastOne!,
		Kernels.SmallPrimesLastSeven!,
		Kernels.SmallPrimesPow2LastOne!,
		Kernels.SmallPrimesPow2LastSeven!
	);

	private static ResiduePrimeViews EnsureSmallPrimesOnDevice(KernelContainer kernels, AcceleratorStream stream)
	{
		if (kernels.SmallPrimesLastOne is { } lastOne &&
			kernels.SmallPrimesLastSeven is { } lastSeven &&
			kernels.SmallPrimesPow2LastOne is { } lastOnePow2 &&
			kernels.SmallPrimesPow2LastSeven is { } lastSevenPow2)
		{
			return new ResiduePrimeViews(lastOne.View, lastSeven.View, lastOnePow2.View, lastSevenPow2.View);
		}

		if (kernels.SmallPrimesLastOne is { } existingLastOne &&
			kernels.SmallPrimesLastSeven is { } existingLastSeven &&
			kernels.SmallPrimesPow2LastOne is { } existingLastOnePow2 &&
			kernels.SmallPrimesPow2LastSeven is { } existingLastSevenPow2)
		{
			return new ResiduePrimeViews(existingLastOne.View, existingLastSeven.View, existingLastOnePow2.View, existingLastSevenPow2.View);
		}

		Accelerator accelerator = SharedGpuContext.Accelerator;
		var hostLastOne = PrimesGenerator.SmallPrimesLastOne;
		var hostLastSeven = PrimesGenerator.SmallPrimesLastSeven;
		var hostLastOnePow2 = PrimesGenerator.SmallPrimesPow2LastOne;
		var hostLastSevenPow2 = PrimesGenerator.SmallPrimesPow2LastSeven;

		var deviceLastOne = accelerator.Allocate1D<uint>(hostLastOne.Length);
		deviceLastOne.View.CopyFromCPU(stream, hostLastOne);
		var deviceLastSeven = accelerator.Allocate1D<uint>(hostLastSeven.Length);
		deviceLastSeven.View.CopyFromCPU(stream, hostLastSeven);
		var deviceLastOnePow2 = accelerator.Allocate1D<ulong>(hostLastOnePow2.Length);
		deviceLastOnePow2.View.CopyFromCPU(stream, hostLastOnePow2);
		var deviceLastSevenPow2 = accelerator.Allocate1D<ulong>(hostLastSevenPow2.Length);
		deviceLastSevenPow2.View.CopyFromCPU(stream, hostLastSevenPow2);

		kernels.SmallPrimesLastOne = deviceLastOne;
		kernels.SmallPrimesLastSeven = deviceLastSeven;
		kernels.SmallPrimesPow2LastOne = deviceLastOnePow2;
		kernels.SmallPrimesPow2LastSeven = deviceLastSevenPow2;

		return new ResiduePrimeViews(deviceLastOne.View, deviceLastSeven.View, deviceLastOnePow2.View, deviceLastSevenPow2.View);
	}

	public static SmallPrimeFactorViews GetSmallPrimeFactorTables() => new(
		Kernels.SmallPrimeFactorsPrimes!,
		Kernels.SmallPrimeFactorsSquares!
	);
	
	private static SmallPrimeFactorViews EnsureSmallPrimeFactorTables(KernelContainer kernels, AcceleratorStream stream)
	{
		if (kernels.SmallPrimeFactorsPrimes is { } primeBuffer && kernels.SmallPrimeFactorsSquares is { } squareBuffer)
		{
			return new SmallPrimeFactorViews(primeBuffer, squareBuffer);
		}

		Accelerator accelerator = SharedGpuContext.Accelerator;
		var hostPrimes = PrimesGenerator.SmallPrimes;
		var hostSquares = PrimesGenerator.SmallPrimesPow2;

		int length = hostPrimes.Length;
		var devicePrimes = accelerator.Allocate1D<uint>(length);
		devicePrimes.View.CopyFromCPU(stream, hostPrimes);
		var deviceSquares = accelerator.Allocate1D<ulong>(length);
		deviceSquares.View.CopyFromCPU(stream, hostSquares);

		kernels.SmallPrimeFactorsPrimes = devicePrimes;
		kernels.SmallPrimeFactorsSquares = deviceSquares;

		return new SmallPrimeFactorViews(devicePrimes, deviceSquares);
	}

	public static GpuKernelLease Rent()
	{
		GpuPrimeWorkLimiter.Acquire();
		var stream = SharedGpuContext.Accelerator.CreateStream();
		return GpuKernelLease.Rent(stream, Kernels);
	}

	/// <summary>
	/// Runs a GPU action with an acquired accelerator and stream.
	/// </summary>
	/// <param name="action">Action to run with (Accelerator, Stream).</param>
	public static void Run(Action<Accelerator, AcceleratorStream> action)
	{
		var lease = Rent();
		var accelerator = SharedGpuContext.Accelerator;
		var stream = lease.Stream;
		action(accelerator, stream);
		lease.Dispose();
	}
}
