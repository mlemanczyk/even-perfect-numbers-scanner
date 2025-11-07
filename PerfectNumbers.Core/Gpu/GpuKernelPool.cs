using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;
using static PerfectNumbers.Core.Gpu.GpuContextPool;

namespace PerfectNumbers.Core.Gpu;

public class GpuKernelPool
{
	private static readonly ConcurrentDictionary<Accelerator, KernelContainer> _kernels = new();

	public static T InitOnce<T>(ref T? target, Func<T> valueFactory) where T : class
	{
		if (target is { } value)
		{
			return value;
		}

		var newValue = valueFactory();
		return Interlocked.CompareExchange(ref target, newValue, null) ?? newValue;
	}

	internal static KernelContainer GetKernels(Accelerator accelerator, AcceleratorStream stream)
	{
		return _kernels.GetOrAdd(accelerator, (accelerator) =>
		{
			var kernels = new KernelContainer();
			PreloadStaticTables(kernels, accelerator, stream);
			return kernels;
		});
	}

	internal static void PreloadStaticTables(KernelContainer kernels, Accelerator accelerator, AcceleratorStream stream)
	{
		EnsureSmallCyclesOnDevice(kernels, accelerator, stream);
		EnsureSmallPrimesOnDevice(kernels, accelerator, stream);
		EnsureSmallPrimeFactorTables(kernels, accelerator, stream);
	}

	// Ensures the small cycles table is uploaded to the device for the given accelerator.
	// Returns the ArrayView to pass into kernels (when kernels are extended to accept it).
	public static ArrayView1D<ulong, Stride1D.Dense> EnsureSmallCyclesOnDevice(KernelContainer kernels, Accelerator accelerator, AcceleratorStream stream)
	{
		if (kernels.SmallCycles is { } buffer)
		{
			return buffer.View;
		}

		var host = MersenneDivisorCycles.Shared.ExportSmallCyclesSnapshot(); // TODO: Preload this device buffer during startup and keep it immutable so we can delete the lock above in favor of the preloaded snapshot.
		var device = accelerator.Allocate1D<ulong>(host.Length);
		device.View.CopyFromCPU(stream, host);
		kernels.SmallCycles = device;
		return device.View;
	}

	public static ResiduePrimeViews EnsureSmallPrimesOnDevice(KernelContainer kernels, Accelerator accelerator, AcceleratorStream stream)
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

	public static SmallPrimeFactorViews EnsureSmallPrimeFactorTables(KernelContainer kernels, Accelerator accelerator, AcceleratorStream stream)
	{
		if (kernels.SmallPrimeFactorsPrimes is { } primeBuffer && kernels.SmallPrimeFactorsSquares is { } squareBuffer)
		{
			return new SmallPrimeFactorViews(primeBuffer, squareBuffer, (int)primeBuffer.Length);
		}

		var hostPrimes = PrimesGenerator.SmallPrimes;
		var hostSquares = PrimesGenerator.SmallPrimesPow2;

		var devicePrimes = accelerator.Allocate1D<uint>(hostPrimes.Length);
		devicePrimes.View.CopyFromCPU(stream, hostPrimes);
		var deviceSquares = accelerator.Allocate1D<ulong>(hostSquares.Length);
		deviceSquares.View.CopyFromCPU(stream, hostSquares);

		kernels.SmallPrimeFactorsPrimes = devicePrimes;
		kernels.SmallPrimeFactorsSquares = deviceSquares;

		return new SmallPrimeFactorViews(devicePrimes, deviceSquares, hostPrimes.Length);
	}

	public static GpuKernelLease Rent()
	{
		GpuPrimeWorkLimiter.Acquire();
		var gpu = GpuContextPool.Rent();
		var stream = gpu.Accelerator.CreateStream();
		var kernels = GetKernels(gpu.Accelerator, stream);
		return GpuKernelLease.Rent(gpu, stream, kernels);
	}

	/// <summary>
	/// Runs a GPU action with an acquired accelerator and stream.
	/// </summary>
	/// <param name="action">Action to run with (Accelerator, Stream).</param>
	public static void Run(Action<Accelerator, AcceleratorStream> action)
	{
		var lease = Rent();
		var accelerator = lease.Accelerator;
		var stream = lease.Stream;
		action(accelerator, stream);
		lease.Dispose();
	}
}
