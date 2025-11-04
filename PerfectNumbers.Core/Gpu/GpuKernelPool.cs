using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;
using static PerfectNumbers.Core.Gpu.GpuContextPool;

namespace PerfectNumbers.Core.Gpu;

public readonly struct ResidueAutomatonArgs
{
	public readonly ulong Q0M10;
	public readonly ulong Step10;
	public readonly ulong Q0M8;
	public readonly ulong Step8;
	public readonly ulong Q0M3;
	public readonly ulong Step3;
	public readonly ulong Q0M5;
	public readonly ulong Step5;

	public ResidueAutomatonArgs(ulong q0m10, ulong step10, ulong q0m8, ulong step8, ulong q0m3, ulong step3, ulong q0m5, ulong step5)
	{
		Q0M10 = q0m10; Step10 = step10; Q0M8 = q0m8; Step8 = step8; Q0M3 = q0m3; Step3 = step3; Q0M5 = q0m5; Step5 = step5;
	}
}


public sealed class KernelContainer : IDisposable
{
	// Serializes first-time initialization of kernels/buffers per accelerator.
	public Action<AcceleratorStream, Index1D, ulong, ulong, ArrayView<GpuUInt128>, ArrayView<ulong>>? Order;
	public Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong,
	ulong, ulong, ulong, ulong, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>>? Incremental;
	public Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong,
		ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>,
		ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>,
		ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>? Pow2Mod;
	public Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong,
			ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>>? IncrementalOrder;
	public Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong,
		ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>>? Pow2ModOrder;
	public Action<AcceleratorStream, Index1D, ulong, uint, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>? SmallPrimeFactor;
	public Action<AcceleratorStream, Index1D, ulong, ArrayView1D<ulong, Stride1D.Dense>, int, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>? SpecialMax;

	// Optional device buffer with small divisor cycles (<= 4M). Index = divisor, value = cycle length.
	public MemoryBuffer1D<ulong, Stride1D.Dense>? SmallCycles;

	public MemoryBuffer1D<uint, Stride1D.Dense>? SmallPrimesLastOne;
	public MemoryBuffer1D<uint, Stride1D.Dense>? SmallPrimesLastSeven;
	public MemoryBuffer1D<ulong, Stride1D.Dense>? SmallPrimesPow2LastOne;
	public MemoryBuffer1D<ulong, Stride1D.Dense>? SmallPrimesPow2LastSeven;

	public MemoryBuffer1D<uint, Stride1D.Dense>? SmallPrimeFactorsPrimes;
	public MemoryBuffer1D<ulong, Stride1D.Dense>? SmallPrimeFactorsSquares;

	public void Dispose()
	{
		// Order = null;
		// Incremental = null;
		// Pow2Mod = null;
		// IncrementalOrder = null;
		// Pow2ModOrder = null;
		// SmallPrimeFactor = null;
		// SpecialMax = null;

		// SmallCycles?.Dispose();
		// SmallPrimesLastOne?.Dispose();
		// SmallPrimesLastSeven?.Dispose();
		// SmallPrimesPow2LastOne?.Dispose();
		// SmallPrimesPow2LastSeven?.Dispose();
		// SmallPrimeFactorsPrimes?.Dispose();
		// SmallPrimeFactorsSquares?.Dispose();
	}
}

public readonly struct ResiduePrimeViews(
	ArrayView1D<uint, Stride1D.Dense> lastOne,
	ArrayView1D<uint, Stride1D.Dense> lastSeven,
	ArrayView1D<ulong, Stride1D.Dense> lastOnePow2,
	ArrayView1D<ulong, Stride1D.Dense> lastSevenPow2)
{
	public readonly ArrayView1D<uint, Stride1D.Dense> LastOne = lastOne;
	public readonly ArrayView1D<uint, Stride1D.Dense> LastSeven = lastSeven;
	public readonly ArrayView1D<ulong, Stride1D.Dense> LastOnePow2 = lastOnePow2;
	public readonly ArrayView1D<ulong, Stride1D.Dense> LastSevenPow2 = lastSevenPow2;
}

public readonly struct SmallPrimeFactorTables(
	MemoryBuffer1D<uint, Stride1D.Dense> primes,
	MemoryBuffer1D<ulong, Stride1D.Dense> squares,
	int count)
{
	public readonly MemoryBuffer1D<uint, Stride1D.Dense> Primes = primes;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> Squares = squares;
	public readonly int Count = count;

	public ArrayView1D<uint, Stride1D.Dense> PrimesView => Primes.View;
	public ArrayView1D<ulong, Stride1D.Dense> SquaresView => Squares.View;
}

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

	internal static KernelContainer GetKernels(Accelerator accelerator)
	{
		return _kernels.GetOrAdd(accelerator, static (accelerator) =>
		{
			var kernels = new KernelContainer();
			PreloadStaticTables(kernels, accelerator);
			return kernels;
		});
	}

	internal static void PreloadStaticTables(KernelContainer kernels, Accelerator accelerator)
	{
		EnsureSmallCyclesOnDevice(kernels, accelerator);
		EnsureSmallPrimesOnDevice(kernels, accelerator);
		EnsureSmallPrimeFactorTables(kernels, accelerator);
	}

	// Ensures the small cycles table is uploaded to the device for the given accelerator.
	// Returns the ArrayView to pass into kernels (when kernels are extended to accept it).
	public static ArrayView1D<ulong, Stride1D.Dense> EnsureSmallCyclesOnDevice(KernelContainer kernels, Accelerator accelerator)
	{
		if (kernels.SmallCycles is { } buffer)
		{
			return buffer.View;
		}

		var host = MersenneDivisorCycles.Shared.ExportSmallCyclesSnapshot(); // TODO: Preload this device buffer during startup and keep it immutable so we can delete the lock above in favor of the preloaded snapshot.
		var device = accelerator.Allocate1D<ulong>(host.Length);
		device.View.CopyFromCPU(host);
		kernels.SmallCycles = device;
		return device.View;
	}

	public static ResiduePrimeViews EnsureSmallPrimesOnDevice(KernelContainer kernels, Accelerator accelerator)
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
		deviceLastOne.View.CopyFromCPU(hostLastOne);
		var deviceLastSeven = accelerator.Allocate1D<uint>(hostLastSeven.Length);
		deviceLastSeven.View.CopyFromCPU(hostLastSeven);
		var deviceLastOnePow2 = accelerator.Allocate1D<ulong>(hostLastOnePow2.Length);
		deviceLastOnePow2.View.CopyFromCPU(hostLastOnePow2);
		var deviceLastSevenPow2 = accelerator.Allocate1D<ulong>(hostLastSevenPow2.Length);
		deviceLastSevenPow2.View.CopyFromCPU(hostLastSevenPow2);

		kernels.SmallPrimesLastOne = deviceLastOne;
		kernels.SmallPrimesLastSeven = deviceLastSeven;
		kernels.SmallPrimesPow2LastOne = deviceLastOnePow2;
		kernels.SmallPrimesPow2LastSeven = deviceLastSevenPow2;

		return new ResiduePrimeViews(deviceLastOne.View, deviceLastSeven.View, deviceLastOnePow2.View, deviceLastSevenPow2.View);
	}

	public static SmallPrimeFactorTables EnsureSmallPrimeFactorTables(KernelContainer kernels, Accelerator accelerator)
	{
		if (kernels.SmallPrimeFactorsPrimes is { } primeBuffer && kernels.SmallPrimeFactorsSquares is { } squareBuffer)
		{
			return new SmallPrimeFactorTables(primeBuffer, squareBuffer, (int)primeBuffer.Length);
		}

		// if (kernels.SmallPrimeFactorsPrimes is { } existingPrimes && kernels.SmallPrimeFactorsSquares is { } existingSquares)
		// {
		// 	return new SmallPrimeFactorTables(existingPrimes, existingSquares, (int)existingPrimes.Length);
		// }

		var hostPrimes = PrimesGenerator.SmallPrimes;
		var hostSquares = PrimesGenerator.SmallPrimesPow2;

		var devicePrimes = accelerator.Allocate1D<uint>(hostPrimes.Length);
		devicePrimes.View.CopyFromCPU(hostPrimes);
		var deviceSquares = accelerator.Allocate1D<ulong>(hostSquares.Length);
		deviceSquares.View.CopyFromCPU(hostSquares);

		kernels.SmallPrimeFactorsPrimes = devicePrimes;
		kernels.SmallPrimeFactorsSquares = deviceSquares;

		return new SmallPrimeFactorTables(devicePrimes, deviceSquares, hostPrimes.Length);
	}





	public static GpuKernelLease GetKernel()
	{
		GpuPrimeWorkLimiter.Acquire();
		var gpu = Rent();
		var kernels = GetKernels(gpu.Accelerator);
		return GpuKernelLease.Rent(gpu, kernels);
	}

	/// <summary>
	/// Runs a GPU action with an acquired accelerator and stream.
	/// </summary>
	/// <param name="action">Action to run with (Accelerator, Stream).</param>
	public static void Run(Action<Accelerator, AcceleratorStream> action)
	{
		var lease = GetKernel();
		var accelerator = lease.Accelerator;
		var stream = lease.Stream;
		action(accelerator, stream);
		lease.Dispose();
	}
}
