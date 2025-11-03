using PerfectNumbers.Core;
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

public sealed class KernelContainer
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
	public Action<AcceleratorStream, Index1D, ulong, ArrayView1D<ulong, Stride1D.Dense>, int, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ushort, Stride1D.Dense>>? SpecialMax;

	// Optional device buffer with small divisor cycles (<= 4M). Index = divisor, value = cycle length.
	public MemoryBuffer1D<ulong, Stride1D.Dense>? SmallCycles;
	public MemoryBuffer1D<uint, Stride1D.Dense>? SmallPrimeFactorsPrimes;
	public MemoryBuffer1D<ulong, Stride1D.Dense>? SmallPrimeFactorsSquares;
	public MemoryBuffer1D<ulong, Stride1D.Dense>? SmallPrimeFactorPrimeSlots;
	public MemoryBuffer1D<int, Stride1D.Dense>? SmallPrimeFactorExponentSlots;
	public MemoryBuffer1D<int, Stride1D.Dense>? SmallPrimeFactorCountSlot;
	public MemoryBuffer1D<ulong, Stride1D.Dense>? SmallPrimeFactorRemainingSlot;
	public MemoryBuffer1D<ulong, Stride1D.Dense>? SpecialMaxFactors;
	public MemoryBuffer1D<ulong, Stride1D.Dense>? SpecialMaxCandidates;
	public MemoryBuffer1D<ushort, Stride1D.Dense>? SpecialMaxResult;
	public MemoryBuffer1D<uint, Stride1D.Dense>? SmallPrimesLastOne;
	public MemoryBuffer1D<ulong, Stride1D.Dense>? SmallPrimesPow2LastOne;
	public MemoryBuffer1D<uint, Stride1D.Dense>? SmallPrimesLastSeven;
	public MemoryBuffer1D<ulong, Stride1D.Dense>? SmallPrimesPow2LastSeven;

	public static T InitOnce<T>(ref T? slot, Func<T> factory) where T : class
	{
		var current = Volatile.Read(ref slot);
		if (current is not null)
		{
			return current;
		}

		current = Volatile.Read(ref slot);
		if (current is null)
		{
			current = factory();
			Volatile.Write(ref slot, current);
		}
		return current;
	}

	public void Dispose() => GpuKernelPool.Return(this);
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

public readonly struct SmallPrimeFactorScratch(
		MemoryBuffer1D<ulong, Stride1D.Dense> primeSlots,
		MemoryBuffer1D<int, Stride1D.Dense> exponentSlots,
		MemoryBuffer1D<int, Stride1D.Dense> countSlot,
		MemoryBuffer1D<ulong, Stride1D.Dense> remainingSlot)
{
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> PrimeSlots = primeSlots;
	public readonly MemoryBuffer1D<int, Stride1D.Dense> ExponentSlots = exponentSlots;
	public readonly MemoryBuffer1D<int, Stride1D.Dense> CountSlot = countSlot;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> RemainingSlot = remainingSlot;

	public void Clear()
	{
		PrimeSlots.MemSetToZero();
		ExponentSlots.MemSetToZero();
		CountSlot.MemSetToZero();
		RemainingSlot.MemSetToZero();
	}
}

public readonly struct SpecialMaxScratch(
	MemoryBuffer1D<ulong, Stride1D.Dense> factorValues,
	MemoryBuffer1D<ulong, Stride1D.Dense> candidateValues,
	MemoryBuffer1D<ushort, Stride1D.Dense> resultSlot)
{
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> FactorValues = factorValues;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> CandidateValues = candidateValues;
	public readonly MemoryBuffer1D<ushort, Stride1D.Dense> ResultSlot = resultSlot;

	public ArrayView1D<ulong, Stride1D.Dense> FactorsView => FactorValues.View;
	public ArrayView1D<ulong, Stride1D.Dense> CandidatesView => CandidateValues.View;
	public ArrayView1D<ushort, Stride1D.Dense> ResultView => ResultSlot.View;
}

public class GpuKernelPool
{
	private static readonly ConcurrentQueue<KernelContainer> KernelCache = new(); // TODO: Replace this concurrent map with a simple accelerator-indexed lookup once kernel launchers are prewarmed during startup so we can drop the thread-safe wrapper entirely.

	public static KernelContainer GetKernels() => KernelCache.TryDequeue(out var container)
		? container
		: new KernelContainer();

	public static void Return(KernelContainer container) => KernelCache.Enqueue(container);

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

		if (kernels.SmallPrimeFactorsPrimes is { } existingPrimes && kernels.SmallPrimeFactorsSquares is { } existingSquares)
		{
			return new SmallPrimeFactorTables(existingPrimes, existingSquares, (int)existingPrimes.Length);
		}

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

	public static SmallPrimeFactorScratch EnsureSmallPrimeFactorScratch(KernelContainer kernels, Accelerator accelerator, int slotCount)
	{
		MemoryBuffer1D<ulong, Stride1D.Dense>? primeSlots = kernels.SmallPrimeFactorPrimeSlots;
		if (primeSlots is null || primeSlots.Length < slotCount)
		{
			primeSlots?.Dispose();
			primeSlots = accelerator.Allocate1D<ulong>(slotCount);
			kernels.SmallPrimeFactorPrimeSlots = primeSlots;
		}

		MemoryBuffer1D<int, Stride1D.Dense>? exponentSlots = kernels.SmallPrimeFactorExponentSlots;
		if (exponentSlots is null || exponentSlots.Length < slotCount)
		{
			exponentSlots?.Dispose();
			exponentSlots = accelerator.Allocate1D<int>(slotCount);
			kernels.SmallPrimeFactorExponentSlots = exponentSlots;
		}

		MemoryBuffer1D<int, Stride1D.Dense>? countSlot = kernels.SmallPrimeFactorCountSlot;
		if (countSlot is null)
		{
			countSlot = accelerator.Allocate1D<int>(1);
			kernels.SmallPrimeFactorCountSlot = countSlot;
		}

		MemoryBuffer1D<ulong, Stride1D.Dense>? remainingSlot = kernels.SmallPrimeFactorRemainingSlot;
		if (remainingSlot is null)
		{
			remainingSlot = accelerator.Allocate1D<ulong>(1);
			kernels.SmallPrimeFactorRemainingSlot = remainingSlot;
		}

		return new SmallPrimeFactorScratch(primeSlots!, exponentSlots!, countSlot!, remainingSlot!);
	}

	public static SpecialMaxScratch EnsureSpecialMaxScratch(KernelContainer kernels, Accelerator accelerator, int factorCapacity)
	{
		MemoryBuffer1D<ulong, Stride1D.Dense>? factorValues = kernels.SpecialMaxFactors;
		if (factorValues is null || factorValues.Length < factorCapacity)
		{
			factorValues?.Dispose();
			factorValues = accelerator.Allocate1D<ulong>(factorCapacity);
			kernels.SpecialMaxFactors = factorValues;
		}

		MemoryBuffer1D<ulong, Stride1D.Dense>? candidateValues = kernels.SpecialMaxCandidates;
		if (candidateValues is null || candidateValues.Length < factorCapacity)
		{
			candidateValues?.Dispose();
			candidateValues = accelerator.Allocate1D<ulong>(factorCapacity);
			kernels.SpecialMaxCandidates = candidateValues;
		}

		MemoryBuffer1D<ushort, Stride1D.Dense>? resultSlot = kernels.SpecialMaxResult;
		if (resultSlot is null)
		{
			resultSlot = accelerator.Allocate1D<ushort>(1);
			kernels.SpecialMaxResult = resultSlot;
		}

		return new SpecialMaxScratch(factorValues!, candidateValues!, resultSlot!);
	}

	public static GpuKernelLease GetKernel(bool useGpuOrder)
	{
		_ = useGpuOrder;

		GpuPrimeWorkLimiter.Acquire();
		var gpu = Rent();
		return GpuKernelLease.Rent(gpu);
	}

	/// <summary>
	/// Runs a GPU action with an acquired accelerator and stream.
	/// </summary>
	/// <param name="action">Action to run with (Accelerator, Stream).</param>
	public static void Run(Action<Accelerator, AcceleratorStream> action)
	{
		var lease = GetKernel(useGpuOrder: true);
		var accelerator = lease.Accelerator;
		var stream = accelerator.CreateStream();
		action(accelerator, stream);
		stream.Dispose();
		lease.Dispose();

	}
}
