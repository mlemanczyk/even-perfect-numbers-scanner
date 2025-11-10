using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu.Accelerators;

internal sealed class LastDigitGpuTables
{
	private static readonly ConcurrentDictionary<Accelerator, LastDigitGpuTables> _sharedTables = new(20_480, PerfectNumberConstants.RollingAccelerators);

	internal static LastDigitGpuTables EnsureStaticTables(Accelerator accelerator, AcceleratorStream stream)
		=> _sharedTables.GetOrAdd(accelerator, _ => new LastDigitGpuTables(accelerator, stream));

	internal LastDigitGpuTables(Accelerator accelerator, AcceleratorStream stream)
	{
		var primesLastOne = DivisorGenerator.SmallPrimesLastOne;
		var primesLastSeven = DivisorGenerator.SmallPrimesLastSeven;
		var primesLastThree = DivisorGenerator.SmallPrimesLastThree;
		var primesLastNine = DivisorGenerator.SmallPrimesLastNine;
		var primesPow2LastOne = DivisorGenerator.SmallPrimesPow2LastOne;
		var primesPow2LastSeven = DivisorGenerator.SmallPrimesPow2LastSeven;
		var primesPow2LastThree = DivisorGenerator.SmallPrimesPow2LastThree;
		var primesPow2LastNine = DivisorGenerator.SmallPrimesPow2LastNine;

		// lock (accelerator)
		{
			DevicePrimesLastOne = accelerator.Allocate1D(stream, primesLastOne);
			DevicePrimesLastSeven = accelerator.Allocate1D(stream, primesLastSeven);
			DevicePrimesLastThree = accelerator.Allocate1D(stream, primesLastThree);
			DevicePrimesLastNine = accelerator.Allocate1D(stream, primesLastNine);
			DevicePrimesPow2LastOne = accelerator.Allocate1D(stream, primesPow2LastOne);
			DevicePrimesPow2LastSeven = accelerator.Allocate1D(stream, primesPow2LastSeven);
			DevicePrimesPow2LastThree = accelerator.Allocate1D(stream, primesPow2LastThree);
			DevicePrimesPow2LastNine = accelerator.Allocate1D(stream, primesPow2LastNine);
		}
	}

	internal readonly MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastOne;
	internal readonly MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastSeven;
	internal readonly MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastThree;
	internal readonly MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastNine;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastOne;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastSeven;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastThree;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastNine;

	internal void Dispose()
	{
		DevicePrimesLastOne.Dispose();
		DevicePrimesLastSeven.Dispose();
		DevicePrimesLastThree.Dispose();
		DevicePrimesLastNine.Dispose();
		DevicePrimesPow2LastOne.Dispose();
		DevicePrimesPow2LastSeven.Dispose();
		DevicePrimesPow2LastThree.Dispose();
		DevicePrimesPow2LastNine.Dispose();
	}
}
