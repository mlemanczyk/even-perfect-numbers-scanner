using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;
public readonly struct SmallPrimeFactorGpuTables

{
	private static readonly Accelerator[] _accelerators = AcceleratorPool.Shared.Accelerators;
	private static readonly SmallPrimeFactorGpuTables?[] _sharedTables = new SmallPrimeFactorGpuTables?[PerfectNumberConstants.RollingAccelerators];

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static SmallPrimeFactorGpuTables? GetStaticTables(int acceleratorIndex) => _sharedTables[acceleratorIndex];

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static void WarmUp(int acceleratorIndex, AcceleratorStream stream) => _sharedTables[acceleratorIndex] = new SmallPrimeFactorGpuTables(_accelerators[acceleratorIndex], stream);

	public readonly MemoryBuffer1D<uint, Stride1D.Dense> Primes;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> Squares;
	public readonly int Count;

	public SmallPrimeFactorGpuTables(Accelerator accelerator, AcceleratorStream stream)
	{
		var hostPrimes = PrimesGenerator.SmallPrimes;
		var hostSquares = PrimesGenerator.SmallPrimesPow2;

		MemoryBuffer1D<uint, Stride1D.Dense>? devicePrimes = accelerator.Allocate1D(stream, hostPrimes);
		MemoryBuffer1D<ulong, Stride1D.Dense>? deviceSquares = accelerator.Allocate1D(stream, hostSquares);

		Primes = devicePrimes;
		Squares = deviceSquares;
		Count = hostPrimes.Length;
	}
}
