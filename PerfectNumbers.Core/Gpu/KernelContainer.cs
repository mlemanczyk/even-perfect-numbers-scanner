using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

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

	public Action<AcceleratorStream, Index1D, ulong, ArrayView1D<ulong, Stride1D.Dense>, int, ArrayView1D<int, Stride1D.Dense>,
		  #if DETAILED_LOG
			  ArrayView1D<ulong, Stride1D.Dense>,
		  #endif
		  ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? BitContradiction;

	public Kernel? SmallPrimeFactor;
	
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
