using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public sealed class GpuKernelLease
{
	private static readonly ConcurrentQueue<GpuKernelLease> Pool = new();

	internal static GpuKernelLease Rent()
	{
		if (!Pool.TryDequeue(out var lease))
		{
			lease = new GpuKernelLease();
		}

		return lease;
	}

	public static readonly Action<AcceleratorStream, Index1D, ulong, ulong, ArrayView<GpuUInt128>, ArrayView<ulong>> OrderKernel = GpuKernelPool.Kernels.Order!;

	public static readonly Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>> Pow2ModKernel = GpuKernelPool.Kernels.Pow2Mod!;

	public static readonly Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ulong, ulong, ulong, ulong, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>> IncrementalKernel = GpuKernelPool.Kernels.Incremental!;

	public static readonly Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>> IncrementalOrderKernel = GpuKernelPool.Kernels.IncrementalOrder!;

	public static readonly Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<ulong, Stride1D.Dense>> Pow2ModOrderKernel = GpuKernelPool.Kernels.Pow2ModOrder!;

	public static readonly Action<AcceleratorStream, Index1D, ulong, ArrayView1D<ulong, Stride1D.Dense>, int, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>> SpecialMaxKernel = GpuKernelPool.Kernels.SpecialMax!;

	public static readonly Action<AcceleratorStream, Index1D, ulong, uint, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>> SmallPrimeFactorKernel = GpuKernelPool.Kernels.SmallPrimeFactor!;

	public void Dispose()
	{
		Pool.Enqueue(this);
		GpuPrimeWorkLimiter.Release();
	}
}
