using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

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
