using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public readonly struct OrderKernelBuffers(
	ArrayView1D<ulong, Stride1D.Dense> phiFactors,
	ArrayView1D<int, Stride1D.Dense> phiExponents,
	ArrayView1D<ulong, Stride1D.Dense> workFactors,
	ArrayView1D<int, Stride1D.Dense> workExponents,
	ArrayView1D<ulong, Stride1D.Dense> candidates,
	ArrayView1D<int, Stride1D.Dense> stackIndex,
	ArrayView1D<int, Stride1D.Dense> stackExponent,
	ArrayView1D<ulong, Stride1D.Dense> stackProduct,
	ArrayView1D<ulong, Stride1D.Dense> result,
	ArrayView1D<byte, Stride1D.Dense> status)
{
	public readonly ArrayView1D<ulong, Stride1D.Dense> PhiFactors = phiFactors;
	public readonly ArrayView1D<int, Stride1D.Dense> PhiExponents = phiExponents;
	public readonly ArrayView1D<ulong, Stride1D.Dense> WorkFactors = workFactors;
	public readonly ArrayView1D<int, Stride1D.Dense> WorkExponents = workExponents;
	public readonly ArrayView1D<ulong, Stride1D.Dense> Candidates = candidates;
	public readonly ArrayView1D<int, Stride1D.Dense> StackIndex = stackIndex;
	public readonly ArrayView1D<int, Stride1D.Dense> StackExponent = stackExponent;
	public readonly ArrayView1D<ulong, Stride1D.Dense> StackProduct = stackProduct;
	public readonly ArrayView1D<ulong, Stride1D.Dense> Result = result;
	public readonly ArrayView1D<byte, Stride1D.Dense> Status = status;
}
