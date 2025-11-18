using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu.Accelerators;

public readonly struct HeuristicCombinedGpuViews(
		ArrayView1D<ulong, Stride1D.Dense> combinedDivisorsEnding1,
		ArrayView1D<ulong, Stride1D.Dense> combinedDivisorsEnding3,
		ArrayView1D<ulong, Stride1D.Dense> combinedDivisorsEnding7,
		ArrayView1D<ulong, Stride1D.Dense> combinedDivisorsEnding9,
		ArrayView1D<ulong, Stride1D.Dense> combinedDivisorSquaresEnding1,
		ArrayView1D<ulong, Stride1D.Dense> combinedDivisorSquaresEnding3,
		ArrayView1D<ulong, Stride1D.Dense> combinedDivisorSquaresEnding7,
		ArrayView1D<ulong, Stride1D.Dense> combinedDivisorSquaresEnding9)
{
	public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorsEnding1 = combinedDivisorsEnding1;
	public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorsEnding3 = combinedDivisorsEnding3;
	public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorsEnding7 = combinedDivisorsEnding7;
	public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorsEnding9 = combinedDivisorsEnding9;
	public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorSquaresEnding1 = combinedDivisorSquaresEnding1;
	public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorSquaresEnding3 = combinedDivisorSquaresEnding3;
	public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorSquaresEnding7 = combinedDivisorSquaresEnding7;
	public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorSquaresEnding9 = combinedDivisorSquaresEnding9;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public ArrayView1D<ulong, Stride1D.Dense> SelectDivisors(byte ending) => ending switch
	{
		1 => CombinedDivisorsEnding1,
		3 => CombinedDivisorsEnding3,
		7 => CombinedDivisorsEnding7,
		9 => CombinedDivisorsEnding9,
		_ => ArrayView1D<ulong, Stride1D.Dense>.Empty,
	};

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public ArrayView1D<ulong, Stride1D.Dense> SelectDivisorSquares(byte ending) => ending switch
	{
		1 => CombinedDivisorSquaresEnding1,
		3 => CombinedDivisorSquaresEnding3,
		7 => CombinedDivisorSquaresEnding7,
		9 => CombinedDivisorSquaresEnding9,
		_ => ArrayView1D<ulong, Stride1D.Dense>.Empty,
	};
}
