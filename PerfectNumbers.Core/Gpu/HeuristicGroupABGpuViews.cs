using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public readonly struct HeuristicGroupABGpuViews(
		ArrayView1D<ulong, Stride1D.Dense> groupADivisors,
		ArrayView1D<ulong, Stride1D.Dense> groupADivisorSquares,
		ArrayView1D<ulong, Stride1D.Dense> groupBDivisorsEnding1,
		ArrayView1D<ulong, Stride1D.Dense> groupBDivisorSquaresEnding1,
		ArrayView1D<ulong, Stride1D.Dense> groupBDivisorsEnding7,
		ArrayView1D<ulong, Stride1D.Dense> groupBDivisorSquaresEnding7,
		ArrayView1D<ulong, Stride1D.Dense> groupBDivisorsEnding9,
		ArrayView1D<ulong, Stride1D.Dense> groupBDivisorSquaresEnding9)
{
	public readonly ArrayView1D<ulong, Stride1D.Dense> GroupADivisors = groupADivisors;
	public readonly ArrayView1D<ulong, Stride1D.Dense> GroupADivisorSquares = groupADivisorSquares;
	public readonly ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorsEnding1 = groupBDivisorsEnding1;
	public readonly ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorSquaresEnding1 = groupBDivisorSquaresEnding1;
	public readonly ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorsEnding7 = groupBDivisorsEnding7;
	public readonly ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorSquaresEnding7 = groupBDivisorSquaresEnding7;
	public readonly ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorsEnding9 = groupBDivisorsEnding9;
	public readonly ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorSquaresEnding9 = groupBDivisorSquaresEnding9;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public ArrayView1D<ulong, Stride1D.Dense> SelectGroupB(byte ending) => ending switch
	{
		1 => GroupBDivisorsEnding1,
		7 => GroupBDivisorsEnding7,
		9 => GroupBDivisorsEnding9,
		_ => ArrayView1D<ulong, Stride1D.Dense>.Empty,
	};

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public ArrayView1D<ulong, Stride1D.Dense> SelectGroupBSquares(byte ending) => ending switch
	{
		1 => GroupBDivisorSquaresEnding1,
		7 => GroupBDivisorSquaresEnding7,
		9 => GroupBDivisorSquaresEnding9,
		_ => ArrayView1D<ulong, Stride1D.Dense>.Empty,
	};

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public ArrayView1D<ulong, Stride1D.Dense> SelectDivisors(HeuristicGpuDivisorTableKind kind) => kind switch
	{
		HeuristicGpuDivisorTableKind.GroupA => GroupADivisors,
		HeuristicGpuDivisorTableKind.GroupBEnding1 => GroupBDivisorsEnding1,
		HeuristicGpuDivisorTableKind.GroupBEnding7 => GroupBDivisorsEnding7,
		HeuristicGpuDivisorTableKind.GroupBEnding9 => GroupBDivisorsEnding9,
		_ => ArrayView1D<ulong, Stride1D.Dense>.Empty,
	};

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public ArrayView1D<ulong, Stride1D.Dense> SelectDivisorSquares(HeuristicGpuDivisorTableKind kind) => kind switch
	{
		HeuristicGpuDivisorTableKind.GroupA => GroupADivisorSquares,
		HeuristicGpuDivisorTableKind.GroupBEnding1 => GroupBDivisorSquaresEnding1,
		HeuristicGpuDivisorTableKind.GroupBEnding7 => GroupBDivisorSquaresEnding7,
		HeuristicGpuDivisorTableKind.GroupBEnding9 => GroupBDivisorSquaresEnding9,
		_ => ArrayView1D<ulong, Stride1D.Dense>.Empty,
	};
}
