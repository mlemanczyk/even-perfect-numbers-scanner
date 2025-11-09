using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

internal readonly struct HeuristicGpuDivisorTables
{
	public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorsEnding1;
	public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorsEnding3;
	public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorsEnding7;
	public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorsEnding9;
	public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorSquaresEnding1;
	public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorSquaresEnding3;
	public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorSquaresEnding7;
	public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorSquaresEnding9;
	public readonly ArrayView1D<ulong, Stride1D.Dense> GroupADivisors;
	public readonly ArrayView1D<ulong, Stride1D.Dense> GroupADivisorSquares;
	public readonly ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorsEnding1;
	public readonly ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorSquaresEnding1;
	public readonly ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorsEnding7;
	public readonly ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorSquaresEnding7;
	public readonly ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorsEnding9;
	public readonly ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorSquaresEnding9;

	public HeuristicGpuDivisorTables(
			ArrayView1D<ulong, Stride1D.Dense> combinedDivisorsEnding1,
			ArrayView1D<ulong, Stride1D.Dense> combinedDivisorsEnding3,
			ArrayView1D<ulong, Stride1D.Dense> combinedDivisorsEnding7,
			ArrayView1D<ulong, Stride1D.Dense> combinedDivisorsEnding9,
			ArrayView1D<ulong, Stride1D.Dense> combinedDivisorSquaresEnding1,
			ArrayView1D<ulong, Stride1D.Dense> combinedDivisorSquaresEnding3,
			ArrayView1D<ulong, Stride1D.Dense> combinedDivisorSquaresEnding7,
			ArrayView1D<ulong, Stride1D.Dense> combinedDivisorSquaresEnding9,
			ArrayView1D<ulong, Stride1D.Dense> groupADivisors,
			ArrayView1D<ulong, Stride1D.Dense> groupADivisorSquares,
			ArrayView1D<ulong, Stride1D.Dense> groupBDivisorsEnding1,
			ArrayView1D<ulong, Stride1D.Dense> groupBDivisorSquaresEnding1,
			ArrayView1D<ulong, Stride1D.Dense> groupBDivisorsEnding7,
			ArrayView1D<ulong, Stride1D.Dense> groupBDivisorSquaresEnding7,
			ArrayView1D<ulong, Stride1D.Dense> groupBDivisorsEnding9,
			ArrayView1D<ulong, Stride1D.Dense> groupBDivisorSquaresEnding9)
	{
		CombinedDivisorsEnding1 = combinedDivisorsEnding1;
		CombinedDivisorsEnding3 = combinedDivisorsEnding3;
		CombinedDivisorsEnding7 = combinedDivisorsEnding7;
		CombinedDivisorsEnding9 = combinedDivisorsEnding9;
		CombinedDivisorSquaresEnding1 = combinedDivisorSquaresEnding1;
		CombinedDivisorSquaresEnding3 = combinedDivisorSquaresEnding3;
		CombinedDivisorSquaresEnding7 = combinedDivisorSquaresEnding7;
		CombinedDivisorSquaresEnding9 = combinedDivisorSquaresEnding9;
		GroupADivisors = groupADivisors;
		GroupADivisorSquares = groupADivisorSquares;
		GroupBDivisorsEnding1 = groupBDivisorsEnding1;
		GroupBDivisorSquaresEnding1 = groupBDivisorSquaresEnding1;
		GroupBDivisorsEnding7 = groupBDivisorsEnding7;
		GroupBDivisorSquaresEnding7 = groupBDivisorSquaresEnding7;
		GroupBDivisorsEnding9 = groupBDivisorsEnding9;
		GroupBDivisorSquaresEnding9 = groupBDivisorSquaresEnding9;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public ArrayView1D<ulong, Stride1D.Dense> SelectCombined(byte ending)
	{
		return ending switch
		{
			1 => CombinedDivisorsEnding1,
			3 => CombinedDivisorsEnding3,
			7 => CombinedDivisorsEnding7,
			9 => CombinedDivisorsEnding9,
			_ => ArrayView1D<ulong, Stride1D.Dense>.Empty,
		};
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public ArrayView1D<ulong, Stride1D.Dense> SelectCombinedSquares(byte ending)
	{
		return ending switch
		{
			1 => CombinedDivisorSquaresEnding1,
			3 => CombinedDivisorSquaresEnding3,
			7 => CombinedDivisorSquaresEnding7,
			9 => CombinedDivisorSquaresEnding9,
			_ => ArrayView1D<ulong, Stride1D.Dense>.Empty,
		};
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public ArrayView1D<ulong, Stride1D.Dense> SelectGroupB(byte ending)
	{
		return ending switch
		{
			1 => GroupBDivisorsEnding1,
			7 => GroupBDivisorsEnding7,
			9 => GroupBDivisorsEnding9,
			_ => ArrayView1D<ulong, Stride1D.Dense>.Empty,
		};
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public ArrayView1D<ulong, Stride1D.Dense> SelectGroupBSquares(byte ending)
	{
		return ending switch
		{
			1 => GroupBDivisorSquaresEnding1,
			7 => GroupBDivisorSquaresEnding7,
			9 => GroupBDivisorSquaresEnding9,
			_ => ArrayView1D<ulong, Stride1D.Dense>.Empty,
		};
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public ArrayView1D<ulong, Stride1D.Dense> SelectDivisors(HeuristicGpuDivisorTableKind kind, byte ending)
	{
		return kind switch
		{
			HeuristicGpuDivisorTableKind.GroupA => GroupADivisors,
			HeuristicGpuDivisorTableKind.GroupBEnding1 => GroupBDivisorsEnding1,
			HeuristicGpuDivisorTableKind.GroupBEnding7 => GroupBDivisorsEnding7,
			HeuristicGpuDivisorTableKind.GroupBEnding9 => GroupBDivisorsEnding9,
			HeuristicGpuDivisorTableKind.Combined => SelectCombined(ending),
			_ => ArrayView1D<ulong, Stride1D.Dense>.Empty,
		};
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public ArrayView1D<ulong, Stride1D.Dense> SelectDivisorSquares(HeuristicGpuDivisorTableKind kind, byte ending)
	{
		return kind switch
		{
			HeuristicGpuDivisorTableKind.GroupA => GroupADivisorSquares,
			HeuristicGpuDivisorTableKind.GroupBEnding1 => GroupBDivisorSquaresEnding1,
			HeuristicGpuDivisorTableKind.GroupBEnding7 => GroupBDivisorSquaresEnding7,
			HeuristicGpuDivisorTableKind.GroupBEnding9 => GroupBDivisorSquaresEnding9,
			HeuristicGpuDivisorTableKind.Combined => SelectCombinedSquares(ending),
			_ => ArrayView1D<ulong, Stride1D.Dense>.Empty,
		};
	}
}
