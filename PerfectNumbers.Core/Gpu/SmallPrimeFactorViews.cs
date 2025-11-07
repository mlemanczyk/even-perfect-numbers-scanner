using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public readonly struct SmallPrimeFactorViews(
	MemoryBuffer1D<uint, Stride1D.Dense> primes,
	MemoryBuffer1D<ulong, Stride1D.Dense> squares,
	int count)
{
	public readonly MemoryBuffer1D<uint, Stride1D.Dense> Primes = primes;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> Squares = squares;
	public readonly int Count = count;

	public ArrayView1D<uint, Stride1D.Dense> PrimesView
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		get
		{
			return Primes.View;
		}
	}

	public ArrayView1D<ulong, Stride1D.Dense> SquaresView
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		get
		{
			return Squares.View;
		}
	}
}
