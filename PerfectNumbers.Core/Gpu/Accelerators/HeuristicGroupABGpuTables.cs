using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu.Accelerators;

internal sealed class HeuristicGroupABGpuTables
{
	private static readonly Accelerator[] _accelerators = AcceleratorPool.Shared.Accelerators;

	private static readonly HeuristicGroupABGpuTables[] _sharedTables = new HeuristicGroupABGpuTables[PerfectNumberConstants.RollingAccelerators];

	private static readonly SemaphoreSlim[] _locks = [..Enumerable.Range(1, PerfectNumberConstants.RollingAccelerators).Select(_ => new SemaphoreSlim(1))];

	internal static HeuristicGroupABGpuTables EnsureStaticTables(int acceleratorIndex, AcceleratorStream stream)
	{
		var @lock = _locks[acceleratorIndex];
		@lock.Wait();
		if (_sharedTables[acceleratorIndex] is {} existing)
		{
			@lock.Release();
			return existing;
		}

		var accelerator = _accelerators[acceleratorIndex];
		existing = new(accelerator, stream);
		_sharedTables[acceleratorIndex] = existing;
		@lock.Release();
		return existing;
	}

	internal HeuristicGroupABGpuTables(Accelerator accelerator, AcceleratorStream stream)
	{
		var heuristicGroupA = HeuristicPrimeSieves.GroupADivisorsStorage;
		HeuristicGroupADivisors = accelerator.CopySpanToDevice(stream, heuristicGroupA);

		var heuristicGroupASquares = HeuristicPrimeSieves.GroupADivisorSquaresStorage;
		HeuristicGroupADivisorSquares = accelerator.CopySpanToDevice(stream, heuristicGroupASquares);

		HeuristicGroupBDivisorsEnding1 = accelerator.CopyUintSpanToDevice(stream, DivisorGenerator.SmallPrimesLastOneWithoutLastThree);
		HeuristicGroupBDivisorSquaresEnding1 = accelerator.CopySpanToDevice(stream, DivisorGenerator.SmallPrimesPow2LastOneWithoutLastThree);

		HeuristicGroupBDivisorsEnding7 = accelerator.CopyUintSpanToDevice(stream, DivisorGenerator.SmallPrimesLastSevenWithoutLastThree);
		HeuristicGroupBDivisorSquaresEnding7 = accelerator.CopySpanToDevice(stream, DivisorGenerator.SmallPrimesPow2LastSevenWithoutLastThree);

		HeuristicGroupBDivisorsEnding9 = accelerator.CopyUintSpanToDevice(stream, DivisorGenerator.SmallPrimesLastNineWithoutLastThree);
		HeuristicGroupBDivisorSquaresEnding9 = accelerator.CopySpanToDevice(stream, DivisorGenerator.SmallPrimesPow2LastNineWithoutLastThree);
	}

	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupADivisors;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupADivisorSquares;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorsEnding1;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorSquaresEnding1;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorsEnding7;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorSquaresEnding7;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorsEnding9;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorSquaresEnding9;

	internal HeuristicGroupABGpuViews CreateViews() => new(
			HeuristicGroupADivisors.View,
			HeuristicGroupADivisorSquares.View,
			HeuristicGroupBDivisorsEnding1.View,
			HeuristicGroupBDivisorSquaresEnding1.View,
			HeuristicGroupBDivisorsEnding7.View,
			HeuristicGroupBDivisorSquaresEnding7.View,
			HeuristicGroupBDivisorsEnding9.View,
			HeuristicGroupBDivisorSquaresEnding9.View);

	internal void Dispose()
	{
		HeuristicGroupADivisors.Dispose();
		HeuristicGroupADivisorSquares.Dispose();
		HeuristicGroupBDivisorsEnding1.Dispose();
		HeuristicGroupBDivisorSquaresEnding1.Dispose();
		HeuristicGroupBDivisorsEnding7.Dispose();
		HeuristicGroupBDivisorSquaresEnding7.Dispose();
		HeuristicGroupBDivisorsEnding9.Dispose();
		HeuristicGroupBDivisorSquaresEnding9.Dispose();
	}
}
