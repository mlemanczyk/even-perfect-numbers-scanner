using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu.Accelerators;

internal readonly struct HeuristicCombinedGpuTables
{
	private static readonly Accelerator[] _accelerators = AcceleratorPool.Shared.Accelerators;

	private static readonly HeuristicCombinedGpuTables[] _sharedTables = new HeuristicCombinedGpuTables[PerfectNumberConstants.RollingAccelerators];

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	internal static void WarmUp(int acceleratorIndex, AcceleratorStream stream) => _sharedTables[acceleratorIndex] = new(_accelerators[acceleratorIndex], stream);

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	internal static HeuristicCombinedGpuTables GetStaticTables(int acceleratorIndex) => _sharedTables[acceleratorIndex];

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	internal HeuristicCombinedGpuTables(Accelerator accelerator, AcceleratorStream stream)
	{
		var combinedEnding1 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding1;
		HeuristicCombinedDivisorsEnding1 = accelerator.CopySpanToDevice(stream, combinedEnding1);
		var combinedSquaresEnding1 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding1Squares;
		HeuristicCombinedDivisorSquaresEnding1 = accelerator.CopySpanToDevice(stream, combinedSquaresEnding1);

		var combinedEnding3 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding3;
		HeuristicCombinedDivisorsEnding3 = accelerator.CopySpanToDevice(stream, combinedEnding3);
		var combinedSquaresEnding3 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding3Squares;
		HeuristicCombinedDivisorSquaresEnding3 = accelerator.CopySpanToDevice(stream, combinedSquaresEnding3);

		var combinedEnding7 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding7;
		HeuristicCombinedDivisorsEnding7 = accelerator.CopySpanToDevice(stream, combinedEnding7);
		var combinedSquaresEnding7 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding7Squares;
		HeuristicCombinedDivisorSquaresEnding7 = accelerator.CopySpanToDevice(stream, combinedSquaresEnding7);

		var combinedEnding9 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding9;
		HeuristicCombinedDivisorsEnding9 = accelerator.CopySpanToDevice(stream, combinedEnding9);
		var combinedSquaresEnding9 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding9Squares;
		HeuristicCombinedDivisorSquaresEnding9 = accelerator.CopySpanToDevice(stream, combinedSquaresEnding9);
	}

	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorsEnding1;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorSquaresEnding1;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorsEnding3;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorSquaresEnding3;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorsEnding7;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorSquaresEnding7;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorsEnding9;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorSquaresEnding9;

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	internal HeuristicCombinedGpuViews CreateViews() => new(
			HeuristicCombinedDivisorsEnding1.View,
			HeuristicCombinedDivisorsEnding3.View,
			HeuristicCombinedDivisorsEnding7.View,
			HeuristicCombinedDivisorsEnding9.View,
			HeuristicCombinedDivisorSquaresEnding1.View,
			HeuristicCombinedDivisorSquaresEnding3.View,
			HeuristicCombinedDivisorSquaresEnding7.View,
			HeuristicCombinedDivisorSquaresEnding9.View);

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	internal void Dispose()
	{
		HeuristicCombinedDivisorsEnding1.Dispose();
		HeuristicCombinedDivisorSquaresEnding1.Dispose();
		HeuristicCombinedDivisorsEnding3.Dispose();
		HeuristicCombinedDivisorSquaresEnding3.Dispose();
		HeuristicCombinedDivisorsEnding7.Dispose();
		HeuristicCombinedDivisorSquaresEnding7.Dispose();
		HeuristicCombinedDivisorsEnding9.Dispose();
		HeuristicCombinedDivisorSquaresEnding9.Dispose();
	}
}
