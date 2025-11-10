using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu.Accelerators;

internal sealed class HeuristicGpuTables
{
	private static readonly ConcurrentDictionary<Accelerator, HeuristicGpuTables> _sharedTables = new(20_480, PerfectNumberConstants.RollingAccelerators);

	internal static HeuristicGpuTables EnsureStaticTables(Accelerator accelerator, AcceleratorStream stream)
		=> _sharedTables.GetOrAdd(accelerator, _ => new HeuristicGpuTables(accelerator, stream));

	internal HeuristicGpuTables(Accelerator accelerator, AcceleratorStream stream)
	{
		// lock (accelerator)
		{
			var heuristicGroupA = HeuristicPrimeSieves.GroupADivisorsStorage;
			HeuristicGroupADivisors = CopySpanToDevice(accelerator, stream, heuristicGroupA);

			var heuristicGroupASquares = HeuristicPrimeSieves.GroupADivisorSquaresStorage;
			HeuristicGroupADivisorSquares = CopySpanToDevice(accelerator, stream, heuristicGroupASquares);

			HeuristicGroupBDivisorsEnding1 = CopyUintSpanToDevice(accelerator, stream, DivisorGenerator.SmallPrimesLastOneWithoutLastThree);
			HeuristicGroupBDivisorSquaresEnding1 = CopySpanToDevice(accelerator, stream, DivisorGenerator.SmallPrimesPow2LastOneWithoutLastThree);

			HeuristicGroupBDivisorsEnding7 = CopyUintSpanToDevice(accelerator, stream, DivisorGenerator.SmallPrimesLastSevenWithoutLastThree);
			HeuristicGroupBDivisorSquaresEnding7 = CopySpanToDevice(accelerator, stream, DivisorGenerator.SmallPrimesPow2LastSevenWithoutLastThree);

			HeuristicGroupBDivisorsEnding9 = CopyUintSpanToDevice(accelerator, stream, DivisorGenerator.SmallPrimesLastNineWithoutLastThree);
			HeuristicGroupBDivisorSquaresEnding9 = CopySpanToDevice(accelerator, stream, DivisorGenerator.SmallPrimesPow2LastNineWithoutLastThree);

			var combinedEnding1 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding1;
			HeuristicCombinedDivisorsEnding1 = CopySpanToDevice(accelerator, stream, combinedEnding1);
			var combinedSquaresEnding1 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding1Squares;
			HeuristicCombinedDivisorSquaresEnding1 = CopySpanToDevice(accelerator, stream, combinedSquaresEnding1);

			var combinedEnding3 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding3;
			HeuristicCombinedDivisorsEnding3 = CopySpanToDevice(accelerator, stream, combinedEnding3);
			var combinedSquaresEnding3 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding3Squares;
			HeuristicCombinedDivisorSquaresEnding3 = CopySpanToDevice(accelerator, stream, combinedSquaresEnding3);

			var combinedEnding7 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding7;
			HeuristicCombinedDivisorsEnding7 = CopySpanToDevice(accelerator, stream, combinedEnding7);
			var combinedSquaresEnding7 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding7Squares;
			HeuristicCombinedDivisorSquaresEnding7 = CopySpanToDevice(accelerator, stream, combinedSquaresEnding7);

			var combinedEnding9 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding9;
			HeuristicCombinedDivisorsEnding9 = CopySpanToDevice(accelerator, stream, combinedEnding9);
			var combinedSquaresEnding9 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding9Squares;
			HeuristicCombinedDivisorSquaresEnding9 = CopySpanToDevice(accelerator, stream, combinedSquaresEnding9);
		}
	}

	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupADivisors;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupADivisorSquares;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorsEnding1;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorSquaresEnding1;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorsEnding7;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorSquaresEnding7;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorsEnding9;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorSquaresEnding9;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorsEnding1;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorSquaresEnding1;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorsEnding3;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorSquaresEnding3;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorsEnding7;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorSquaresEnding7;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorsEnding9;
	internal readonly MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorSquaresEnding9;

	internal HeuristicGpuDivisorTables CreateHeuristicDivisorTables() => new(
			HeuristicCombinedDivisorsEnding1.View,
			HeuristicCombinedDivisorsEnding3.View,
			HeuristicCombinedDivisorsEnding7.View,
			HeuristicCombinedDivisorsEnding9.View,
			HeuristicCombinedDivisorSquaresEnding1.View,
			HeuristicCombinedDivisorSquaresEnding3.View,
			HeuristicCombinedDivisorSquaresEnding7.View,
			HeuristicCombinedDivisorSquaresEnding9.View,
			HeuristicGroupADivisors.View,
			HeuristicGroupADivisorSquares.View,
			HeuristicGroupBDivisorsEnding1.View,
			HeuristicGroupBDivisorSquaresEnding1.View,
			HeuristicGroupBDivisorsEnding7.View,
			HeuristicGroupBDivisorSquaresEnding7.View,
			HeuristicGroupBDivisorsEnding9.View,
			HeuristicGroupBDivisorSquaresEnding9.View);

	private static MemoryBuffer1D<ulong, Stride1D.Dense> CopySpanToDevice(Accelerator accelerator, AcceleratorStream stream, in ulong[] span)
	{
		MemoryBuffer1D<ulong, Stride1D.Dense>? buffer;
		// lock (accelerator)
		{
			buffer = accelerator.Allocate1D(stream, span);
		}

		return buffer;
	}

	private static MemoryBuffer1D<ulong, Stride1D.Dense> CopyUintSpanToDevice(Accelerator accelerator, AcceleratorStream stream, ReadOnlySpan<uint> span)
	{
		MemoryBuffer1D<ulong, Stride1D.Dense>? buffer;
		// lock (accelerator)
		{
			buffer = accelerator.Allocate1D<ulong>(span.Length);
		}

		if (!span.IsEmpty)
		{
			var converted = new ulong[span.Length];
			for (int i = 0; i < span.Length; i++)
			{
				converted[i] = span[i];
			}

			buffer.View.CopyFromCPU(stream, converted);
		}

		return buffer;
	}

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
