using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu.Accelerators;

public static class AcceleratorExtensions
{
	public static MemoryBuffer1D<ulong, Stride1D.Dense> CopySpanToDevice(this Accelerator accelerator, AcceleratorStream stream, in ulong[] span)
	{
		MemoryBuffer1D<ulong, Stride1D.Dense>? buffer;
		buffer = accelerator.Allocate1D(stream, span);

		return buffer;
	}

	public static MemoryBuffer1D<ulong, Stride1D.Dense> CopyUintSpanToDevice(this Accelerator accelerator, AcceleratorStream stream, ReadOnlySpan<uint> span)
	{
		MemoryBuffer1D<ulong, Stride1D.Dense>? buffer;
		buffer = accelerator.Allocate1D<ulong>(span.Length);

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
}
