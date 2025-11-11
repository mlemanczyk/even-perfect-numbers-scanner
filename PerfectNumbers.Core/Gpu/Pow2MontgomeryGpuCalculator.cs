using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core.Gpu;

public static class Pow2MontgomeryGpuCalculator
{
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong CalculateKeep(ulong exponent, in MontgomeryDivisorData divisor)
	{

		var gpu = Pow2MontgomeryAccelerator.Rent(1);
		Accelerator? accelerator = gpu.Accelerator;
		AcceleratorStream stream = gpu.Stream!;
		MemoryBuffer1D<ulong, Stride1D.Dense> exponentBuffer = gpu.Input;
		MemoryBuffer1D<ulong, Stride1D.Dense> resultBuffer = gpu.Output;

		exponentBuffer.View.CopyFromCPU(stream, ref exponent, 1);
		// We don't need to worry about any left-overs here.
		// resultBuffer.MemSetToZero(stream);

		var kernel = gpu.KeepMontgomeryKernel;

		GpuPrimeWorkLimiter.Acquire();
		kernel(stream, 1, exponentBuffer.View, divisor, resultBuffer.View);

		ulong result = 0UL;
		resultBuffer.View.CopyToCPU(stream, ref result, 1);
		stream.Synchronize();
		GpuPrimeWorkLimiter.Release();

		Pow2MontgomeryAccelerator.Return(gpu);
		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong CalculateConvert(ulong exponent, in MontgomeryDivisorData divisor)
	{

		var gpu = Pow2MontgomeryAccelerator.Rent(1);
		AcceleratorStream stream = gpu.Stream!;
		MemoryBuffer1D<ulong, Stride1D.Dense> exponentBuffer = gpu.Input;
		MemoryBuffer1D<ulong, Stride1D.Dense> resultBuffer = gpu.Output;

		exponentBuffer.View.CopyFromCPU(stream, ref exponent, 1);
		// We don't need to worry about any left-overs here.
		// resultBuffer.MemSetToZero(stream);

		var kernel = gpu.ConvertToStandardKernel;

		GpuPrimeWorkLimiter.Acquire();
		kernel(stream, 1, exponentBuffer.View, divisor, resultBuffer.View);

		ulong result = 0UL;
		resultBuffer.View.CopyToCPU(stream, ref result, 1);
		stream.Synchronize();
		GpuPrimeWorkLimiter.Release();

		Pow2MontgomeryAccelerator.Return(gpu);
		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong CalculateConvert(Pow2MontgomeryAccelerator gpu, ulong exponent, in MontgomeryDivisorData divisor)
	{
		AcceleratorStream stream = gpu.Stream!;
		MemoryBuffer1D<ulong, Stride1D.Dense> exponentBuffer = gpu.Input;
		MemoryBuffer1D<ulong, Stride1D.Dense> resultBuffer = gpu.Output;

		exponentBuffer.View.CopyFromCPU(stream, ref exponent, 1);
		// We don't need to worry about any left-overs here.
		// resultBuffer.MemSetToZero(stream);

		var kernel = gpu.ConvertToStandardKernel;
		kernel(stream, 1, exponentBuffer.View, divisor, resultBuffer.View);

		ulong result = 0UL;
		resultBuffer.View.CopyToCPU(stream, ref result, 1);
		stream.Synchronize();

		return result;
	}
}
