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
		GpuPrimeWorkLimiter.Acquire();

		var gpu = Pow2MontgomeryAccelerator.Rent();
		Accelerator? accelerator = gpu.Accelerator;
		AcceleratorStream stream = gpu.Stream!;
		MemoryBuffer1D<ulong, Stride1D.Dense> exponentBuffer = gpu.Input;
		MemoryBuffer1D<ulong, Stride1D.Dense> resultBuffer = gpu.Output;

		exponentBuffer.View.CopyFromCPU(stream, ref exponent, 1);
		// We don't need to worry about any left-overs here.
		// resultBuffer.MemSetToZero(stream);

		var kernel = gpu.KeepMontgomeryKernel;
		kernel(stream, 1, exponentBuffer.View, divisor, resultBuffer.View);

		ulong result = 0UL;
		resultBuffer.View.CopyToCPU(stream, ref result, 1);
		stream.Synchronize();

		Pow2MontgomeryAccelerator.Return(gpu);
		GpuPrimeWorkLimiter.Release();
		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong CalculateConvert(ulong exponent, in MontgomeryDivisorData divisor)
	{
		GpuPrimeWorkLimiter.Acquire();

		var gpu = Pow2MontgomeryAccelerator.Rent();
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

		Pow2MontgomeryAccelerator.Return(gpu);
		GpuPrimeWorkLimiter.Release();
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
