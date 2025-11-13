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

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		Accelerator? accelerator = gpu.Accelerator;
		AcceleratorStream stream = AcceleratorStreamPool.Rent(accelerator);
		MemoryBuffer1D<ulong, Stride1D.Dense> exponentBuffer = gpu.Input;

		exponentBuffer.View.CopyFromCPU(stream, ref exponent, 1);
		// We don't need to worry about any left-overs here.
		// resultBuffer.MemSetToZero(stream);

		MemoryBuffer1D<ulong, Stride1D.Dense> resultBuffer = gpu.OutputUlong;
		var kernel = gpu.KeepMontgomeryKernel;
		var kernelLauncher = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>>();

		kernelLauncher(stream, 1, exponentBuffer.View, divisor, resultBuffer.View);

		ulong result = 0UL;
		resultBuffer.View.CopyToCPU(stream, ref result, 1);
		stream.Synchronize();
		AcceleratorStreamPool.Return(stream);
		GpuPrimeWorkLimiter.Release();

		PrimeOrderCalculatorAccelerator.Return(gpu);
		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong CalculateConvert(ulong exponent, in MontgomeryDivisorData divisor)
	{
		GpuPrimeWorkLimiter.Acquire();

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		AcceleratorStream stream = AcceleratorStreamPool.Rent(gpu.Accelerator);
		MemoryBuffer1D<ulong, Stride1D.Dense> exponentBuffer = gpu.Input;

		exponentBuffer.View.CopyFromCPU(stream, ref exponent, 1);
		// We don't need to worry about any left-overs here.
		// resultBuffer.MemSetToZero(stream);

		MemoryBuffer1D<ulong, Stride1D.Dense> resultBuffer = gpu.OutputUlong;
		var kernel = gpu.ConvertToStandardKernel;

		kernel.Launch(stream, 1, 1, exponentBuffer.View, divisor, resultBuffer.View);

		ulong result = 0UL;
		resultBuffer.View.CopyToCPU(stream, ref result, 1);
		stream.Synchronize();
		AcceleratorStreamPool.Return(stream);
		GpuPrimeWorkLimiter.Release();

		PrimeOrderCalculatorAccelerator.Return(gpu);
		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong CalculateConvert(PrimeOrderCalculatorAccelerator gpu, ulong exponent, in MontgomeryDivisorData divisor)
	{
		AcceleratorStream stream = AcceleratorStreamPool.Rent(gpu.Accelerator);
		MemoryBuffer1D<ulong, Stride1D.Dense> exponentBuffer = gpu.Input;

		exponentBuffer.View.CopyFromCPU(stream, ref exponent, 1);
		// We don't need to worry about any left-overs here.
		// resultBuffer.MemSetToZero(stream);

		MemoryBuffer1D<ulong, Stride1D.Dense> resultBuffer = gpu.OutputUlong;
		var kernel = gpu.ConvertToStandardKernel;
		var kernelLauncher = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>>();

		kernelLauncher(stream, 1,exponentBuffer.View, divisor, resultBuffer.View);
	
		ulong result = 0UL;
		resultBuffer.View.CopyToCPU(stream, ref result, 1);
		stream.Synchronize();
		AcceleratorStreamPool.Return(stream);

		return result;
	}
}
