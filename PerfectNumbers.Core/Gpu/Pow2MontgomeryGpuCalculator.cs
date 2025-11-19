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
		// GpuPrimeWorkLimiter.Acquire();

		ulong result = 0UL;
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		int acceleratorIndex = gpu.AcceleratorIndex;
		Accelerator? accelerator = gpu.Accelerator;
		MemoryBuffer1D<ulong, Stride1D.Dense> exponentBuffer = gpu.Input;
		MemoryBuffer1D<ulong, Stride1D.Dense> resultBuffer = gpu.OutputUlong;
		var kernel = gpu.KeepMontgomeryKernel;

		AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);

		exponentBuffer.View.CopyFromCPU(stream, ref exponent, 1);
		// We don't need to worry about any left-overs here.
		// resultBuffer.MemSetToZero(stream);

		var kernelLauncher = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>>();

		kernelLauncher(stream, 1, exponentBuffer.View, divisor, resultBuffer.View);

		resultBuffer.View.CopyToCPU(stream, ref result, 1);
		stream.Synchronize();
		AcceleratorStreamPool.Return(acceleratorIndex, stream);
		// GpuPrimeWorkLimiter.Release();

		PrimeOrderCalculatorAccelerator.Return(gpu);
		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong CalculateConvert(ulong exponent, in MontgomeryDivisorData divisor)
	{
		// GpuPrimeWorkLimiter.Acquire();

		ulong result = 0UL;
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		MemoryBuffer1D<ulong, Stride1D.Dense> exponentBuffer = gpu.Input;
		MemoryBuffer1D<ulong, Stride1D.Dense> resultBuffer = gpu.OutputUlong;

		int acceleratorIndex = gpu.AcceleratorIndex;
		AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		exponentBuffer.View.CopyFromCPU(stream, ref exponent, 1);
		// We don't need to worry about any left-overs here.
		// resultBuffer.MemSetToZero(stream);

		var kernelLauncher = gpu.ConvertToStandardKernelLauncher!;

		kernelLauncher(stream, 1, exponentBuffer.View, divisor, resultBuffer.View);			

		resultBuffer.View.CopyToCPU(stream, ref result, 1);
		stream.Synchronize();

		AcceleratorStreamPool.Return(acceleratorIndex, stream);
		// GpuPrimeWorkLimiter.Release();
		PrimeOrderCalculatorAccelerator.Return(gpu);

		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong CalculateConvert(PrimeOrderCalculatorAccelerator gpu, ulong exponent, in MontgomeryDivisorData divisor)
	{
		ulong result = 0UL;
		MemoryBuffer1D<ulong, Stride1D.Dense> exponentBuffer = gpu.Input;
		MemoryBuffer1D<ulong, Stride1D.Dense> resultBuffer = gpu.OutputUlong;

		int acceleratorIndex = gpu.AcceleratorIndex;
		AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		exponentBuffer.View.CopyFromCPU(stream, ref exponent, 1);
		// We don't need to worry about any left-overs here.
		// resultBuffer.MemSetToZero(stream);

		
		var accelerator = gpu.Accelerator;
		var kernelLauncher = gpu.ConvertToStandardKernelLauncher;
		kernelLauncher(stream, 1, exponentBuffer.View, divisor, resultBuffer.View);			

		resultBuffer.View.CopyToCPU(stream, ref result, 1);
		stream.Synchronize();
		
		AcceleratorStreamPool.Return(acceleratorIndex, stream);

		return result;
	}
}
