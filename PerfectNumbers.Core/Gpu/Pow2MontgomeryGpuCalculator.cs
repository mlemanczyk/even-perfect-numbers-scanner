using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core.Gpu;

public static class Pow2MontgomeryGpuCalculator
{
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong CalculateKeep(PrimeOrderCalculatorAccelerator gpu, ulong exponent, in MontgomeryDivisorData divisor)
	{
		// GpuPrimeWorkLimiter.Acquire();

		ulong result = 0UL;
		int acceleratorIndex = gpu.AcceleratorIndex;
		Accelerator? accelerator = gpu.Accelerator;
		var exponentBufferView = gpu.InputView;
		var resultBufferView = gpu.OutputUlongView;

		AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		exponentBufferView.CopyFromCPU(stream, ref exponent, 1);
		// We don't need to worry about any left-overs here.
		// resultBuffer.MemSetToZero(stream);

		var kernelLauncher = gpu.KeepMontgomeryKernelLauncher;
		kernelLauncher(stream, 1, exponentBufferView, divisor.Modulus, divisor.NPrime, divisor.MontgomeryOne, divisor.MontgomeryTwo, divisor.MontgomeryTwoSquared, resultBufferView);

		resultBufferView.CopyToCPU(stream, ref result, 1);
		stream.Synchronize();
		AcceleratorStreamPool.Return(acceleratorIndex, stream);
		// GpuPrimeWorkLimiter.Release();

		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong CalculateConvert(PrimeOrderCalculatorAccelerator gpu, ulong exponent, in MontgomeryDivisorData divisor)
	{
		ulong result = 0UL;
		var exponentBufferView = gpu.InputView;
		var resultBufferView = gpu.OutputUlongView;

		int acceleratorIndex = gpu.AcceleratorIndex;
		AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		exponentBufferView.CopyFromCPU(stream, ref exponent, 1);
		// We don't need to worry about any left-overs here.
		// resultBuffer.MemSetToZero(stream);

		
		var accelerator = gpu.Accelerator;
		var kernelLauncher = gpu.ConvertToStandardKernelLauncher;
		kernelLauncher(stream, 1, exponentBufferView, divisor.Modulus, divisor.NPrime, divisor.MontgomeryOne, divisor.MontgomeryTwo, divisor.MontgomeryTwoSquared, resultBufferView);

		resultBufferView.CopyToCPU(stream, ref result, 1);
		stream.Synchronize();
		
		AcceleratorStreamPool.Return(acceleratorIndex, stream);

		return result;
	}
}
