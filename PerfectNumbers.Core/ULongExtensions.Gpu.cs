using System.Runtime.CompilerServices;
using ILGPU.Algorithms;
using PerfectNumbers.Core.Gpu;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core;

public static partial class ULongExtensions
{
	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	[Obsolete("Use ULongExtensions.MulHighCpu or ULongExtensions.MulHighGpu.")]
	public static ulong MulHighGpuCompatible(this ulong x, ulong y)
	{
		GpuUInt128 product = new(x);
		GpuUInt128.Mul64(ref product, 0UL, y);
		return product.High;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	[Obsolete("Use ULongExtensions.MulModGpu for GPU-compatible host code or GpuUInt128.MulMod inside kernels.")]
	public static ulong MulMod64Gpu(this ulong a, ulong b, ulong modulus)
	{
		// TODO: Remove this GPU-compatible shim from production once callers migrate to MulMod64,
		// which the benchmarks show is roughly 6-7× faster on dense 64-bit inputs.
		GpuUInt128 state = new(a % modulus);
		return state.MulMod(b, modulus);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	[Obsolete("Use ULongExtensions.MulModGpu for GPU-compatible host code or GpuUInt128.MulMod inside kernels.")]
	public static ulong MulMod64GpuDeferred(this ulong a, ulong b, ulong modulus)
	{
		// TODO: Move this deferred helper to the benchmark suite; the baseline MulMod64 avoids the
		// 5-40× slowdown seen across real-world operand distributions.
		GpuUInt128 state = new(a);
		return state.MulModWithNativeModulo(b, modulus);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static ulong MulModGpu(this ulong left, ulong right, ulong modulus)
	{
		GpuUInt128 state = new(left);
		return state.MulMod(right, modulus);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static ulong Pow2MontgomeryModWindowedKeepGpu(this ulong exponent, PrimeOrderCalculatorAccelerator gpu, in MontgomeryDivisorData divisor)
		=> Pow2MontgomeryGpuCalculator.CalculateKeep(gpu, exponent, divisor);

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static ulong Pow2MontgomeryModWindowedConvertGpu(this ulong exponent, PrimeOrderCalculatorAccelerator gpu, in MontgomeryDivisorData divisor)
		=> Pow2MontgomeryGpuCalculator.CalculateConvert(gpu, exponent, divisor);

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static ulong Pow2MontgomeryModWithCycleGpu(this ulong exponent, PrimeOrderCalculatorAccelerator gpu, ulong cycleLength, in MontgomeryDivisorData divisor)
	{
		ulong rotationCount = exponent % cycleLength;
		return Pow2MontgomeryGpuCalculator.CalculateConvert(gpu, rotationCount, divisor);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static ulong Pow2MontgomeryModFromCycleRemainderGpu(this ulong reducedExponent, PrimeOrderCalculatorAccelerator gpu, in MontgomeryDivisorData divisor)
	{
		return Pow2MontgomeryGpuCalculator.CalculateConvert(gpu, reducedExponent, divisor);
	}
}
