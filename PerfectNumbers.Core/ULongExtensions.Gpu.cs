using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

public static partial class ULongExtensions
{
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong MulHighGpuCompatible(this ulong x, ulong y)
	{
		GpuUInt128 product = new(x);
		GpuUInt128.Mul64(ref product, 0UL, y);
		return product.High;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	[Obsolete("Use ULongExtensions.GpuCompatibleMulModSimplifiedExtension for GPU-compatible host code or GpuUInt128.MulMod inside kernels.")]
	public static ulong MulMod64Gpu(this ulong a, ulong b, ulong modulus)
	{
		// TODO: Remove this GPU-compatible shim from production once callers migrate to MulMod64,
		// which the benchmarks show is roughly 6-7× faster on dense 64-bit inputs.
		GpuUInt128 state = new(a % modulus);
		return state.MulMod(b, modulus);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	[Obsolete("Use ULongExtensions.GpuCompatibleMulModSimplifiedExtension for GPU-compatible host code or GpuUInt128.MulMod inside kernels.")]
	public static ulong MulMod64GpuDeferred(this ulong a, ulong b, ulong modulus)
	{
		// TODO: Move this deferred helper to the benchmark suite; the baseline MulMod64 avoids the
		// 5-40× slowdown seen across real-world operand distributions.
		GpuUInt128 state = new(a);
		return state.MulModWithNativeModulo(b, modulus);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong GpuCompatibleMulModSimplifiedExtension(this ulong left, ulong right, ulong modulus)
	{
		GpuUInt128 state = new(left);
		return state.MulModSimplified(right, modulus);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong Pow2ModWindowedGpu(this ulong exponent, ulong modulus)
	{
		// EvenPerfectBitScanner never routes moduli ≤ 1 through this path; keep the guard commented out so the hot loop stays branch-free.
		// if (modulus <= 1UL)
		// {
		//         return 0UL;
		// }

		// EvenPerfectBitScanner always works with positive exponents here; leave the defensive check commented for future reference.
		// if (exponent == 0UL)
		// {
		//         return 1UL % modulus;
		// }

		int bitLength = GetPortableBitLengthGpu(exponent);
		int windowSize = GetWindowSizeGpu(bitLength);
		int oddPowerCount = 1 << (windowSize - 1);

		// oddPowerCount is always ≥ 1 for the production workloads handled by EvenPerfectBitScanner.
		// if (oddPowerCount <= 0)
		// {
		//         return 1UL;
		// }

		ulong[] oddPowersArray = ThreadStaticPools.UlongPool.Rent(oddPowerCount);
		Span<ulong> oddPowers = oddPowersArray.AsSpan(0, oddPowerCount);

		InitializeStandardOddPowers(modulus, oddPowers);

		ulong result = 1UL;
		int index = bitLength - 1;

		while (index >= 0)
		{
			ulong currentBit = (exponent >> index) & 1UL;
			ulong squared = result.MulMod64(result, modulus);
			result = currentBit == 0UL ? squared : result;
			index = currentBit == 0UL ? index - 1 : index;
			if (currentBit == 0UL)
			{
				continue;
			}

			int windowStart = index - windowSize + 1;
			// EvenPerfectBitScanner guarantees windowStart stays non-negative for production exponents, so keep the guard commented out to avoid extra branching.
			// if (windowStart < 0)
			// {
			//         windowStart = 0;
			// }

			windowStart = GetNextSetBitIndexGpu(exponent, windowStart);

			int windowLength = index - windowStart + 1;
			for (int square = 0; square < windowLength; square++)
			{
				result = result.MulMod64(result, modulus);
			}

			ulong mask = (1UL << windowLength) - 1UL;
			ulong windowValue = (exponent >> windowStart) & mask;
			int tableIndex = (int)((windowValue - 1UL) >> 1);
			ulong multiplier = oddPowers[tableIndex];
			result = result.MulMod64(multiplier, modulus);

			index = windowStart - 1;
		}

		ThreadStaticPools.UlongPool.Return(oddPowersArray);

		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static ulong Pow2ModWindowedGpuKernel(ulong exponent, ulong modulus)
	{
		int bitLength = GetPortableBitLengthGpu(exponent);
		int windowSize = GetWindowSizeGpu(bitLength);
		ReadOnlyGpuUInt128 modulusWide = new(modulus);
		GpuUInt128 result = new(1UL);
		int index = bitLength - 1;

		while (index >= 0)
		{
			ulong currentBit = (exponent >> index) & 1UL;
			if (currentBit == 0UL)
			{
				result.MulMod(result.AsReadOnly(), modulusWide);
				index--;
				continue;
			}

			int windowStartCandidate = index - windowSize + 1;
			int negativeMask = windowStartCandidate >> 31;
			windowStartCandidate &= ~negativeMask;
			int windowStart = GetNextSetBitIndexGpu(exponent, windowStartCandidate);
			int windowLength = index - windowStart + 1;

			for (int square = 0; square < windowLength; square++)
			{
				result.MulMod(result.AsReadOnly(), modulusWide);
			}

			ulong mask = (1UL << windowLength) - 1UL;
			ulong windowValue = (exponent >> windowStart) & mask;
			ulong multiplier = ComputeWindowedOddPowerGpuKernel(windowValue, modulusWide);
			result.MulMod(multiplier, modulusWide);

			index = windowStart - 1;
		}

		return result.Low;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ulong ComputeWindowedOddPowerGpuKernel(ulong windowValue, in ReadOnlyGpuUInt128 modulus)
	{
		GpuUInt128 result = new(2UL);
		if (windowValue == 1UL)
		{
			// Single-bit windows show up whenever the bit scan finds an isolated exponent bit,
			// which happens routinely on the GPU path, so keep the fast return enabled.
			return 2UL;
		}

		ulong remaining = (windowValue - 1UL) >> 1;
		GpuUInt128 squareBase = new(2UL);
		squareBase.MulMod(squareBase.AsReadOnly(), modulus);

		while (remaining != 0UL)
		{
			if ((remaining & 1UL) != 0UL)
			{
				result.MulMod(squareBase.AsReadOnly(), modulus);
			}

			remaining >>= 1;
			if (remaining != 0UL)
			{
				squareBase.MulMod(squareBase.AsReadOnly(), modulus);
			}
		}

		return result.Low;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static ulong Pow2MontgomeryModWindowedGpuKeepMontgomery(in MontgomeryDivisorData divisor, ulong exponent)
	{
		return Pow2MontgomeryModWindowedGpuMontgomeryResult(divisor, exponent);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static ulong Pow2MontgomeryModWindowedGpuConvertToStandard(in MontgomeryDivisorData divisor, ulong exponent)
	{
		ulong modulus = divisor.Modulus;
		ulong nPrime = divisor.NPrime;
		ulong montgomeryResult = Pow2MontgomeryModWindowedGpuMontgomeryResult(divisor, exponent);
		return MontgomeryMultiply(montgomeryResult, 1UL, modulus, nPrime);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ulong Pow2MontgomeryModWindowedGpuMontgomeryResult(in MontgomeryDivisorData divisor, ulong exponent)
	{
		ulong modulus = divisor.Modulus;
		// Reuse the optimized MontgomeryMultiply extension highlighted in MontgomeryMultiplyBenchmarks so the GPU path matches the fastest CPU reduction.
		// We shouldn't hit it in production code. If we do, we're doing something wrong.
		// if (exponent == 0UL)
		// {
		//         return divisor.MontgomeryOne;
		// }

		// We barely ever hit it in production code. It's e.g. 772 calls out of billions
		// if (exponent <= Pow2WindowFallbackThreshold)
		// {
		//         return Pow2MontgomeryModSingleBit(exponent, divisor);
		// }

		int bitLength = GetPortableBitLengthGpu(exponent);
		int windowSize = GetWindowSizeGpu(bitLength);
		ulong result = divisor.MontgomeryOne;
		ulong nPrime = divisor.NPrime;

		int index = bitLength - 1;
		while (index >= 0)
		{
			ulong currentBit = (exponent >> index) & 1UL;
			ulong squared = MontgomeryMultiply(result, result, modulus, nPrime);
			bool processWindow = currentBit != 0UL;

			result = processWindow ? result : squared;
			index -= (int)(currentBit ^ 1UL);

			int windowStartCandidate = index - windowSize + 1;
			// if (windowStartCandidate < 0)
			// {
			//         windowStartCandidate = 0;
			// }
			// Clamp the negative offset without branching so the GPU loop stays divergence-free.
			// This still handles windows that would otherwise extend past the most significant bit.
			int negativeMask = windowStartCandidate >> 31;
			windowStartCandidate &= ~negativeMask;

			int windowStart = processWindow ? GetNextSetBitIndexGpu(exponent, windowStartCandidate) : windowStartCandidate;
			int windowLength = processWindow ? index - windowStart + 1 : 0;
			for (int square = 0; square < windowLength; square++)
			{
				result = MontgomeryMultiply(result, result, modulus, nPrime);
			}

			result = processWindow
					? MontgomeryMultiply(
							result,
							ComputeMontgomeryOddPowerGpu(
									(exponent >> windowStart) & ((1UL << windowLength) - 1UL),
									divisor,
									modulus,
									nPrime),
							modulus,
							nPrime)
					: result;

			index = processWindow ? windowStart - 1 : index;
		}

		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong Pow2MontgomeryModWindowedGpu(this ulong exponent, in MontgomeryDivisorData divisor, bool keepMontgomery)
	{
		// We shouldn't hit it in production code. If we do, we're doing something wrong.
		// ulong modulus = divisor.Modulus;
		// if (exponent == 0UL)
		// {
		//      return divisor.MontgomeryOne;
		// }

		// We barely ever hit it in production code. It's e.g. 772 calls out of billions
		// if (exponent <= Pow2WindowFallbackThreshold)
		// {
		//      return Pow2MontgomeryModSingleBit(exponent, divisor);
		// }

		return keepMontgomery
				? Pow2MontgomeryGpuExecutor.ExecuteKeep(exponent, divisor)
				: Pow2MontgomeryGpuExecutor.ExecuteConvert(exponent, divisor);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void InitializeStandardOddPowers(ulong modulus, Span<ulong> oddPowers)
	{
		// oddPowers spans at least one entry for production workloads; keep the guard commented out to avoid redundant checks in the hot loop.
		// if (oddPowers.IsEmpty)
		// {
		//         return;
		// }

		// EvenPerfectBitScanner feeds odd prime moduli (≥ 3) here, so the base value stays within range without a modulo reduction.
		ulong baseValue = 2UL;
		oddPowers[0] = baseValue;
		if (oddPowers.Length == 1)
		{
			return;
		}

		ulong square = baseValue.MulMod64(baseValue, modulus);
		for (int i = 1; i < oddPowers.Length; i++)
		{
			ulong previous = oddPowers[i - 1];
			oddPowers[i] = previous.MulMod64(square, modulus);
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ulong ComputeMontgomeryOddPowerGpu(ulong exponent, in MontgomeryDivisorData divisor, ulong modulus, ulong nPrime)
	{
		ulong baseValue = divisor.MontgomeryTwo;
		ulong power = divisor.MontgomeryOne;
		ulong remaining = exponent;

		while (remaining != 0UL)
		{
			ulong bit = remaining & 1UL;
			ulong multiplied = MontgomeryMultiply(power, baseValue, modulus, nPrime);
			ulong mask = (ulong)-(long)bit;
			power = (power & ~mask) | (multiplied & mask);

			remaining >>= 1;
			if (remaining == 0UL)
			{
				break;
			}

			baseValue = MontgomeryMultiply(baseValue, baseValue, modulus, nPrime);
		}

		return power;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static int GetPortableBitLengthGpu(ulong value)
	{
		// Keep this commented out. It will never happen in production code.
		// if (value == 0UL)
		// {
		//         return 0;
		// }

		return 64 - XMath.LeadingZeroCount(value);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static int GetWindowSizeGpu(int n)
	{
		int w = Pow2WindowSize;

		w = (n <= 671) ? 7 : w;
		w = (n <= 239) ? 6 : w;
		w = (n <= 79) ? 5 : w;
		w = (n <= 23) ? 4 : w;

		int m = (n >= 1) ? n : 1;
		w = (n <= 6) ? m : w;

		return w;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static int GetNextSetBitIndexGpu(ulong exponent, int startIndex)
	{
		ulong guard = (ulong)(((long)startIndex - 64) >> 63);
		int shift = startIndex & 63;
		ulong mask = (~0UL << shift) & guard;
		ulong masked = exponent & mask;

		return XMath.TrailingZeroCount(masked);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong Pow2MontgomeryModWithCycleGpu(this ulong exponent, ulong cycleLength, in MontgomeryDivisorData divisor)
	{
		ulong rotationCount = exponent % cycleLength;
		return Pow2MontgomeryGpuExecutor.ExecuteConvert(rotationCount, divisor);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong Pow2MontgomeryModFromCycleRemainderGpu(this ulong reducedExponent, in MontgomeryDivisorData divisor)
	{
		return Pow2MontgomeryGpuExecutor.ExecuteConvert(reducedExponent, divisor);
	}

	private static class Pow2MontgomeryGpuExecutor
	{
		private static readonly ConcurrentDictionary<Accelerator, Pow2MontgomeryKernelGroup> KernelCache = new();

		public static ulong ExecuteKeep(ulong exponent, in MontgomeryDivisorData divisor)
		{
			return Execute(exponent, divisor, useKeepKernel: true);
		}

		public static ulong ExecuteConvert(ulong exponent, in MontgomeryDivisorData divisor)
		{
			return Execute(exponent, divisor, useKeepKernel: false);
		}

		private static ulong Execute(ulong exponent, in MontgomeryDivisorData divisor, bool useKeepKernel)
		{
			ulong result = 0UL;

			GpuKernelLease lease = GpuKernelPool.GetKernel(useGpuOrder: true);
			var execution = lease.EnterExecutionScope();

			Accelerator accelerator = lease.Accelerator;
			// Keep this commented out. It should never happen in production code.
			// if (accelerator.AcceleratorType == AcceleratorType.CPU)
			// {
			//         return 0UL;
			// }

			var kernelGroup = KernelCache.GetOrAdd(accelerator, static accel => Pow2MontgomeryKernelGroup.Create(accel));

			var exponentBuffer = accelerator.Allocate1D<ulong>(1);
			var resultBuffer = accelerator.Allocate1D<ulong>(1);

			exponentBuffer.View.CopyFromCPU(ref exponent, 1);
			// We don't need to worry about any left-overs here.
			// resultBuffer.MemSetToZero();

			var kernel = useKeepKernel ? kernelGroup.KeepMontgomery : kernelGroup.ConvertToStandard;
			AcceleratorStream stream = lease.Stream;
			kernel(stream, 1, exponentBuffer.View, divisor, resultBuffer.View);
			stream.Synchronize();

			resultBuffer.View.CopyToCPU(ref result, 1);

			resultBuffer.Dispose();
			exponentBuffer.Dispose();
			execution.Dispose();
			lease.Dispose();
			// Keep this commented. We don't want to catch any exceptions. All should crash the scanner.
			// catch (AcceleratorException)
			// {
			//         return 0UL;
			// }
			// catch (NotSupportedException)
			// {
			//         return 0UL;
			// }
			// Intentionally avoid exception handling here; any accelerator failure should crash the scanner.
			return result;
		}

		private sealed class Pow2MontgomeryKernelGroup
		{
			internal Pow2MontgomeryKernelGroup(
					Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>> keepMontgomery,
					Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>> convertToStandard)
			{
				KeepMontgomery = keepMontgomery;
				ConvertToStandard = convertToStandard;
			}

			internal Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>> KeepMontgomery { get; }

			internal Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>> ConvertToStandard { get; }

			internal static Pow2MontgomeryKernelGroup Create(Accelerator accelerator)
			{
				var keepKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>(Pow2MontgomeryKernels.Pow2MontgomeryKernelKeepMontgomery);
				var keepLauncher = KernelUtil.GetKernel(keepKernel);
				var keepDelegate = keepLauncher.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>>();

				var convertKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>(Pow2MontgomeryKernels.Pow2MontgomeryKernelConvertToStandard);
				var convertLauncher = KernelUtil.GetKernel(convertKernel);
				var convertDelegate = convertLauncher.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>>();

				return new Pow2MontgomeryKernelGroup(keepDelegate, convertDelegate);
			}
		}

	}
}
