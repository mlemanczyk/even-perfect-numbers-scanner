using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu.Kernels;

public static class Pow2MontgomeryKernels
{
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static GpuUInt128 AdvancePolynomialGpu(GpuUInt128 x, ulong c, ulong modulus)
	{
		x.Pow2();
		x.AddMod(c, modulus);
		return x;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ulong ComputeMontgomeryOddPowerGpu(ulong exponent, in MontgomeryDivisorDataGpu divisor, ulong modulus, ulong nPrime)
	{
		ulong baseValue = divisor.MontgomeryTwo;
		ulong power = divisor.MontgomeryOne;
		ulong remaining = exponent;

		while (remaining != 0UL)
		{
			ulong bit = remaining & 1UL;
			ulong multiplied = MontgomeryMultiplyGpu(power, baseValue, modulus, nPrime);
			ulong mask = (ulong)-(long)bit;
			power = (power & ~mask) | (multiplied & mask);

			remaining >>= 1;
			if (remaining == 0UL)
			{
				break;
			}

			baseValue = MontgomeryMultiplyGpu(baseValue, baseValue, modulus, nPrime);
		}

		return power;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static int GetPortableBitLengthGpu(ulong value) => 64 - XMath.LeadingZeroCount(value);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static int GetWindowSizeGpu(int n)
	{
		int w = PerfectNumberConstants.Pow2WindowSize;

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

		ulong square = baseValue.MulModGpu(baseValue, modulus);
		for (int i = 1; i < oddPowers.Length; i++)
		{
			ulong previous = oddPowers[i - 1];
			oddPowers[i] = previous.MulModGpu(square, modulus);
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static ulong MontgomeryMultiplyGpu(this ulong a, ulong b, ulong modulus, ulong nPrime)
	{
		// Mirror the struct-based MontMul64 helper measured as the second-fastest GPU-compatible
		// option in GpuUInt128Montgomery64Benchmarks so accelerator kernels reuse the validated path.
		MultiplyPartsGpu(a, b, out ulong productHigh, out ulong productLow);
		MultiplyPartsGpu(productLow, nPrime, out _, out ulong mLow);
		MultiplyPartsGpu(mLow, modulus, out ulong mTimesModulusHigh, out ulong mTimesModulusLow);

		ulong sumLow = unchecked(productLow + mTimesModulusLow);
		ulong carry = sumLow < productLow ? 1UL : 0UL;
		ulong sumHigh = unchecked(productHigh + mTimesModulusHigh + carry);

		if (sumHigh >= modulus)
		{
			sumHigh -= modulus;
		}

		return sumHigh;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
	private static void MultiplyPartsGpu(ulong left, ulong right, out ulong high, out ulong low)
	{
		ulong leftLow = (uint)left;
		ulong leftHigh = left >> 32;
		ulong rightLow = (uint)right;
		ulong rightHigh = right >> 32;

		ulong lowProduct = unchecked(leftLow * rightLow);
		ulong cross1 = unchecked(leftHigh * rightLow);
		ulong cross2 = unchecked(leftLow * rightHigh);
		ulong highProduct = unchecked(leftHigh * rightHigh);

		ulong carry = unchecked((lowProduct >> 32) + (uint)cross1 + (uint)cross2);
		low = unchecked((lowProduct & 0xFFFFFFFFUL) | (carry << 32));
		high = unchecked(highProduct + (cross1 >> 32) + (cross2 >> 32) + (carry >> 32));
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	internal static ulong Pow2MontgomeryModWindowedGpuKeepMontgomery(in MontgomeryDivisorDataGpu divisor, ulong exponent)
	{
		return Pow2MontgomeryModWindowedGpuMontgomeryResult(divisor, exponent);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static ulong Pow2MontgomeryModWindowedGpuConvertToStandard(this MontgomeryDivisorDataGpu divisor, ulong exponent)
	{
		ulong modulus = divisor.Modulus;
		ulong nPrime = divisor.NPrime;
		ulong montgomeryResult = Pow2MontgomeryModWindowedGpuMontgomeryResult(divisor, exponent);
		return MontgomeryMultiplyGpu(montgomeryResult, 1UL, modulus, nPrime);
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static ulong Pow2MontgomeryModWindowedGpuMontgomeryResult(in MontgomeryDivisorDataGpu divisor, ulong exponent)
	{
		ulong modulus = divisor.Modulus;
		// Reuse the GPU-compatible MontgomeryMultiply helper highlighted in MontgomeryMultiplyBenchmarks so the accelerator path
		// mirrors the fastest CPU reduction without relying on UInt128 intrinsics.
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
			ulong squared = MontgomeryMultiplyGpu(result, result, modulus, nPrime);
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
				result = MontgomeryMultiplyGpu(result, result, modulus, nPrime);
			}

			result = processWindow
					? MontgomeryMultiplyGpu(
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
			ulong squared = result.MulModGpu(result, modulus);
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
				result = result.MulModGpu(result, modulus);
			}

			ulong mask = (1UL << windowLength) - 1UL;
			ulong windowValue = (exponent >> windowStart) & mask;
			int tableIndex = (int)((windowValue - 1UL) >> 1);
			ulong multiplier = oddPowers[tableIndex];
			result = result.MulModGpu(multiplier, modulus);

			index = windowStart - 1;
		}

		ThreadStaticPools.UlongPool.Return(oddPowersArray);

		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong Pow2ModWindowedGpuKernel(this ulong exponent, ulong modulus)
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

	public static void Pow2MontgomeryKernelKeepMontgomery(
		Index1D index,
		ArrayView1D<ulong, Stride1D.Dense> exponents,
		ulong divisorModulus,
		ulong divisorNPrime,
		ulong divisorMontgomeryOne,
		ulong divisorMontgomeryTwo,
		ulong divisorMontgomeryTwoSquared,
		ArrayView1D<ulong, Stride1D.Dense> results)
	{
		MontgomeryDivisorDataGpu divisor = new(divisorModulus, divisorNPrime, divisorMontgomeryOne, divisorMontgomeryTwo, divisorMontgomeryTwoSquared);
		results[0] = Pow2MontgomeryModWindowedGpuKeepMontgomery(divisor, exponents[0]);
	}

	public static void Pow2MontgomeryKernelConvertToStandard(
		Index1D index,
		ArrayView1D<ulong, Stride1D.Dense> exponents,
		ulong divisorModulus,
		ulong divisorNPrime,
		ulong divisorMontgomeryOne,
		ulong divisorMontgomeryTwo,
		ulong divisorMontgomeryTwoSquared,
		ArrayView1D<ulong, Stride1D.Dense> results)
	{
		MontgomeryDivisorDataGpu divisor = new(divisorModulus, divisorNPrime, divisorMontgomeryOne, divisorMontgomeryTwo, divisorMontgomeryTwoSquared);
		results[0] = Pow2MontgomeryModWindowedGpuConvertToStandard(divisor, exponents[0]);
	}


	public static void TryPollardRhoKernel(ulong n, int limit, ArrayView1D<ulong, Stride1D.Dense> randomState, ArrayView1D<byte, Stride1D.Dense> factored, ArrayView1D<ulong, Stride1D.Dense> factor)
	{
		factored[0] = 0;
		if ((n & 1UL) == 0UL)
		{
			factor[0] = 2UL;
			factored[0] = 1;
			return;
		}

		int attempt = 0;
		GpuUInt128 nGpuUInt128 = (GpuUInt128)n;
		DeterministicRandomGpu random = new(randomState[0]);
		while (attempt < limit)
		{
			ulong c = (random.NextUInt64() % (n - 1UL)) + 1UL;
			GpuUInt128 x = (random.NextUInt64() % (n - 2UL)) + 2UL;
			GpuUInt128 y = x;
			GpuUInt128 d = GpuUInt128.One;

			while (d.CompareTo(GpuUInt128.One) == 0)
			{
				x = AdvancePolynomialGpu(x, c, modulus: n);
				y = AdvancePolynomialGpu(y, c, n);
				y = AdvancePolynomialGpu(y, c, n);
				GpuUInt128 diff = x.CompareTo(y) > 0 ? x - y : y - x;
				d = GpuUInt128.BinaryGcd(diff, nGpuUInt128);
			}

			if (d != n)
			{
				if (d.High == 0)
				{
					factor[0] = d.Low;
					factored[0] = 1;
					return;
				}
			}

			attempt++;
		}

		randomState[0] = random.State;
	}
}
