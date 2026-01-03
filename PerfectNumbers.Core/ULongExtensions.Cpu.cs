using System.Numerics;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public static partial class ULongExtensions
{
	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static ulong Pow2MontgomeryModWindowedKeepMontgomeryCpu(this ulong exponent, in MontgomeryDivisorData divisor)
	{
		ulong modulus = divisor.Modulus;
		// This should never happen in production code.
		// if (exponent == 0UL)
		// {
		//     return keepMontgomery ? divisor.MontgomeryOne : 1UL % modulus;
		// }

		// Small divisors come from the precomputed <= 4,000,000 snapshot and tiny rotation counts
		// appear when cycle reductions trigger. Route both cases through the single-bit ladder so the
		// CPU path avoids renting the window tables when a cheaper method is available.
		if (exponent <= Pow2WindowFallbackThreshold || modulus <= PerfectNumberConstants.MaxQForDivisorCycles)
		{
			return Pow2MontgomeryModSingleBitKeepMontgomery(exponent, divisor);
		}

		int bitLength = GetPortableBitLengthCpu(exponent);
		int windowSize = GetWindowSizeCpu(bitLength);
		int oddPowerCount = 1 << (windowSize - 1);

		ulong result = divisor.MontgomeryOne;
		ulong nPrime = divisor.NPrime;

		ulong[] oddPowersArray = ThreadStaticPools.UlongPool.Rent(oddPowerCount);
		Span<ulong> oddPowers = oddPowersArray.AsSpan(0, oddPowerCount);
		InitializeMontgomeryOddPowersCpu(divisor, modulus, nPrime, oddPowers);

		int index = bitLength - 1;
		while (index >= 0)
		{
			if (((exponent >> index) & 1UL) == 0UL)
			{
				result = result.MontgomeryMultiplyCpu(result, modulus, nPrime);
				index--;
				continue;
			}

			int windowStart = index - windowSize + 1;
			if (windowStart < 0)
			{
				windowStart = 0;
			}

			while (((exponent >> windowStart) & 1UL) == 0UL)
			{
				windowStart++;
			}

			int windowLength = index - windowStart + 1;
			for (int square = 0; square < windowLength; square++)
			{
				result = result.MontgomeryMultiplyCpu(result, modulus, nPrime);
			}

			ulong mask = (1UL << windowLength) - 1UL;
			ulong windowValue = (exponent >> windowStart) & mask;
			int tableIndex = (int)((windowValue - 1UL) >> 1);
			ulong multiplier = oddPowers[tableIndex];
			result = result.MontgomeryMultiplyCpu(multiplier, modulus, nPrime);

			index = windowStart - 1;
		}

		ThreadStaticPools.UlongPool.Return(oddPowersArray);

		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static ulong Pow2MontgomeryModWindowedConvertToStandardCpu(this ulong exponent, in MontgomeryDivisorData divisor)
	{
		ulong modulus = divisor.Modulus;
		// This should never happen in production code.
		// if (exponent == 0UL)
		// {
		//     return keepMontgomery ? divisor.MontgomeryOne : 1UL % modulus;
		// }

		// Small divisors come from the precomputed <= 4,000,000 snapshot and tiny rotation counts
		// appear when cycle reductions trigger. Route both cases through the single-bit ladder so the
		// CPU path avoids renting the window tables when a cheaper method is available.
		if (exponent <= Pow2WindowFallbackThreshold || modulus <= PerfectNumberConstants.MaxQForDivisorCycles)
		{
			return Pow2MontgomeryModSingleBitConvertToStandard(exponent, divisor);
		}

		int bitLength = GetPortableBitLengthCpu(exponent);
		int windowSize = GetWindowSizeCpu(bitLength);
		int oddPowerCount = 1 << (windowSize - 1);

		ulong result = divisor.MontgomeryOne;
		ulong nPrime = divisor.NPrime;

		ulong[] oddPowersArray = ThreadStaticPools.UlongPool.Rent(oddPowerCount);
		Span<ulong> oddPowers = oddPowersArray.AsSpan(0, oddPowerCount);
		InitializeMontgomeryOddPowersCpu(divisor, modulus, nPrime, oddPowers);

		int index = bitLength - 1;
		while (index >= 0)
		{
			if (((exponent >> index) & 1UL) == 0UL)
			{
				result = result.MontgomeryMultiplyCpu(result, modulus, nPrime);
				index--;
				continue;
			}

			int windowStart = index - windowSize + 1;
			if (windowStart < 0)
			{
				windowStart = 0;
			}

			while (((exponent >> windowStart) & 1UL) == 0UL)
			{
				windowStart++;
			}

			int windowLength = index - windowStart + 1;
			for (int square = 0; square < windowLength; square++)
			{
				result = result.MontgomeryMultiplyCpu(result, modulus, nPrime);
			}

			ulong mask = (1UL << windowLength) - 1UL;
			ulong windowValue = (exponent >> windowStart) & mask;
			int tableIndex = (int)((windowValue - 1UL) >> 1);
			ulong multiplier = oddPowers[tableIndex];
			result = result.MontgomeryMultiplyCpu(multiplier, modulus, nPrime);

			index = windowStart - 1;
		}

		ThreadStaticPools.UlongPool.Return(oddPowersArray);

		result = result.MontgomeryMultiplyCpu(1UL, modulus, nPrime);
		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static void InitializeMontgomeryOddPowersCpu(in MontgomeryDivisorData divisor, ulong modulus, ulong nPrime, Span<ulong> oddPowers)
	{
		oddPowers[0] = divisor.MontgomeryTwo;
		if (oddPowers.Length == 1)
		{
			return;
		}

		ulong previous;
		ulong square = divisor.MontgomeryTwoSquared;

		for (int i = 1; i < oddPowers.Length; i++)
		{
			previous = oddPowers[i - 1];
			oddPowers[i] = previous.MontgomeryMultiplyCpu(square, modulus, nPrime);
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static ulong Pow2MontgomeryModWithCycleConvertToStandardCpu(this ulong exponent, ulong cycleLength, in MontgomeryDivisorData divisor)
	{
		ulong rotationCount = exponent % cycleLength;
		return Pow2MontgomeryModWindowedConvertToStandardCpu(rotationCount, divisor);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static ulong Pow2MontgomeryModFromCycleRemainderCpu(this ulong reducedExponent, in MontgomeryDivisorData divisor)
	{
		return Pow2MontgomeryModWindowedConvertToStandardCpu(reducedExponent, divisor);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static ulong CalculateMersenneDivisorCycleLengthUnrolledHexCpu(this ulong divisor)
	{
		// EvenPerfectBitScanner only routes odd divisors here; keep the guard commented out for benchmarks.
		// if ((divisor & (divisor - 1UL)) == 0UL)
		// {
		//         return 1UL;
		// }

		ulong order = 1UL;
		ulong pow = 2UL;

		while (true)
		{
			if (AdvanceMersenneDivisorCycleStepCpu(ref pow, divisor, ref order))
			{
				return order;
			}

			if (AdvanceMersenneDivisorCycleStepCpu(ref pow, divisor, ref order))
			{
				return order;
			}

			if (AdvanceMersenneDivisorCycleStepCpu(ref pow, divisor, ref order))
			{
				return order;
			}

			if (AdvanceMersenneDivisorCycleStepCpu(ref pow, divisor, ref order))
			{
				return order;
			}

			if (AdvanceMersenneDivisorCycleStepCpu(ref pow, divisor, ref order))
			{
				return order;
			}

			if (AdvanceMersenneDivisorCycleStepCpu(ref pow, divisor, ref order))
			{
				return order;
			}

			if (AdvanceMersenneDivisorCycleStepCpu(ref pow, divisor, ref order))
			{
				return order;
			}

			if (AdvanceMersenneDivisorCycleStepCpu(ref pow, divisor, ref order))
			{
				return order;
			}

			if (AdvanceMersenneDivisorCycleStepCpu(ref pow, divisor, ref order))
			{
				return order;
			}

			if (AdvanceMersenneDivisorCycleStepCpu(ref pow, divisor, ref order))
			{
				return order;
			}

			if (AdvanceMersenneDivisorCycleStepCpu(ref pow, divisor, ref order))
			{
				return order;
			}

			if (AdvanceMersenneDivisorCycleStepCpu(ref pow, divisor, ref order))
			{
				return order;
			}

			if (AdvanceMersenneDivisorCycleStepCpu(ref pow, divisor, ref order))
			{
				return order;
			}

			if (AdvanceMersenneDivisorCycleStepCpu(ref pow, divisor, ref order))
			{
				return order;
			}

			if (AdvanceMersenneDivisorCycleStepCpu(ref pow, divisor, ref order))
			{
				return order;
			}

			if (AdvanceMersenneDivisorCycleStepCpu(ref pow, divisor, ref order))
			{
				return order;
			}
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static bool AdvanceMersenneDivisorCycleStepCpu(ref ulong pow, ulong divisor, ref ulong order)
	{
		pow += pow;
		if (pow >= divisor)
		{
			pow -= divisor;
		}

		order++;
		return pow == 1UL;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static int GetPortableBitLengthCpu(ulong value)
	{
		// Keep this commented out. It will never happen in production code.
		// if (value == 0UL)
		// {
		//         return 0;
		// }

		return 64 - BitOperations.LeadingZeroCount(value);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static int GetWindowSizeCpu(int bitLength)
	{
		if (bitLength <= 6)
		{
			return Math.Max(bitLength, 1);
		}

		if (bitLength <= 23)
		{
			return 4;
		}

		if (bitLength <= 79)
		{
			return 5;
		}

		if (bitLength <= 239)
		{
			return 6;
		}

		if (bitLength <= 671)
		{
			return 7;
		}

		return PerfectNumberConstants.Pow2WindowSize;
	}

}
