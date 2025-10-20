using System.Numerics;
using System.Runtime.CompilerServices;
using ILGPU.Algorithms;

namespace PerfectNumbers.Core;

public static partial class ULongExtensions
{
	private const int Pow2WindowSize = 8;
	private const ulong Pow2WindowFallbackThreshold = 32UL;
	private const int Pow2WindowOddPowerCount = 1 << (Pow2WindowSize - 1);

	private static readonly byte[] BitStatsZeroCount = new byte[256];
	private static readonly byte[] BitStatsPrefixZero = new byte[256];
	private static readonly byte[] BitStatsSuffixZero = new byte[256];
	private static readonly byte[] BitStatsMaxZeroRun = new byte[256];

	static ULongExtensions()
	{
		int valueIndex = 0;
		int zeros;
		int pref;
		int suff;
		int maxIn;
		int run;
		int bit;
		bool isZero;
		int boundaryCounter;
		int boundaryBitIndex;
		for (; valueIndex < 256; valueIndex++)
		{
			zeros = 0;
			pref = 0;
			suff = 0;
			maxIn = 0;
			run = 0;

			for (bit = 7; bit >= 0; bit--)
			{
				// MSB-first within each byte: bit 7 down to bit 0.
				isZero = ((valueIndex >> bit) & 1) == 0;
				if (isZero)
				{
					zeros++;
					run++;
					if (run > maxIn)
					{
						maxIn = run;
					}
				}
				else
				{
					run = 0;
				}

				if (bit == 7)
				{
					// Leading zeros (prefix).
					boundaryCounter = 0;
					for (boundaryBitIndex = 7; boundaryBitIndex >= 0; boundaryBitIndex--)
					{
						if (((valueIndex >> boundaryBitIndex) & 1) == 0)
						{
							boundaryCounter++;
						}
						else
						{
							break;
						}
					}
					pref = boundaryCounter;
				}

				if (bit == 0)
				{
					// Trailing zeros (suffix).
					// Reuse boundaryCounter to count trailing zeros after the prefix detection above.
					boundaryCounter = 0;
					for (boundaryBitIndex = 0; boundaryBitIndex < 8; boundaryBitIndex++)
					{
						if (((valueIndex >> boundaryBitIndex) & 1) == 0)
						{
							boundaryCounter++;
						}
						else
						{
							break;
						}
					}
					suff = boundaryCounter;
				}
			}

			BitStatsZeroCount[valueIndex] = (byte)zeros;
			BitStatsPrefixZero[valueIndex] = (byte)pref;
			BitStatsSuffixZero[valueIndex] = (byte)suff;
			BitStatsMaxZeroRun[valueIndex] = (byte)maxIn;
		}
	}

	public static ulong CalculateOrder(this ulong q)
	{
		if (q <= 2UL)
		{
			return 0UL;
		}

		ulong order = q - 1UL, prime, temp;
		uint[] smallPrimes = PrimesGenerator.SmallPrimes;
		ulong[] smallPrimesPow2 = PrimesGenerator.SmallPrimesPow2;

		int i = 0, primesLength = smallPrimes.Length;
		UInt128 q128 = q,
				cycle = MersenneDivisorCycles.GetCycle(q128);
		// TODO: When the shared cycle snapshot cannot serve this divisor, trigger an on-demand
		// GPU computation (respecting the configured device) without promoting the result into
		// the cache so the order calculator still benefits from cycle stepping while keeping the
		// single-block memory plan intact.

		for (; i < primesLength; i++)
		{
			if (smallPrimesPow2[i] > order)
			{
				break;
			}

			prime = smallPrimes[i];
			// TODO: Replace this `%` driven factor peeling with the divisor-cycle aware
			// factoring helper so large orders reuse the cached remainders highlighted in
			// the latest divisor-cycle benchmarks instead of recomputing slow modulo checks.
			while (order % prime == 0UL)
			{
				temp = order / prime;
				if (temp.PowModWithCycle(q128, cycle) == UInt128.One)
				{
					order = temp;
				}
				else
				{
					break;
				}
			}
		}

		return order;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static int GetBitLength(this ulong value)
	{
		return 64 - BitOperations.LeadingZeroCount(value);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static void ComputeBitStats(this ulong value, out int bitLen, out int zeroCount, out int maxZeroBlock)
	{
		bitLen = 64 - int.CreateChecked(ulong.LeadingZeroCount(value));
		zeroCount = 0;
		maxZeroBlock = 0;
		if (bitLen <= 0)
		{
			return;
		}

		int msbIndex = (bitLen - 1) >> 3;
		int bitsInTopByte = ((bitLen - 1) & 7) + 1;
		int currentRun = 0;
		int byteIndex = msbIndex;
		int zeroCountInByte;
		int prefixZeros;
		int suffixZeros;
		int maxZeroRunInByte;
		int candidate;

		for (; byteIndex >= 0; byteIndex--)
		{
			byte inspectedByte = (byte)(value >> (byteIndex * 8));
			if (byteIndex == msbIndex && bitsInTopByte < 8)
			{
				// Mask off leading unused bits by setting them to 1 so the statistics ignore them.
				inspectedByte |= (byte)(0xFF << bitsInTopByte);
			}

			// TODO: Replace this byte-by-byte scan with the lookup-table based statistics collector validated in the BitStats benchmarks so zero runs leverage cached results instead of recomputing per bit.
			zeroCountInByte = BitStatsZeroCount[inspectedByte];
			zeroCount += zeroCountInByte;

			if (zeroCountInByte == 8)
			{
				currentRun += 8;
				if (currentRun > maxZeroBlock)
				{
					maxZeroBlock = currentRun;
				}

				continue;
			}

			prefixZeros = BitStatsPrefixZero[inspectedByte];
			suffixZeros = BitStatsSuffixZero[inspectedByte];
			maxZeroRunInByte = BitStatsMaxZeroRun[inspectedByte];

			candidate = currentRun + prefixZeros;
			if (candidate > maxZeroBlock)
			{
				maxZeroBlock = candidate;
			}

			if (maxZeroRunInByte > maxZeroBlock)
			{
				maxZeroBlock = maxZeroRunInByte;
			}

			currentRun = suffixZeros;
		}

		if (currentRun > maxZeroBlock)
		{
			maxZeroBlock = currentRun;
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong MultiplyShiftRight(ulong value, ulong multiplier, int shift)
	{
		UInt128 product = (UInt128)value * multiplier;
		return (ulong)(product >> shift);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong MultiplyShiftRightShiftFirst(ulong value, ulong multiplier, int shift)
	{
		ulong high = value >> shift;
		ulong mask = (1UL << shift) - 1UL;
		ulong low = value & mask;

		UInt128 highContribution = (UInt128)high * multiplier;
		UInt128 lowContribution = (UInt128)low * multiplier;

		UInt128 combined = highContribution + (lowContribution >> shift);
		return (ulong)combined;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong FastDiv64(this ulong value, ulong divisor, ulong mul)
	{
		ulong quotient = (ulong)(((UInt128)value * mul) >> 64);
		UInt128 remainder = (UInt128)value - ((UInt128)quotient * divisor);
		if (remainder >= divisor)
		{
			quotient++;
		}

		return quotient;
	}

	public const ulong WordBitMask = 0xFFFFUL;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsPrimeCandidate(this ulong n)
	{
		int i = 0;
		uint[] smallPrimes = PrimesGenerator.SmallPrimes;
		ulong[] smallPrimesPow2 = PrimesGenerator.SmallPrimesPow2;
		int len = smallPrimes.Length;
		ulong p;
		for (; i < len; i++)
		{
			if (smallPrimesPow2[i] > n)
			{
				break;
			}

			p = smallPrimes[i];
			// TODO: Swap this modulo check for the shared small-prime cycle filter once the
			// divisor-cycle cache is mandatory, matching the PrimeTester improvements noted in
			// the CPU sieve benchmarks.
			if ((n % p) == 0UL)
			{
				return n == p;
			}
		}

		return true;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsCompositeByGcd(this ulong value)
	{
		if (value < 2UL)
		{
			return true;
		}

		ulong exponentLog = (ulong)BitOperations.Log2(value);
		return value.BinaryGcd(exponentLog) > 1UL;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong BinaryGcd(this ulong a, ulong b)
	{
		if (a == 0UL)
		{
			return b;
		}

		if (b == 0UL)
		{
			return a;
		}

		int shift = BitOperations.TrailingZeroCount(a | b);
		a >>= BitOperations.TrailingZeroCount(a);

		do
		{
			b >>= BitOperations.TrailingZeroCount(b);

			if (a > b)
			{
				(a, b) = (b, a);
			}

			b -= a;
		}
		while (b != 0UL);

		return a << shift;
	}

	// Benchmarks (Mod5ULongBenchmarks) show the direct `% 5` is still cheaper (~0.26 ns vs 0.43 ns), so keep the modulo until a faster lookup is proven.
	// (Mod8/Mod10 stay masked because they win; Mod5 currently does not.)
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong Mod10(this ulong value) => (value & 1UL) == 0UL
			? (value % 5UL) switch
			{
				0UL => 0UL,
				1UL => 6UL,
				2UL => 2UL,
				3UL => 8UL,
				_ => 4UL,
			}
			: (value % 5UL) switch
			{
				0UL => 5UL,
				1UL => 1UL,
				2UL => 7UL,
				3UL => 3UL,
				_ => 9UL,
			};

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong Mod128(this ulong value) => value & 127UL;

	// Benchmarks confirm `%` beats our current Mod5/Mod3 helpers for 64-bit inputs, so leave these modulo operations in place until a superior lookup is available.
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static void Mod10_8_5_3(this ulong value, out ulong mod10, out ulong mod8, out ulong mod5, out ulong mod3)
	{
		mod8 = value & 7UL;
		// Benchmarks show `%` remains faster for the Mod5/Mod3 pair on ulong, so we keep the modulo path here for now.
		mod5 = value % 5UL;
		mod3 = value % 3UL;

		mod10 = (mod8 & 1UL) == 0UL
			? mod5 switch
			{
				0UL => 0UL,
				1UL => 6UL,
				2UL => 2UL,
				3UL => 8UL,
				_ => 4UL,
			}
			: mod5 switch
			{
				0UL => 5UL,
				1UL => 1UL,
				2UL => 7UL,
				3UL => 3UL,
				_ => 9UL,
			};
	}

	// Mod5/Mod3 lookup tables are currently slower on 64-bit operands; keep the direct modulo until benchmarks flip.
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static void Mod10_8_5_3Steps(this ulong value, out ulong step10, out ulong step8, out ulong step5, out ulong step3)
	{
		ulong mod8 = value & 7UL;
		// Same rationale: `%` wins in Mod5/Mod3 benches today, so avoid swapping until a faster lookup exists.
		ulong mod5 = value % 5UL;
		ulong mod3 = value % 3UL;
		ulong mod10 = (mod8 & 1UL) == 0UL
			? mod5 switch
			{
				0UL => 0UL,
				1UL => 6UL,
				2UL => 2UL,
				3UL => 8UL,
				_ => 4UL,
			}
			: mod5 switch
			{
				0UL => 5UL,
				1UL => 1UL,
				2UL => 7UL,
				3UL => 3UL,
				_ => 9UL,
			};

		step10 = mod10 + mod10;
		if (step10 >= 10UL)
		{
			step10 -= 10UL;
		}

		step8 = (mod8 + mod8) & 7UL;

		step5 = mod5 + mod5;
		if (step5 >= 5UL)
		{
			step5 -= 5UL;
		}

		step3 = mod3 + mod3;
		if (step3 >= 3UL)
		{
			step3 -= 3UL;
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static UInt128 Mul64(this ulong a, ulong b) => ((UInt128)a.MulHigh(b) << 64) | (UInt128)(a * b);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong MulHigh(this ulong x, ulong y)
	{
		// TODO: Investigate replacing this manual decomposition with the UInt128-based implementation
		// for CPU callers; the latest benchmarks show the intrinsic path is an order of magnitude
		// faster, while GPU code can keep using GpuUInt128.MulHigh.
		ulong xLow = (uint)x;
		ulong xHigh = x >> 32;
		ulong yLow = (uint)y;
		ulong yHigh = y >> 32;

		ulong w1 = xLow * yHigh;
		ulong w2 = xHigh * yLow;
		ulong w3 = xLow * yLow;

		// Matching the layout used in GpuUInt128.MulHigh: introducing the
		// intermediate result looks like one extra store, but it lets RyuJIT keep
		// the accumulated high word entirely in registers. Without this explicit
		// local the JIT spills the partial sum, which is where the performance
		// regression in the benchmarks came from.
		ulong result = (xHigh * yHigh) + (w1 >> 32) + (w2 >> 32);
		result += ((w3 >> 32) + (uint)w1 + (uint)w2) >> 32;
		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong AddMod64(this ulong value, ulong addend, ulong modulus)
	{
		if (modulus <= 1UL)
		{
			return 0UL;
		}

		UInt128 sum = (UInt128)value + addend;
		UInt128 mod = modulus;
		if (sum >= mod)
		{
			sum -= mod;
			if (sum >= mod)
			{
				sum -= mod;
			}
		}

		return (ulong)sum;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ulong FoldMontgomery(ulong value, ulong modulus)
	{
		if (modulus <= 1UL)
		{
			return 0UL;
		}

		if (value < modulus)
		{
			return value;
		}

		ulong folded = 0UL;
		int leadingZeros = BitOperations.LeadingZeroCount(value);
		int bitIndex = 63 - leadingZeros;
		for (; bitIndex >= 0; bitIndex--)
		{
			folded = folded.AddMod64(folded, modulus);
			if (((value >> bitIndex) & 1UL) != 0UL)
			{
				folded = folded.AddMod64(1UL, modulus);
			}
		}

		return folded;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong ModPow64(this ulong value, ulong exponent, ulong modulus)
	{
		if (modulus <= 1UL)
		{
			return 0UL;
		}

		ulong result = 1UL;
		value = FoldMontgomery(value, modulus);

		while (exponent != 0UL)
		{
			if ((exponent & 1UL) != 0UL)
			{
				result = MulMod64(result, value, modulus);
			}

			value = MulMod64(value, value, modulus);
			exponent >>= 1;
		}

		return result;
	}

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        // Reuse the direct UInt128 reduction path that dominated MulMod64Benchmarks to avoid
        // the redundant operand modulo operations in the older fallback.
        public static ulong MulMod64(this ulong a, ulong b, ulong modulus)
        {
                return (ulong)(((UInt128)a * b) % modulus);
        }

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong MontgomeryMultiply(this ulong a, ulong b, ulong modulus, ulong nPrime)
	{
		ulong tLow = unchecked(a * b);
		ulong m = unchecked(tLow * nPrime);
		ulong mTimesModulusLow = unchecked(m * modulus);

		ulong result = unchecked(a.MulHigh(b) + m.MulHigh(modulus) + (unchecked(tLow + mTimesModulusLow) < tLow ? 1UL : 0UL));
		if (result >= modulus)
		{
			result -= modulus;
		}

		return result;
	}

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong Pow2MontgomeryModSingleBit(ulong exponent, in MontgomeryDivisorData divisor, bool keepMontgomery)
	{
		ulong modulus = divisor.Modulus;
		ulong nPrime = divisor.NPrime;
		ulong result = divisor.MontgomeryOne;
		ulong baseVal = divisor.MontgomeryTwo;
		ulong remainingExponent = exponent;

		while (remainingExponent != 0UL)
		{
			if ((remainingExponent & 1UL) != 0UL)
			{
				result = result.MontgomeryMultiply(baseVal, modulus, nPrime);
			}

			remainingExponent >>= 1;
			if (remainingExponent == 0UL)
			{
				break;
			}

			baseVal = baseVal.MontgomeryMultiply(baseVal, modulus, nPrime);
		}

		return keepMontgomery ? result : result.MontgomeryMultiply(1UL, modulus, nPrime);
	}


	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static UInt128 PowMod(this ulong exponent, UInt128 modulus)
	{
		UInt128 result = UInt128.One;
		ulong exponentLoopIndex = 0UL;

		// TODO: Port this scalar PowMod fallback to the ProcessEightBitWindows helper so CPU callers get the
		// eight-bit window wins measured against the classic square-and-subtract implementation.
		// Return 1 because 2^0 = 1
		if (exponent == 0UL)
			return result;

		// Any number mod 1 is 0
		if (modulus == UInt128.One)
			return UInt128.Zero;

		// For small exponents, do classic method
		if (exponent < 64 || modulus < 4)
		{
			for (; exponentLoopIndex < exponent; exponentLoopIndex++)
			{
				result <<= 1;
				if (result >= modulus)
					result -= modulus;
			}

			return result;
		}

		// Special case: if modulus is a power of two, use bitmasking for efficiency and correctness
		if ((modulus & (modulus - 1)) == 0)
		{
			result = UInt128.One << (int)(exponent & 127);
			return result & (modulus - 1);
		}

		// Reusing exponentLoopIndex to iterate again over the exponent range for the general-case accumulation.
		exponentLoopIndex = 0UL;
		// Reusing result after resetting it for the general modulus accumulation phase.
		result = UInt128.One;
		for (; exponentLoopIndex < exponent; exponentLoopIndex++)
		{
			result <<= 1;
			if (result >= modulus)
				result -= modulus;
		}

		return result;
	}

	/// <summary>
	/// Computes 2^exponent mod modulus using a known cycle length.
	/// </summary>
	public static UInt128 PowModWithCycle(this ulong exponent, UInt128 modulus, ulong cycleLength)
	{
		UInt128 result = UInt128.One;
		ulong exponentLoopIndex = 0UL;

		// TODO: Wire this cycle-aware overload into the ProcessEightBitWindows helper so the reduced exponent path
		// inherits the faster windowed pow2 routine highlighted in the Pow2Montgomery benchmarks.
		// Return 1 because 2^0 = 1
		if (exponent == 0UL)
			return result;

		// Any number mod 1 is 0
		if (modulus == UInt128.One)
			return UInt128.Zero;

		// For small exponents, do classic method
		if (exponent < 64 || modulus < 4)
		{
			for (; exponentLoopIndex < exponent; exponentLoopIndex++)
			{
				result <<= 1;
				if (result >= modulus)
					result -= modulus;
			}

			return result;
		}

		// Special case: if modulus is a power of two, use bitmasking for efficiency and correctness
		if ((modulus & (modulus - 1)) == 0)
		{
			result = UInt128.One << (int)(exponent & 127);
			return result & (modulus - 1);
		}

		// Reusing exponentLoopIndex to iterate over the rotation count for the cycle-aware path.
		exponentLoopIndex = 0UL;
		// Reusing result after resetting it for the rotation-accumulation pass.
		result = UInt128.One;
		// TODO: Replace this modulo with the cached cycle remainder produced by the divisor-cycle cache so PowModWithCycle avoids
		// repeated `%` work, matching the ProcessEightBitWindows wins captured in Pow2MontgomeryModCycleComputationBenchmarks.
		ulong rotationCount = exponent % cycleLength;
		for (; exponentLoopIndex < rotationCount; exponentLoopIndex++)
		{
			result <<= 1;
			if (result >= modulus)
				result -= modulus;
		}

		return result;
	}

	/// <summary>
	/// Computes 2^exponent mod modulus using a known cycle length.
	/// </summary>
	public static UInt128 PowModWithCycle(this ulong exponent, UInt128 modulus, UInt128 cycleLength)
	{
		UInt128 result = UInt128.One;
		ulong exponentLoopIndex = 0UL;

		// TODO: Replace this UInt128-cycle overload with the ProcessEightBitWindows helper so large-exponent CPU scans
		// reuse the faster windowed pow2 ladder instead of the manual rotation loop measured to lag behind in benchmarks.
		// Return 1 because 2^0 = 1
		if (exponent == UInt128.Zero)
			return result;

		// Any number mod 1 is 0
		if (modulus == UInt128.One)
			return UInt128.Zero;

		// For small exponents, do classic method
		if (exponent < UInt128Numbers.SixtyFour || modulus < UInt128Numbers.Four)
		{
			for (; exponentLoopIndex < exponent; exponentLoopIndex++)
			{
				result <<= 1;
				if (result >= modulus)
					result -= modulus;
			}

			return result;
		}

		// Special case: if modulus is a power of two, use bitmasking for efficiency and correctness
		if ((modulus & (modulus - UInt128.One)) == UInt128.Zero)
		{
			result = UInt128.One << (int)(exponent & UInt128Numbers.OneHundredTwentySeven);
			return result & (modulus - 1);
		}

		// Reusing result after resetting it for the rotation-driven accumulation phase.
		result = UInt128.One;
		// TODO: Swap this modulo with the upcoming UInt128 cycle remainder helper so large-exponent scans reuse cached
		// reductions instead of recomputing `%` for every lookup, as demonstrated in Pow2MontgomeryModCycleComputationBenchmarks.
		UInt128 rotationCount = exponent % cycleLength;
		UInt128 rotationIndex = UInt128.Zero;
		while (rotationIndex < rotationCount)
		{
			result <<= 1;
			if (result >= modulus)
				result -= modulus;

			rotationIndex += UInt128.One;
		}

		return result;
	}

	/// <summary>
	/// Computes 2^exponent mod modulus using a known cycle length.
	/// </summary>
	public static UInt128 PowModWithCycle(this UInt128 exponent, UInt128 modulus, UInt128 cycleLength)
	{
		UInt128 one = UInt128.One,
				result = one,
				zero = UInt128.Zero;
		ulong exponentLoopIndex = 0UL;

		// TODO: Migrate this UInt128 exponent overload to ProcessEightBitWindows so the large-cycle reductions drop the
		// slow manual loop that underperforms the windowed pow2 helper in the Pow2 benchmark suite.
		// Return 1 because 2^0 = 1
		if (exponent == zero)
			return result;

		// Any number mod 1 is 0
		if (modulus == one)
			return zero;

		// For small exponents, do classic method
		if (exponent < UInt128Numbers.SixtyFour || modulus < UInt128Numbers.Four)
		{
			for (; exponentLoopIndex < exponent; exponentLoopIndex++)
			{
				result <<= 1;
				if (result >= modulus)
					result -= modulus;
			}

			return result;
		}

		// Special case: if modulus is a power of two, use bitmasking for efficiency and correctness
		if ((modulus & (modulus - one)) == zero)
		{
			result = one << (int)(exponent & UInt128Numbers.OneHundredTwentySeven);
			return result & (modulus - 1);
		}

		// Reusing result after resetting it for the rotation-driven accumulation phase.
		result = one;
		// TODO: Swap this modulo with the shared UInt128 cycle remainder helper once available so CRT powmods reuse cached
		// reductions in the windowed ladder, avoiding the `%` cost highlighted in Pow2MontgomeryModCycleComputationBenchmarks.
		UInt128 rotationCount = exponent % cycleLength;

		// We're reusing "zero" as rotation index for just a little better performance
		while (zero < rotationCount)
		{
			result <<= 1;
			if (result >= modulus)
				result -= modulus;

			zero += one;
		}

		return result;
	}

	/// <summary>
	/// Computes 2^exponent mod modulus using iterative CRT composition from mod 10 up to modulus.
	/// Only for modulus >= 10 and reasonable size.
	/// </summary>
	public static UInt128 PowModCrt(this ulong exponent, UInt128 modulus, MersenneDivisorCycles cycles)
	{
		if (modulus < 10)
			return PowMod(exponent, modulus); // fallback to classic

		// Use cycle length 4 for mod 10
		UInt128 currentModulus = 10,
				cycle,
				modulusCandidate = 11,
				remainderForCandidate,
				result = PowModWithCycle(exponent, 10, 4),
				zero = UInt128.Zero;

		for (; modulusCandidate <= modulus; modulusCandidate++)
		{
			cycle = MersenneDivisorCycles.GetCycle(modulusCandidate);
			remainderForCandidate = cycle > zero
					? PowModWithCycle(exponent, modulusCandidate, cycle)
					: PowMod(exponent, modulusCandidate);

			// Solve x ≡ result mod currentModulus
			//      x ≡ remM   mod m
			// Find x mod (currentModulus * m)
			// Since currentModulus and m are coprime, use CRT:
			// x = result + currentModulus * t, where t ≡ (remM - result) * inv(currentModulus, m) mod m

			// TODO: Replace this `% modulusCandidate` with the cached residue helper derived from Mod10_8_5_3Benchmarks so CRT
			// composition avoids repeated modulo divisions when combining residues for large divisor sets.
			result += currentModulus * ((remainderForCandidate + modulusCandidate - (result % modulusCandidate)) * ModInverse(currentModulus, modulusCandidate) % modulusCandidate);
			currentModulus *= modulusCandidate;

			if (currentModulus >= modulus)
				break;
		}

		// TODO: Swap this final `% modulus` with the pooled remainder cache so the CRT result write-back avoids one more division,
		// aligning with the optimizations captured in Mod10_8_5_3Benchmarks.
		return result % modulus;
	}

	// Helper: modular inverse (extended Euclidean algorithm)
	private static UInt128 ModInverse(UInt128 a, UInt128 m)
	{
		UInt128 m0 = m,
				originalA,
				originalM,
				temp,
				x0 = 0,
				x1 = 1;

		if (m == 1)
		{
			return 0;
		}

		while (a > 1)
		{
			originalA = a;
			originalM = m;
			m = originalA % originalM;
			a = originalM;
			temp = x0;
			x0 = x1 - (originalA / originalM) * x0;
			x1 = temp;
		}

		if (x1 < 0)
		{
			x1 += m0;
		}

		return x1;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool SharesFactorWithExponentMinusOne(this ulong exponent)
	{
		ulong prime, value = exponent - 1UL;
		value >>= BitOperations.TrailingZeroCount(value);
		uint[] smallPrimes = PrimesGenerator.SmallPrimes;
		ulong[] smallPrimesPow2 = PrimesGenerator.SmallPrimesPow2;
		int i = 1, smallPrimesLength = smallPrimes.Length;

		for (; i < smallPrimesLength && smallPrimesPow2[i] <= value; i++)
		{
			prime = smallPrimes[i];
			if (value % prime != 0UL)
			{
				continue;
			}

			if (exponent % prime.CalculateOrder() == 0UL)
			{
				return true;
			}

			do
			{
				value /= prime;
			}
			while (value % prime == 0UL);
		}

		if (value > 1UL && exponent % value.CalculateOrder() == 0UL)
		{
			return true;
		}

		return false;
	}

}
