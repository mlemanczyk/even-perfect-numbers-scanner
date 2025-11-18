using System.Numerics;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public static partial class ULongExtensions
{
	private const int Pow2WindowSize = 8;
	private const ulong Pow2WindowFallbackThreshold = 32UL;
	private const int Pow2WindowOddPowerCount = 1 << (Pow2WindowSize - 1);

        private static readonly UInt128 Pow2WindowedModulusThreshold = (UInt128)PerfectNumberConstants.MaxQForDivisorCycles;

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

		while (true)
		{
			b >>= BitOperations.TrailingZeroCount(b);

			bool swap = a > b;
			ulong min = Select(a, b, swap);
			ulong max = Select(b, a, swap);
			b = max - min;
			a = min;

			if (b == 0UL)
			{
				return a << shift;
			}
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ulong Select(ulong left, ulong right, bool useRight)
	{
		ulong mask = useRight ? ulong.MaxValue : 0UL;
		return left ^ ((left ^ right) & mask);
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
	public static UInt128 Mul64(this ulong a, ulong b) => ((UInt128)a.MulHighCpu(b) << 64) | (UInt128)(a * b);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong MulHighCpu(this ulong x, ulong y)
	{
		return (ulong)(((UInt128)x * y) >> 64);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ulong MulHighGpu(this ulong x, ulong y)
	{
		// Retain this manual decomposition for GPU-style arithmetic so callers staging accelerator work
		// can avoid the UInt128 intrinsic while keeping parity with GpuUInt128.MulHigh.
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
		// Every CPU caller (Pollard Rho and the ModPow64 ladder) supplies moduli of at least three, so leave this guard
		// documented but disabled.
		// if (modulus <= 1UL)
		// {
		// 	return 0UL;
		// }

		// Both flows keep their operands below the modulus ahead of time, letting us skip the re-fold the old helper performed.
		UInt128 sum = (UInt128)value + addend;
		if (sum < modulus)
		{
			return (ulong)sum;
		}

		sum -= modulus;
		// After one subtraction the sum always falls below the modulus on this path, making the old guard redundant.
		// if (sum >= modulus)
		// {
		// 	sum -= modulus;
		// }

		return (ulong)sum;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ulong FoldMontgomery(ulong value, ulong modulus)
	{
		// Pollard Rho and ModPow64 both feed moduli of at least three, so retain the documentation and skip the runtime guard.
		// if (modulus <= 1UL)
		// {
		// 	return 0UL;
		// }

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
		// Pollard Rho and the Mersenne testers never pass moduli below two, so keep the guard commented for clarity.
		// if (modulus <= 1UL)
		// {
		// 	return 0UL;
		// }

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
	public static ulong MontgomeryMultiplyCpu(this ulong a, ulong b, ulong modulus, ulong nPrime)
	{
		ulong tLow = unchecked(a * b);
		ulong m = unchecked(tLow * nPrime);
		ulong mTimesModulusLow = unchecked(m * modulus);

		ulong result = unchecked(a.MulHighCpu(b) + m.MulHighCpu(modulus) + (unchecked(tLow + mTimesModulusLow) < tLow ? 1UL : 0UL));
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
				result = result.MontgomeryMultiplyCpu(baseVal, modulus, nPrime);
			}

			remainingExponent >>= 1;
			if (remainingExponent == 0UL)
			{
				break;
			}

			baseVal = baseVal.MontgomeryMultiplyCpu(baseVal, modulus, nPrime);
		}

		return keepMontgomery ? result : result.MontgomeryMultiplyCpu(1UL, modulus, nPrime);
	}


	[MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt128 PowMod(this ulong exponent, UInt128 modulus)
        {
                // EvenPerfectBitScanner starts CPU scans at p >= 138,000,000 (Program.cs),
                // so production never routes exponent == 0 here. Keep the guard commented out
                // for targeted micro-benchmarks.
                // if (exponent == 0UL)
                // {
                //         return UInt128.One;
                // }

                // MersenneResidueAutomaton seeds q = 2 * p * k + 1 (ModuloAutomata.cs),
                // making modulus >= 2 * p + 1 > 276,000,000 and odd on the CPU path. The
                // modulus == 1 guard is therefore unreachable during EvenPerfectBitScanner
                // runs.
                // if (modulus == UInt128.One)
                // {
                //         return UInt128.Zero;
                // }

                // Production exponents stay above 64 bits and modulus never drops below 4
                // for admissible divisors, so the classic doubling fallback does not
                // trigger in scans.
                // if (exponent < 64 || modulus < 4)
                // {
                //         return Pow2ByDoubling(exponent, modulus);
                // }

                // Every admissible divisor q is odd (2 * p * k + 1), preventing the
                // power-of-two mask path from activating in production.
                // if ((modulus & (modulus - 1)) == 0)
                // {
                //         UInt128 masked = UInt128.One << (int)(exponent & 127);
                //         return masked & (modulus - 1);
                // }

                if (!ShouldUseWindowedPow2(modulus))
                {
                        return Pow2ByDoubling(exponent, modulus);
                }

                return exponent.Pow2MontgomeryModWindowed(modulus);
        }


	/// <summary>
	/// Computes 2^exponent mod modulus using a known cycle length.
	/// </summary>
        
        public static UInt128 PowModWithCycle(this ulong exponent, UInt128 modulus, ulong cycleLength)
        {
                // EvenPerfectBitScanner keeps exponent >= 138,000,000 on CPU (Program.cs),
                // so the zero-exponent guard stays disabled outside tests.
                // if (exponent == 0UL)
                // {
                //         return UInt128.One;
                // }

                // The divisor generator produces odd q = 2 * p * k + 1 (ModuloAutomata.cs),
                // so modulus never reaches 1 in production.
                // if (modulus == UInt128.One)
                // {
                //         return UInt128.Zero;
                // }

                // Every admissible divisor is odd, so the power-of-two shortcut cannot
                // trigger while scanning real candidates.
                // if ((modulus & (modulus - 1)) == 0)
                // {
                //         UInt128 masked = UInt128.One << (int)(exponent & 127);
                //         return masked & (modulus - 1);
                // }

                bool hasCycle = cycleLength != 0UL;
                // TODO: Replace this modulo with the cached cycle remainder produced by the divisor-cycle cache so PowModWithCycle avoids
                // repeated `%` work, matching the ProcessEightBitWindows wins captured in Pow2MontgomeryModCycleComputationBenchmarks.
                ulong rotationCount = hasCycle && exponent >= cycleLength ? exponent % cycleLength : exponent;

                // Keep this fast path: valid divisors satisfy cycle | exponent, producing
                // a zero remainder that means 2^exponent ≡ 1 (mod q).
                if (hasCycle && rotationCount == 0UL)
                {
                        return UInt128.One;
                }

                // FastDiv64 can return tiny divisors (see MersenneNumberIncrementalCpuTester),
                // so the small-rotation fallback still fires on the CPU path.
                if (modulus < 4 || rotationCount < 64)
                {
                        return Pow2ByDoubling(rotationCount, modulus);
                }

                if (ShouldUseWindowedPow2(modulus))
                {
                        UInt128 exponent128 = exponent;
                        UInt128 rotationCount128 = rotationCount;
                        return exponent128.Pow2MontgomeryModWindowedWithCycle(modulus, cycleLength, rotationCount128);
                }

                return Pow2ByDoubling(rotationCount, modulus);
        }




	/// <summary>
	/// Computes 2^exponent mod modulus using a known cycle length.
	/// </summary>
        
        public static UInt128 PowModWithCycle(this ulong exponent, UInt128 modulus, UInt128 cycleLength)
        {
                // CPU scans only feed exponents >= 138,000,000 (Program.cs), so keep the
                // zero-exponent shortcut commented out for tests.
                // if (exponent == 0UL)
                // {
                //         return UInt128.One;
                // }

                // Modulus originates from q = 2 * p * k + 1 (ModuloAutomata.cs), therefore
                // modulus == 1 never occurs outside dedicated benchmarks.
                // if (modulus == UInt128.One)
                // {
                //         return UInt128.Zero;
                // }

                // Admissible q are odd, rendering the power-of-two shortcut unreachable.
                // if ((modulus & (modulus - UInt128.One)) == UInt128.Zero)
                // {
                //         UInt128 masked = UInt128.One << (int)(exponent & UInt128Numbers.OneHundredTwentySeven);
                //         return masked & (modulus - UInt128.One);
                // }

                bool hasCycle = cycleLength != UInt128.Zero;
                UInt128 exponent128 = exponent;
                // TODO: Swap this modulo with the upcoming UInt128 cycle remainder helper so large-exponent scans reuse cached
                // reductions instead of recomputing `%` for every lookup, as demonstrated in Pow2MontgomeryModCycleComputationBenchmarks.
                UInt128 rotationCount = hasCycle && exponent128 >= cycleLength ? exponent128 % cycleLength : exponent128;

                if (hasCycle && rotationCount == UInt128.Zero)
                {
                        return UInt128.One;
                }

                if (modulus < UInt128Numbers.Four || rotationCount < UInt128Numbers.SixtyFour)
                {
                        return Pow2ByDoubling(rotationCount, modulus);
                }

                if (ShouldUseWindowedPow2(modulus))
                {
                        return exponent128.Pow2MontgomeryModWindowedWithCycle(modulus, cycleLength, rotationCount);
                }

                return Pow2ByDoubling(rotationCount, modulus);
        }




	/// <summary>
	/// Computes 2^exponent mod modulus using a known cycle length.
	/// </summary>
        
        public static UInt128 PowModWithCycle(this UInt128 exponent, UInt128 modulus, UInt128 cycleLength)
        {
                UInt128 one = UInt128.One;
                UInt128 zero = UInt128.Zero;

                // EvenPerfectBitScanner drives this path with exponents derived from p, phi,
                // or divisor quotients, none of which are zero on the production scan.
                // if (exponent == zero)
                // {
                //         return one;
                // }

                // Modulus again comes from q = 2 * p * k + 1, so modulus == 1 is unreachable.
                // if (modulus == one)
                // {
                //         return zero;
                // }

                // Candidate divisors stay odd, therefore the power-of-two shortcut stays off.
                // if ((modulus & (modulus - one)) == zero)
                // {
                //         UInt128 masked = one << (int)(exponent & UInt128Numbers.OneHundredTwentySeven);
                //         return masked & (modulus - one);
                // }

                bool hasCycle = cycleLength != zero;
                // TODO: Swap this modulo with the shared UInt128 cycle remainder helper once available so CRT powmods reuse cached
                // reductions in the windowed ladder, avoiding the `%` cost highlighted in Pow2MontgomeryModCycleComputationBenchmarks.
                UInt128 rotationCount = hasCycle && exponent >= cycleLength ? exponent % cycleLength : exponent;

                if (hasCycle && rotationCount == zero)
                {
                        return one;
                }

                if (modulus < UInt128Numbers.Four || rotationCount < UInt128Numbers.SixtyFour)
                {
                        return Pow2ByDoubling(rotationCount, modulus);
                }

                if (ShouldUseWindowedPow2(modulus))
                {
                        return exponent.Pow2MontgomeryModWindowedWithCycle(modulus, cycleLength, rotationCount);
                }

                return Pow2ByDoubling(rotationCount, modulus);
        }




        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool ShouldUseWindowedPow2(UInt128 modulus) => modulus > Pow2WindowedModulusThreshold;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static UInt128 Pow2ByDoubling(ulong exponent, UInt128 modulus)
        {
                UInt128 result = UInt128.One;
                for (ulong i = 0; i < exponent; i++)
                {
                        result <<= 1;
                        while (result >= modulus)
                        {
                                result -= modulus;
                        }
                }

                return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static UInt128 Pow2ByDoubling(UInt128 exponent, UInt128 modulus)
        {
                UInt128 result = UInt128.One;
                UInt128 index = UInt128.Zero;

                while (index < exponent)
                {
                        result <<= 1;
                        while (result >= modulus)
                        {
                                result -= modulus;
                        }

                        index += UInt128.One;
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
