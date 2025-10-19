using System;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public static partial class ULongExtensions
{
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Pow2MontgomeryModWindowedCpu(this ulong exponent, in MontgomeryDivisorData divisor, bool keepMontgomery)
        {
                ulong modulus = divisor.Modulus;
                // This should never happen in production code. If it does, we need to fix it.
                // if (exponent == 0UL)
                // {
                //     return keepMontgomery ? divisor.MontgomeryOne : 1UL % modulus;
                // }

                // There is no point in having extra branch, because in real use-cases it'll barely ever hit.
                // Out of all 3_358_944_545 Pow2MontgomeryModWindowedCpu calls, we hit Pow2MontgomeryModSingleBit only 772 times.
                // if (exponent <= Pow2WindowFallbackThreshold)
                // {
                //     return Pow2MontgomeryModSingleBit(exponent, divisor, keepMontgomery);
                // }

                int bitLength = GetPortableBitLength(exponent);
                int windowSize = GetWindowSize(bitLength);
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
                                result = result.MontgomeryMultiply(result, modulus, nPrime);
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
                                result = result.MontgomeryMultiply(result, modulus, nPrime);
                        }

                        ulong mask = (1UL << windowLength) - 1UL;
                        ulong windowValue = (exponent >> windowStart) & mask;
                        int tableIndex = (int)((windowValue - 1UL) >> 1);
                        ulong multiplier = oddPowers[tableIndex];
                        result = result.MontgomeryMultiply(multiplier, modulus, nPrime);

                        index = windowStart - 1;
                }

                ThreadStaticPools.UlongPool.Return(oddPowersArray);

                if (!keepMontgomery)
                {
                        result = result.MontgomeryMultiply(1UL, modulus, nPrime);
                }

                return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
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
                        oddPowers[i] = previous.MontgomeryMultiply(square, modulus, nPrime);
                }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Pow2MontgomeryModWithCycleCpu(this ulong exponent, ulong cycleLength, in MontgomeryDivisorData divisor)
        {
                ulong rotationCount = exponent % cycleLength;
                return Pow2MontgomeryModWindowedCpu(rotationCount, divisor, keepMontgomery: false);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Pow2MontgomeryModFromCycleRemainderCpu(this ulong reducedExponent, in MontgomeryDivisorData divisor)
        {
                return Pow2MontgomeryModWindowedCpu(reducedExponent, divisor, keepMontgomery: false);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
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
}
