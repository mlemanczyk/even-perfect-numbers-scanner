using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

/// <summary>
/// Mutable helper for 128-bit arithmetic derived from the .NET runtime UInt128 implementation.
/// Portions adapted from https://github.com/dotnet/runtime/blob/v8.0.0/src/libraries/System.Private.CoreLib/src/System/UInt128.cs.
/// </summary>
public struct MutableUInt128
{
    public ulong High;
    public ulong Low;

    public MutableUInt128(ulong value)
    {
        Low = value;
    }

    public MutableUInt128(ulong high, ulong low)
    {
        High = high;
        Low = low;
    }

    public MutableUInt128(UInt128 value)
    {
        High = (ulong)(value >> 64);
        Low = (ulong)value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Set(ulong value)
    {
        High = 0UL;
        Low = value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Set(UInt128 value)
    {
        High = (ulong)(value >> 64);
        Low = (ulong)value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Set(ulong high, ulong low)
    {
        High = high;
        Low = low;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly UInt128 ToUInt128() => ((UInt128)High << 64) | Low;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Add(ulong value)
    {
        ulong low = Low + value;
        if (low < value)
        {
            High++;
        }

        Low = low;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Add(in MutableUInt128 value)
    {
        ulong temp = Low + value.Low;
        Low = temp;
        temp = temp < Low ? 1UL : 0UL;
		temp += High;
		temp += value.High;
        High = temp;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Sub(ulong value)
    {
        ulong low = Low;
        ulong result = low - value;
        ulong borrow = result > low ? 1UL : 0UL;

        Low = result;
        High -= borrow;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Sub(in MutableUInt128 value)
    {
        ulong low = Low;
        ulong result = low - value.Low;
        ulong borrow = low < value.Low ? 1UL : 0UL;

        Low = result;
        High = High - value.High - borrow;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Mul(ulong value)
    {
        ulong currentHigh = High;
        ulong currentLow = Low;

        ulong carry = Mul64(currentLow, value, out ulong newLow);
        Mul64(currentHigh, value, out ulong shiftedHigh);
        ulong newHigh = shiftedHigh + carry;

        Low = newLow;
        High = newHigh;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Mul64(ulong left, ulong right, out ulong low)
    {
        const ulong Mask32 = 0xFFFF_FFFFUL;

        ulong leftLow = left & Mask32;
        ulong leftHigh = left >> 32;
        ulong rightLow = right & Mask32;
        ulong rightHigh = right >> 32;

        ulong lowLow = leftLow * rightLow;
        ulong lowHigh = leftLow * rightHigh;
        ulong highLow = leftHigh * rightLow;
        ulong highHigh = leftHigh * rightHigh;

        ulong carry = (lowLow >> 32) + (lowHigh & Mask32) + (highLow & Mask32);
        ulong high = highHigh + (lowHigh >> 32) + (highLow >> 32) + (carry >> 32);

        low = ((carry & Mask32) << 32) | (lowLow & Mask32);
        return high;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void MulAdd(ulong multiplier, ulong addend)
    {
        const ulong Mask32 = 0xFFFF_FFFFUL;

        ulong multiplierLow = multiplier & Mask32;
        ulong multiplierHigh = multiplier >> 32;

        ulong product11 = Low;
        ulong product01 = product11 & Mask32;
        product11 >>= 32;

        ulong product00 = product01 * multiplierLow;
        product01 *= multiplierHigh;
        ulong product10 = product11 * multiplierLow;
        product11 *= multiplierHigh;

        ulong high = (product00 >> 32) + (product01 & Mask32) + (product10 & Mask32);
        product11 = product11 + (product01 >> 32) + (product10 >> 32) + (high >> 32);
        product10 = ((high & Mask32) << 32) | (product00 & Mask32);

        high = High;
        product00 = high & Mask32;
        high >>= 32;

        product01 = product00 * multiplierLow;
        product00 *= multiplierHigh;
        high *= multiplierLow;

        high = (product01 >> 32) + (product00 & Mask32) + (high & Mask32);
		product01 = ((high & Mask32) << 32) | (product01 & Mask32);

		// Calculate low. We're assigning addend to local variable for a tiny bit better performance.
		product00 = addend;
		high = product10 + product00;
        Low = high;

		// Calculate high including carry
		high = high < product00 ? 1UL : 0UL;
		high += product11 + product01;
		
        // Discard overflow above 128 bits to match Multiply's truncation semantics.
        High = high;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly ulong Mod(ulong modulus)
    {
        if (modulus == 0UL)
        {
            throw new DivideByZeroException();
        }

        if (High == 0UL)
        {
            return Low % modulus;
        }

        MutableUInt128 remainder = default;
        int highBitLength = 64 - BitOperations.LeadingZeroCount(High);
        if (highBitLength > 0)
        {
            ProcessWord(ref remainder, High, highBitLength, modulus);
        }

        ProcessWord(ref remainder, Low, 64, modulus);
        return remainder.Low;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ProcessWord(ref MutableUInt128 remainder, ulong word, int bitCount, ulong modulus)
    {
        for (int bit = bitCount - 1; bit >= 0; bit--)
        {
            remainder.ShiftLeft();
            if (((word >> bit) & 1UL) != 0)
            {
                remainder.Add(1UL);
            }

            remainder.Reduce(modulus);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ShiftLeft()
    {
        High = (High << 1) | (Low >> 63);
        Low <<= 1;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Reduce(ulong modulus)
    {
        if (High != 0UL || Low >= modulus)
        {
            Sub(modulus);
            if (High != 0UL || Low >= modulus)
            {
                Sub(modulus);
            }
        }
    }
}
