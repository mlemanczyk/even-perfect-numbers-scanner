using System;
using System.Numerics;

namespace PerfectNumbers.Core;

/// <summary>
/// Mutable helper for 128-bit arithmetic derived from the .NET runtime UInt128 implementation.
/// Portions adapted from https://github.com/dotnet/runtime/blob/v8.0.0/src/libraries/System.Private.CoreLib/src/System/UInt128.cs.
/// </summary>
internal sealed class MutableUInt128
{
    public ulong High { get; private set; }
    public ulong Low { get; private set; }

    public MutableUInt128()
    {
    }

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
        Set(value);
    }

    public void Set(UInt128 value)
    {
        High = (ulong)(value >> 64);
        Low = (ulong)value;
    }

    public void Set(ulong high, ulong low)
    {
        High = high;
        Low = low;
    }

    public UInt128 ToUInt128() => ((UInt128)High << 64) | Low;

    public void Add(ulong value)
    {
        ulong low = Low + value;
        if (low < value)
        {
            High++;
        }

        Low = low;
    }

    public void Add(in MutableUInt128 value)
    {
        ulong low = Low + value.Low;
        ulong carry = low < Low ? 1UL : 0UL;
        ulong high = High + value.High + carry;

        Low = low;
        High = high;
    }

    public void Subtract(ulong value)
    {
        ulong low = Low;
        ulong result = low - value;
        ulong borrow = result > low ? 1UL : 0UL;

        Low = result;
        High -= borrow;
    }

    public void Subtract(in MutableUInt128 value)
    {
        ulong low = Low;
        ulong result = low - value.Low;
        ulong borrow = low < value.Low ? 1UL : 0UL;

        Low = result;
        High = High - value.High - borrow;
    }

    public void Multiply(ulong value)
    {
        ulong high = High;
        ulong low = Low;

        ulong highContribution = Math.BigMul(low, value, out ulong newLow);
        Math.BigMul(high, value, out ulong shiftedHigh);
        ulong newHigh = shiftedHigh + highContribution;

        Low = newLow;
        High = newHigh;
    }

    public void MultiplyAdd(ulong multiplier, ulong addend)
    {
        Multiply(multiplier);
        Add(addend);
    }
}
