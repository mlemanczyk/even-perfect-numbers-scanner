using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core.Gpu;

public readonly struct ReadOnlyGpuUInt128 : IEquatable<ReadOnlyGpuUInt128>
{
    public readonly ulong High;
    public readonly ulong Low;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlyGpuUInt128(ulong high, ulong low)
    {
        High = high;
        Low = low;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlyGpuUInt128(ulong low)
        : this(0UL, low)
    {
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlyGpuUInt128(UInt128 value)
    {
        High = (ulong)(value >> 64);
        Low = (ulong)value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlyGpuUInt128(GpuUInt128 value)
    {
        High = value.High;
        Low = value.Low;
    }

    public bool IsZero
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => High == 0UL && Low == 0UL;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int GetBitLength()
    {
        if (High != 0UL)
        {
            return 64 + (64 - BitOperations.LeadingZeroCount(High));
        }

        if (Low == 0UL)
        {
            return 0;
        }

        return 64 - BitOperations.LeadingZeroCount(Low);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GpuUInt128 ToMutable() => new(High, Low);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Equals(ReadOnlyGpuUInt128 other) => High == other.High && Low == other.Low;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override bool Equals(object? obj) => obj is ReadOnlyGpuUInt128 other && Equals(other);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override int GetHashCode() => HashCode.Combine(High, Low);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator ReadOnlyGpuUInt128(ulong value) => new(value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator ReadOnlyGpuUInt128(UInt128 value) => new(value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator ReadOnlyGpuUInt128(GpuUInt128 value) => new(value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator UInt128(ReadOnlyGpuUInt128 value) => ((UInt128)value.High << 64) | value.Low;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlyGpuUInt128 WithHigh(ulong high) => new(high, Low);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlyGpuUInt128 WithLow(ulong low) => new(High, low);
}
