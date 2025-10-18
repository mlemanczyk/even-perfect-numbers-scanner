using System;
using System.Globalization;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public ref struct PooledValueStringBuilder
{
    private Span<char> _span;
    private char[]? _arrayFromPool;
    private int _pos;

    public PooledValueStringBuilder(Span<char> initialBuffer)
    {
        _span = initialBuffer;
        _arrayFromPool = null;
        _pos = 0;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Append(char value)
    {
        int position = _pos;
        if (position < _span.Length)
        {
            _span[position] = value;
            _pos = position + 1;
            return;
        }

        Grow(1);
        _span[_pos++] = value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Append(string? value)
    {
        if (!string.IsNullOrEmpty(value))
        {
            Append(value.AsSpan());
        }
    }

    public void Append(ReadOnlySpan<char> value)
    {
        if (value.Length == 0)
        {
            return;
        }

        int remaining = _span.Length - _pos;
        if (value.Length <= remaining)
        {
            value.CopyTo(_span[_pos..]);
            _pos += value.Length;
            return;
        }

        Grow(value.Length);
        value.CopyTo(_span[_pos..]);
        _pos += value.Length;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Append(int value)
    {
        AppendSpanFormattable(value);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Append(ulong value)
    {
        AppendSpanFormattable(value);
    }

    public ReadOnlySpan<char> AsSpan() => _span[.._pos];

    public override string ToString()
    {
        return new string(_span[.._pos]);
    }

    public void Dispose()
    {
        if (_arrayFromPool is { } array)
        {
            ThreadStaticPools.CharPool.Return(array);
            _arrayFromPool = null;
        }

        _span = Span<char>.Empty;
        _pos = 0;
    }

    private void AppendSpanFormattable<T>(T value) where T : ISpanFormattable
    {
        int charsWritten;
        while (!value.TryFormat(_span[_pos..], out charsWritten, default, CultureInfo.InvariantCulture))
        {
            Grow(Math.Max(4, _span.Length));
        }

        _pos += charsWritten;
    }

    private void Grow(int additionalCapacity)
    {
        int required = _pos + additionalCapacity;
        int currentLength = _span.Length;
        int newCapacity = currentLength == 0 ? Math.Max(4, required) : Math.Max(currentLength * 2, required);
        char[] newArray = ThreadStaticPools.CharPool.Rent(newCapacity);
        _span[.._pos].CopyTo(newArray.AsSpan());
        if (_arrayFromPool is { } toReturn)
        {
            ThreadStaticPools.CharPool.Return(toReturn);
        }

        _span = newArray;
        _arrayFromPool = newArray;
    }
}
