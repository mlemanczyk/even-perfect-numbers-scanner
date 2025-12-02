using System.Buffers;
using System.Buffers.Text;
using System.Text;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner;

internal static class Utf8CliParser
{
    private const int StackAllocThreshold = 128;

    public static ulong ParseUInt64(ReadOnlySpan<char> value)
    {
        if (!TryParseUInt64(value, out ulong result))
        {
            ThrowFormatException(value);
        }

        return result;
    }

    public static int ParseInt32(ReadOnlySpan<char> value)
    {
        if (!TryParseInt32(value, out int result))
        {
            ThrowFormatException(value);
        }

        return result;
    }

    public static uint ParseUInt32(ReadOnlySpan<char> value)
    {
        if (!TryParseUInt32(value, out uint result))
        {
            ThrowFormatException(value);
        }

        return result;
    }

    public static bool TryParseUInt64(ReadOnlySpan<char> value, out ulong result)
    {
        return TryParse(value, static (ReadOnlySpan<byte> source, out ulong parsed, out int consumed) => Utf8Parser.TryParse(source, out parsed, out consumed), out result);
    }

    public static bool TryParseUInt32(ReadOnlySpan<char> value, out uint result)
    {
        return TryParse(value, static (ReadOnlySpan<byte> source, out uint parsed, out int consumed) => Utf8Parser.TryParse(source, out parsed, out consumed), out result);
    }

    public static bool TryParseInt32(ReadOnlySpan<char> value, out int result)
    {
        return TryParse(value, static (ReadOnlySpan<byte> source, out int parsed, out int consumed) => Utf8Parser.TryParse(source, out parsed, out consumed), out result);
    }

    public static bool TryParseDouble(ReadOnlySpan<char> value, out double result)
    {
        return TryParse(value, static (ReadOnlySpan<byte> source, out double parsed, out int consumed) => Utf8Parser.TryParse(source, out parsed, out consumed), out result);
    }

    public static bool TryParseBoolean(ReadOnlySpan<char> value, out bool result)
    {
        value = value.Trim();
        if (value.Equals("true", StringComparison.OrdinalIgnoreCase))
        {
            result = true;
            return true;
        }

        if (value.Equals("false", StringComparison.OrdinalIgnoreCase))
        {
            result = false;
            return true;
        }

        if (value.Equals("1", StringComparison.Ordinal))
        {
            result = true;
            return true;
        }

        if (value.Equals("0", StringComparison.Ordinal))
        {
            result = false;
            return true;
        }

        return TryParse(value, static (ReadOnlySpan<byte> source, out bool parsed, out int consumed) => Utf8Parser.TryParse(source, out parsed, out consumed), out result);
    }

    private static bool TryParse<T>(ReadOnlySpan<char> value, Utf8ParserDelegate<T> parser, out T result)
        where T : struct
    {
        value = value.Trim();
        if (value.IsEmpty)
        {
            result = default;
            return false;
        }

        Span<byte> stackBuffer = stackalloc byte[StackAllocThreshold];
        if (value.Length <= stackBuffer.Length)
        {
            int bytesWritten = Encoding.UTF8.GetBytes(value, stackBuffer);
            ReadOnlySpan<byte> utf8 = stackBuffer[..bytesWritten];
            if (!parser(utf8, out result, out int consumed) || consumed != bytesWritten)
            {
                result = default;
                return false;
            }

            return true;
        }

        ArrayPool<byte> pool = ThreadStaticPools.BytePool;
        byte[] rented = pool.Rent(value.Length);
        int rentedBytesWritten = Encoding.UTF8.GetBytes(value, rented);
        ReadOnlySpan<byte> rentedUtf8 = rented.AsSpan(0, rentedBytesWritten);
        bool parsed = parser(rentedUtf8, out result, out int rentedConsumed) && rentedConsumed == rentedBytesWritten;
        pool.Return(rented);
        if (!parsed)
        {
            result = default;
            return false;
        }

        return true;
    }

    private static void ThrowFormatException(ReadOnlySpan<char> value)
    {
        throw new FormatException($"Invalid value '{value.ToString()}'.");
    }

    private delegate bool Utf8ParserDelegate<T>(ReadOnlySpan<byte> source, out T value, out int bytesConsumed)
        where T : struct;
}
