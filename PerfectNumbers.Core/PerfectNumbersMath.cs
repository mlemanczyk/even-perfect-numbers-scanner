using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public static class PerfectNumbersMath
{
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static uint FastRemainder3(uint value)
	{
		uint quotient = (uint)(((ulong)value * 0xAAAAAAABUL) >> 33);
		return value - quotient * 3U;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static uint FastRemainder5(uint value)
	{
		uint quotient = (uint)(((ulong)value * 0xCCCCCCCDUL) >> 34);
		return value - quotient * 5U;
	}

}