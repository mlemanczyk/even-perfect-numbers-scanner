using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public static class UIntExtensions
{
	private const ulong Mod3Mask = 0xAAAAAAABUL;
	public const ulong Mod5Mask = 0xCCCCCCCDUL;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static uint Mod3(uint value) => value - (uint)((value * Mod3Mask) >> 33) * 3U;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static uint Mod5(this uint value) => value - (uint)((value * Mod5Mask) >> 34) * 5U;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static uint Mod8(this uint value) => value & 7U;
}

