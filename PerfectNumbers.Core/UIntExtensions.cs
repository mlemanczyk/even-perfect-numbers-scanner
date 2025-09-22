using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public static class UIntExtensions
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Mod3(this uint value) => value % 3U;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Mod5(this uint value) => value % 5U;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Mod6(this uint value) => value % 6U;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Mod7(this uint value) => value % 7U;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Mod8(this uint value) => value & 7U;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Mod10(this uint value) => value % 10U;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Mod11(this uint value) => value % 11U;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Mod128(this uint value) => value & 127U;
}

