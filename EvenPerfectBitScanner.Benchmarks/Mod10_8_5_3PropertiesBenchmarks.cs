using System.Numerics;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
public class Mod10_8_5_3PropertiesBenchmarks
{
	public ulong Mod10R;
	public ulong Mod8R;
	public ulong Mod3R;
	public ulong Mod5R; 

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Mod8(UInt128 value) => (ulong)value & 7UL;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Mod3(UInt128 value)
    {
        // 2^64 ≡ 1 (mod 3)
        ulong high = (ulong)(value >> 64);
        ulong low = (ulong)value;
        ulong rem = (low % 3UL) + (high % 3UL);
        return rem >= 3UL ? rem - 3UL : rem;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Mod5(UInt128 value)
    {
        // 2^64 ≡ 1 (mod 5)
        ulong high = (ulong)(value >> 64);
        ulong low = (ulong)value;
        ulong rem = (low % 5UL) + (high % 5UL);
        return rem >= 5UL ? rem - 5UL : rem;
    }

    [Params(3UL, 8191UL, 131071UL, 2147483647UL)]
    public ulong Value { get; set; }

    [Benchmark(Baseline = true)]
    public (ulong Mod10, ulong Mod8, ulong Mod5, ulong Mod3) LegacyModulo()
    {
		UInt128 value = Value;
        Mod10R = value.Mod10();
        Mod8R  = Mod8(value);
        Mod5R  = Mod5(value);
        Mod3R  = Mod3(value);
		return (Mod10R, Mod8R, Mod5R, Mod3R);
    }

    [Benchmark]
    public (ulong Mod10, ulong Mod8, ulong Mod3, ulong Mod5) ModMethodModulo()
    {
		Value.Mod10_8_5_3(out var mod10, out ulong mod8, out ulong mod5, out ulong mod3);
		Mod10R = mod10;
		Mod8R = mod8;
		Mod5R = mod5;
		Mod3R = mod3;

        return (Mod10R, Mod8R, Mod3R, Mod5R);
    }
}

