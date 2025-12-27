using System.Numerics;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

/// <summary>
/// Maintains rolling remainders of an increasing sequence modulo a BigInteger cycle length.
/// Mirrors <see cref="CycleRemainderStepper"/> for values that exceed 64 bits.
/// </summary>
[method: MethodImpl(MethodImplOptions.AggressiveInlining)]
public struct BigIntegerCycleRemainderStepper(byte cycleLength)
{
    private readonly BigInteger _cycleLength = (BigInteger)cycleLength;
    private BigInteger _previousValue = BigInteger.Zero;
    private BigInteger _currentRemainder = 0;

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public byte Initialize(BigInteger value)
    {
        _previousValue = value;
        _currentRemainder = value % _cycleLength;
        return (byte)_currentRemainder;
    }

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public bool NextIsNotDivisible(BigInteger value)
    {
		BigInteger delta = value - _previousValue;
        _previousValue = value;
		delta = (_currentRemainder + delta) % _cycleLength;
        _currentRemainder = delta;
        return !delta.IsZero;
    }

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public byte ComputeNext(BigInteger value)
    {
		BigInteger delta = value - _previousValue;
        _previousValue = value;
		delta = (_currentRemainder + delta) % _cycleLength;
        _currentRemainder = delta;

        return (byte)delta;
    }
}
