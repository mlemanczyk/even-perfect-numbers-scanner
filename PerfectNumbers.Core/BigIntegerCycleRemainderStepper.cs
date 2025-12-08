using System.Numerics;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

/// <summary>
/// Maintains rolling remainders of an increasing sequence modulo a BigInteger cycle length.
/// Mirrors <see cref="CycleRemainderStepper"/> for values that exceed 64 bits.
/// </summary>
[method: MethodImpl(MethodImplOptions.AggressiveInlining)]
public struct BigIntegerCycleRemainderStepper(BigInteger cycleLength)
{
    private readonly BigInteger _cycleLength = cycleLength;
    private BigInteger _previousValue = BigInteger.Zero;
    private BigInteger _currentRemainder = BigInteger.Zero;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
    public BigInteger Initialize(BigInteger value)
    {
        _previousValue = value;
        _currentRemainder = value.ReduceCycleRemainder(_cycleLength);
        return _currentRemainder;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public BigInteger ComputeNext(BigInteger value)
    {
        BigInteger delta = value - _previousValue;
        _previousValue = value;
        _currentRemainder = (_currentRemainder + delta).ReduceCycleRemainder(_cycleLength);

        return _currentRemainder;
    }
}
