using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public struct AscendingDivisorRemainderStepper
{
    private readonly ulong _dividend;
    private ulong _maxDivisor;

    [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
    public AscendingDivisorRemainderStepper(ulong value)
    {
        _dividend = value;
        ulong root = (ulong)Math.Sqrt(value);
        if ((UInt128)root * root < value)
        {
            root++;
        }

        _maxDivisor = root;
    }

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public readonly bool ShouldContinue(ulong divisor) => divisor <= _maxDivisor;

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public bool Divides(ulong divisor)
    {
		ulong remainder = _dividend;
		ulong quotient = remainder / divisor;
        remainder -= quotient * divisor;
        if (quotient < _maxDivisor)
        {
            _maxDivisor = quotient;
        }

        return remainder == 0UL;
    }
}
