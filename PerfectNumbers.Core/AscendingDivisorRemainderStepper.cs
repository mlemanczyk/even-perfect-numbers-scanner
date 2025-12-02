using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public struct AscendingDivisorRemainderStepper
{
    private readonly ulong dividend;
    private ulong maxDivisor;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public AscendingDivisorRemainderStepper(ulong value)
    {
        dividend = value;
        ulong root = (ulong)Math.Sqrt(value);
        if ((UInt128)root * root < value)
        {
            root++;
        }

        maxDivisor = root;
    }

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public readonly bool ShouldContinue(ulong divisor) => divisor <= maxDivisor;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Divides(ulong divisor)
    {
        ulong quotient = dividend / divisor;
        ulong remainder = dividend - (quotient * divisor);
        if (quotient < maxDivisor)
        {
            maxDivisor = quotient;
        }

        return remainder == 0UL;
    }
}
