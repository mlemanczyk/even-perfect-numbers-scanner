using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public struct CycleRemainderStepper(ulong cycleLength)
{
    private readonly ulong _cycleLength = cycleLength;
    private ulong _previousPrime = 0UL;
    private ulong _currentRemainder = 0UL;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void Reset()
	{
		_previousPrime = 0UL;
		_currentRemainder = 0UL;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ulong Initialize(ulong prime)
    {
        ulong cycleLength = _cycleLength;
        ulong remainder = prime.ReduceCycleRemainder(cycleLength);

        _previousPrime = prime;
        _currentRemainder = remainder;
        return remainder;
    }

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ulong ComputeNext(ulong prime)
    {
        ulong cycleLength = _cycleLength;
        ulong delta = prime - _previousPrime;
        _previousPrime = prime;

        if (ulong.MaxValue - _currentRemainder < delta)
        {
            UInt128 extended = (UInt128)_currentRemainder + delta;
            _currentRemainder = ((ulong)(extended % cycleLength)).ReduceCycleRemainder(cycleLength);
            return _currentRemainder;
        }

        _currentRemainder += delta;

        if (_currentRemainder >= cycleLength)
        {
            _currentRemainder -= cycleLength;
            if (_currentRemainder >= cycleLength)
            {
                _currentRemainder = _currentRemainder.ReduceCycleRemainder(cycleLength);
            }
        }
        return _currentRemainder;
    }
}
