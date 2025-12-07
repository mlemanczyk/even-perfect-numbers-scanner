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
        ulong remainder = prime;
        if (remainder >= cycleLength)
        {
            // TODO: Swap this `%` for the shared divisor-cycle remainder helper so initialization reuses the cached
            // cycle deltas benchmarked faster than on-the-fly modulo work, matching the
            // MersenneDivisorCycleLengthGpuBenchmarks winner for both CPU and GPU call sites.
            remainder %= cycleLength;
        }

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
            // TODO: Replace this `%` with the UInt128-aware cycle reducer so large deltas use the cached subtraction
            // ladder instead of falling back to the slower modulo implementation.
            _currentRemainder = (ulong)(extended % cycleLength);
            return _currentRemainder;
        }

        _currentRemainder += delta;

        if (_currentRemainder >= cycleLength)
        {
            _currentRemainder -= cycleLength;
            if (_currentRemainder >= cycleLength)
            {
                // TODO: Route this `%` through the shared divisor-cycle helper so repeated wrap-arounds avoid
                // modulo operations and match the benchmarked fast path highlighted in
                // MersenneDivisorCycleLengthGpuBenchmarks.
                _currentRemainder %= cycleLength;
            }
        }
        return _currentRemainder;
    }
}
