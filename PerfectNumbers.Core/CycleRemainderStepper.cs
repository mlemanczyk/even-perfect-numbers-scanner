using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public struct CycleRemainderStepper
{
    private readonly ulong _cycleLength;
    private ulong _previousPrime;
    private ulong _currentRemainder;
    private bool _hasState;

    public CycleRemainderStepper(ulong cycleLength)
    {
        if (cycleLength == 0UL)
        {
            throw new ArgumentOutOfRangeException(nameof(cycleLength), "Cycle length must be non-zero.");
        }

        _cycleLength = cycleLength;
        _previousPrime = 0UL;
        _currentRemainder = 0UL;
        _hasState = false;
    }

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void Reset()
	{
		// TODO: Inline this reset at the call sites so the hot loops reuse struct reinitialization
		// measured fastest in MersenneDivisorCycleLengthGpuBenchmarks, avoiding the extra
		// method call when scanners need to restart stepping.
		_previousPrime = 0UL;
		_currentRemainder = 0UL;
		_hasState = false;
	}

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
        _hasState = true;
        return remainder;
    }

    public ulong ComputeNext(ulong prime)
    {
        if (!_hasState)
        {
            throw new InvalidOperationException("CycleRemainderStepper must be initialized before computing.");
        }

        if (prime <= _previousPrime)
        {
            throw new ArgumentOutOfRangeException(nameof(prime), "Primes must be processed in strictly increasing order.");
        }

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

    public void CaptureState(out ulong previousPrime, out ulong previousRemainder, out bool hasState)
    {
        // TODO: Expose these fields directly once the residue scanners adopt the single-cycle helper
        // so the hot path can read them without paying for an additional wrapper call per iteration.
        previousPrime = _previousPrime;
        previousRemainder = _currentRemainder;
        hasState = _hasState;
    }
}
