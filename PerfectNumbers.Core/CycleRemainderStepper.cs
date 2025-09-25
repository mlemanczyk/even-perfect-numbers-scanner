using System;
using System.Numerics;

namespace PerfectNumbers.Core;

internal struct CycleRemainderStepper
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

    public void Reset()
    {
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
            _currentRemainder = (ulong)(extended % cycleLength);
            return _currentRemainder;
        }

        _currentRemainder += delta;

        if (_currentRemainder >= cycleLength)
        {
            _currentRemainder -= cycleLength;
            if (_currentRemainder >= cycleLength)
            {
                _currentRemainder %= cycleLength;
            }
        }
        return _currentRemainder;
    }

    public void CaptureState(out ulong previousPrime, out ulong previousRemainder, out bool hasState)
    {
        previousPrime = _previousPrime;
        previousRemainder = _currentRemainder;
        hasState = _hasState;
    }
}
