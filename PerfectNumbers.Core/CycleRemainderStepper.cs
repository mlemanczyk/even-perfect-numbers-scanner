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
        _cycleLength = cycleLength;
        _previousPrime = 0UL;
        _currentRemainder = 0UL;
        _hasState = false;
    }

    public bool HasCycle => _cycleLength != 0UL;

    public void Reset()
    {
        _previousPrime = 0UL;
        _currentRemainder = 0UL;
        _hasState = false;
    }

    public void InitializeState(ulong previousPrime, ulong previousRemainder, bool hasState)
    {
        _previousPrime = previousPrime;
        _currentRemainder = previousRemainder;
        _hasState = hasState;
    }

    public ulong ComputeNext(ulong prime)
    {
        if (_cycleLength == 0UL)
        {
            _previousPrime = prime;
            _currentRemainder = 0UL;
            _hasState = true;
            return 0UL;
        }

        if (!_hasState || prime <= _previousPrime)
        {
            _currentRemainder = prime % _cycleLength;
            _hasState = true;
        }
        else
        {
            ulong delta = prime - _previousPrime;
            if (ulong.MaxValue - _currentRemainder < delta)
            {
                UInt128 extended = (UInt128)_currentRemainder + delta;
                _currentRemainder = (ulong)(extended % _cycleLength);
            }
            else
            {
                _currentRemainder += delta;

                if (_currentRemainder >= _cycleLength)
                {
                    _currentRemainder -= _cycleLength;
                    if (_currentRemainder >= _cycleLength)
                    {
                        _currentRemainder %= _cycleLength;
                    }
                }
            }
        }

        _previousPrime = prime;
        return _currentRemainder;
    }

    public void CaptureState(out ulong previousPrime, out ulong previousRemainder, out bool hasState)
    {
        previousPrime = _previousPrime;
        previousRemainder = _currentRemainder;
        hasState = _hasState;
    }
}
