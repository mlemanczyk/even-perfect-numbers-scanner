using System;
using System.Numerics;

namespace PerfectNumbers.Core;

internal struct CycleRemainderStepper
{
    private readonly ulong _cycleLength;
    private readonly UInt128 _cycleLength128;
    private readonly int _cycleLengthLog2;
    private readonly bool _cycleIsPowerOfTwo;
    private readonly ulong _cycleMask;
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
        _cycleLength128 = cycleLength;
        _cycleIsPowerOfTwo = (cycleLength & (cycleLength - 1UL)) == 0UL;
        _cycleMask = cycleLength - 1UL;
        _cycleLengthLog2 = BitOperations.Log2(cycleLength);
        _previousPrime = 0UL;
        _currentRemainder = 0UL;
        _hasState = false;
    }

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

        ulong delta = prime - _previousPrime;
        _previousPrime = prime;

        UInt128 extended = (UInt128)_currentRemainder + delta;
        _currentRemainder = Reduce(extended);
        return _currentRemainder;
    }

    private ulong Reduce(UInt128 value)
    {
        if (value < _cycleLength128)
        {
            return (ulong)value;
        }

        if (_cycleIsPowerOfTwo)
        {
            return (ulong)value & _cycleMask;
        }

        UInt128 remainder = value;
        UInt128 modulus = _cycleLength128;
        int modulusLog2 = _cycleLengthLog2;

        while (remainder >= modulus)
        {
            int remainderLog2 = Log2(remainder);
            int shift = remainderLog2 - modulusLog2;
            if (shift <= 0)
            {
                remainder -= modulus;
                continue;
            }

            UInt128 scaled = modulus << shift;
            if (scaled > remainder)
            {
                shift--;
                if (shift <= 0)
                {
                    remainder -= modulus;
                    continue;
                }

                scaled = modulus << shift;
            }

            remainder -= scaled;
        }

        return (ulong)remainder;
    }

    private static int Log2(UInt128 value)
    {
        ulong high = (ulong)(value >> 64);
        if (high != 0UL)
        {
            return 63 - BitOperations.LeadingZeroCount(high) + 64;
        }

        ulong low = (ulong)value;
        return 63 - BitOperations.LeadingZeroCount(low);
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
