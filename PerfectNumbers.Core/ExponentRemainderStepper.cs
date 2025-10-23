namespace PerfectNumbers.Core;

internal struct ExponentRemainderStepper
{
    private readonly MontgomeryDivisorData _divisor;
    private readonly ulong _modulus;
    private readonly ulong _nPrime;
    private readonly ulong _montgomeryOne;
    private ulong _previousExponent;
    private ulong _currentMontgomery;
    private bool _hasState;

    public ExponentRemainderStepper(in MontgomeryDivisorData divisor)
    {
        _divisor = divisor;
        _modulus = divisor.Modulus;
        _nPrime = divisor.NPrime;
        _montgomeryOne = divisor.MontgomeryOne;
        _previousExponent = 0UL;
        _currentMontgomery = divisor.MontgomeryOne;
        _hasState = false;
    }

    public bool IsValidModulus => _modulus > 1UL && (_modulus & 1UL) != 0UL;

    public bool HasState => _hasState;

    public ulong PreviousExponent => _previousExponent;

    public bool MatchesDivisor(in MontgomeryDivisorData divisor)
    {
        return _modulus == divisor.Modulus && _nPrime == divisor.NPrime && _montgomeryOne == divisor.MontgomeryOne;
    }

    public void Reset()
    {
        _previousExponent = 0UL;
        _currentMontgomery = _montgomeryOne;
        _hasState = false;
    }

    public ulong ComputeNext(ulong exponent)
    {
        if (!IsValidModulus)
        {
            _hasState = false;
            return 0UL;
        }

        if (!_hasState || exponent <= _previousExponent)
        {
            _currentMontgomery = exponent.Pow2MontgomeryModWindowedCpu(_divisor, keepMontgomery: true);
            _previousExponent = exponent;
            _hasState = true;
            return ReduceCurrent();
        }

        ulong delta = exponent - _previousExponent;
        // TODO: Once divisor cycle lengths are mandatory, pull the delta multiplier from the
        // single-block divisor-cycle snapshot so we can skip the powmod entirely and reuse the
        // cached Montgomery residue ladder highlighted in MersenneDivisorCycleLengthGpuBenchmarks.
        ulong multiplier = delta.Pow2MontgomeryModWindowedCpu(_divisor, keepMontgomery: true);
        _currentMontgomery = _currentMontgomery.MontgomeryMultiply(multiplier, _modulus, _nPrime);
        _previousExponent = exponent;
        return ReduceCurrent();
    }

    public bool ComputeNextIsUnity(ulong exponent)
    {
        if (!IsValidModulus)
        {
			throw new InvalidOperationException($"Modulus is invalid for exponent {exponent}");
            // _hasState = false;
            // return false;
        }

        if (!_hasState || exponent <= _previousExponent)
        {
            _currentMontgomery = exponent.Pow2MontgomeryModWindowedCpu(_divisor, keepMontgomery: true);
            _previousExponent = exponent;
            _hasState = true;
            return _currentMontgomery == _montgomeryOne;
        }

        ulong delta = exponent - _previousExponent;
        // TODO: Reuse the divisor-cycle derived Montgomery delta once the cache exposes single-cycle
        // lookups so this branch also avoids recomputing powmods when the snapshot lacks the divisor.
        ulong multiplier = delta.Pow2MontgomeryModWindowedCpu(_divisor, keepMontgomery: true);
        _currentMontgomery = _currentMontgomery.MontgomeryMultiply(multiplier, _modulus, _nPrime);
        _previousExponent = exponent;
        return _currentMontgomery == _montgomeryOne;
    }

    public bool TryInitializeFromMontgomeryResult(ulong exponent, ulong montgomeryResult, out bool isUnity)
    {
        if (!IsValidModulus)
        {
            _hasState = false;
            isUnity = false;
            return false;
        }

        _currentMontgomery = montgomeryResult;
        _previousExponent = exponent;
        _hasState = true;
        isUnity = _currentMontgomery == _montgomeryOne;
        return true;
    }

    public bool TryAdvanceWithMontgomeryDelta(ulong exponent, ulong montgomeryDelta, out bool isUnity)
    {
        if (!IsValidModulus)
        {
            _hasState = false;
            isUnity = false;
            return false;
        }

        if (!_hasState || exponent <= _previousExponent)
        {
            _currentMontgomery = exponent.Pow2MontgomeryModWindowedCpu(_divisor, keepMontgomery: true);
            _previousExponent = exponent;
            _hasState = true;
            isUnity = _currentMontgomery == _montgomeryOne;
            return true;
        }

        // TODO: Once the divisor-cycle cache exposes a direct Montgomery delta, multiply it here instead
        // of relying on the caller-provided delta so incremental scans remain in sync with the snapshot
        // without computing additional cycles or mutating cache state.
        _currentMontgomery = _currentMontgomery.MontgomeryMultiply(montgomeryDelta, _modulus, _nPrime);
        _previousExponent = exponent;
        isUnity = _currentMontgomery == _montgomeryOne;
        return true;
    }

    private ulong ReduceCurrent() => _currentMontgomery.MontgomeryMultiply(1UL, _modulus, _nPrime);
}
