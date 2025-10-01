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
            // TODO: Route these state resets through the ProcessEightBitWindows helper once the scalar
            // Pow2MontgomeryMod implementation adopts it so delta stepping inherits the benchmarked
            // 2× gains for large exponents.
            _currentMontgomery = exponent.Pow2MontgomeryModMontgomery(_divisor);
            _previousExponent = exponent;
            _hasState = true;
            return ReduceCurrent();
        }

        ulong delta = exponent - _previousExponent;
        // TODO: Replace this per-delta powmod with the upcoming windowed ladder so incremental
        // updates stop paying the single-bit cost highlighted in GpuPow2ModBenchmarks.
        ulong multiplier = delta.Pow2MontgomeryModMontgomery(_divisor);
        _currentMontgomery = _currentMontgomery.MontgomeryMultiply(multiplier, _modulus, _nPrime);
        _previousExponent = exponent;
        return ReduceCurrent();
    }

    public bool ComputeNextIsUnity(ulong exponent)
    {
        if (!IsValidModulus)
        {
            _hasState = false;
            return false;
        }

        if (!_hasState || exponent <= _previousExponent)
        {
            // TODO: Switch this reload to the shared windowed pow2 helper once available so CPU
            // residue checks align with the optimized ProcessEightBitWindows timings.
            _currentMontgomery = exponent.Pow2MontgomeryModMontgomery(_divisor);
            _previousExponent = exponent;
            _hasState = true;
            return _currentMontgomery == _montgomeryOne;
        }

        ulong delta = exponent - _previousExponent;
        // TODO: Use the windowed delta pow2 helper here as well to avoid the single-bit ladder that
        // currently lags behind the benchmarked implementation for huge divisor cycles.
        ulong multiplier = delta.Pow2MontgomeryModMontgomery(_divisor);
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
            // TODO: Replace this fallback path with the upcoming ProcessEightBitWindows helper so
            // fresh Montgomery states also benefit from the faster pow2 ladder measured on CPUs.
            _currentMontgomery = exponent.Pow2MontgomeryModMontgomery(_divisor);
            _previousExponent = exponent;
            _hasState = true;
            isUnity = _currentMontgomery == _montgomeryOne;
            return true;
        }

        _currentMontgomery = _currentMontgomery.MontgomeryMultiply(montgomeryDelta, _modulus, _nPrime);
        _previousExponent = exponent;
        isUnity = _currentMontgomery == _montgomeryOne;
        return true;
    }

    private ulong ReduceCurrent() => _currentMontgomery.MontgomeryMultiply(1UL, _modulus, _nPrime);
}
