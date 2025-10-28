namespace PerfectNumbers.Core;

/// <summary>
/// Tracks successive Montgomery residues for a single divisor while scanning exponent candidates on the CPU path.
/// Callers are responsible for seeding the stepper once per divisor and then advancing exponents in ascending order.
/// </summary>
internal struct ExponentRemainderStepperCpu
{
    private readonly MontgomeryDivisorData _divisor;
    private readonly ulong _modulus;
    private readonly ulong _nPrime;
    private readonly ulong _montgomeryOne;
    public ulong PreviousExponent;
    private ulong _currentMontgomery;

    public ExponentRemainderStepperCpu(in MontgomeryDivisorData divisor)
    {
        _divisor = divisor;
        _modulus = divisor.Modulus;
        _nPrime = divisor.NPrime;
        _montgomeryOne = divisor.MontgomeryOne;
        PreviousExponent = 0UL;
        _currentMontgomery = divisor.MontgomeryOne;
    }

    public bool MatchesDivisor(in MontgomeryDivisorData divisor)
    {
        return _modulus == divisor.Modulus && _nPrime == divisor.NPrime && _montgomeryOne == divisor.MontgomeryOne;
    }

    /// <summary>
    /// Clears the cached Montgomery residue so the stepper can be reused for another divisor.
    /// Callers must reinitialize the stepper before consuming any residues after a reset.
    /// </summary>
    public void Reset()
    {
        PreviousExponent = 0UL;
        _currentMontgomery = _montgomeryOne;
    }

    /// <summary>
    /// Initializes the stepper for CPU calculations using the first exponent in a strictly ascending sequence.
    /// The returned value is the canonical (non-Montgomery) residue corresponding to the provided exponent.
    /// </summary>
    /// <param name="exponent">First exponent evaluated for the current divisor. The caller must reuse the same divisor metadata.</param>
    /// <returns>The canonical residue produced by <c>2^exponent mod divisor</c>.</returns>
    public ulong InitializeCpu(ulong exponent)
    {
        InitializeCpuState(exponent);
        return ReduceCurrent();
    }

    /// <summary>
    /// Initializes the CPU stepping state and reports whether the seed exponent already yields unity.
    /// </summary>
    /// <param name="exponent">First exponent evaluated for the current divisor. Must match the divisor passed to the constructor.</param>
    /// <returns><see langword="true"/> when the seed exponent produces a Montgomery residue equal to one.</returns>
    public bool InitializeCpuIsUnity(ulong exponent)
    {
        InitializeCpuState(exponent);
        return _currentMontgomery == _montgomeryOne;
    }

    /// <summary>
    /// Advances the CPU stepping state to the next exponent in a strictly ascending sequence that shares the same divisor.
    /// Callers must invoke <see cref="InitializeCpu(ulong)"/> beforehand.
    /// </summary>
    /// <param name="exponent">Next exponent in the ascending sequence. Callers must never move backwards or reuse an older exponent.</param>
    /// <returns>The canonical residue produced by <c>2^exponent mod divisor</c>.</returns>
    public ulong ComputeNextCpu(ulong exponent)
    {
        ulong delta = exponent - PreviousExponent;
        // TODO: Once divisor cycle lengths are mandatory, pull the delta multiplier from the
        // single-block divisor-cycle snapshot so we can skip the powmod entirely and reuse the
        // cached Montgomery residue ladder highlighted in MersenneDivisorCycleLengthGpuBenchmarks.
        ulong multiplier = delta.Pow2MontgomeryModWindowedCpu(_divisor, keepMontgomery: true);
        _currentMontgomery = _currentMontgomery.MontgomeryMultiplyCpu(multiplier, _modulus, _nPrime);
        PreviousExponent = exponent;
        return ReduceCurrent();
    }

    /// <summary>
    /// Advances the CPU stepping state to the next exponent and reports whether it evaluates to unity.
    /// Callers must seed the stepper through <see cref="InitializeCpu(ulong)"/> or <see cref="InitializeCpuIsUnity(ulong)"/> and then
    /// pass exponents in strictly ascending order.
    /// </summary>
    /// <param name="exponent">Next exponent in the ascending sequence.</param>
    /// <returns><see langword="true"/> when the canonical residue equals one.</returns>
    public bool ComputeNextIsUnity(ulong exponent)
    {
        ulong delta = exponent - PreviousExponent;
        // TODO: Reuse the divisor-cycle derived Montgomery delta once the cache exposes single-cycle
        // lookups so this branch also avoids recomputing powmods when the snapshot lacks the divisor.
        ulong multiplier = delta.Pow2MontgomeryModWindowedCpu(_divisor, keepMontgomery: true);
        _currentMontgomery = _currentMontgomery.MontgomeryMultiplyCpu(multiplier, _modulus, _nPrime);
        PreviousExponent = exponent;
        return _currentMontgomery == _montgomeryOne;
    }

    /// <summary>
    /// Seeds the stepper with a precomputed Montgomery residue produced by the caller.
    /// The provided exponent becomes the new baseline and all subsequent calls must continue with greater exponents.
    /// </summary>
    /// <param name="exponent">Exponent associated with <paramref name="montgomeryResult"/>.</param>
    /// <param name="montgomeryResult">Residue in Montgomery form produced externally for the configured divisor.</param>
    /// <param name="isUnity">Outputs whether the supplied residue equals Montgomery one.</param>
    /// <returns><see langword="true"/> when the residue is accepted and cached.</returns>
    public bool TryInitializeFromMontgomeryResult(ulong exponent, ulong montgomeryResult, out bool isUnity)
    {
        _currentMontgomery = montgomeryResult;
        PreviousExponent = exponent;
        isUnity = _currentMontgomery == _montgomeryOne;
        return true;
    }

    /// <summary>
    /// Advances the stepper using a caller-supplied Montgomery delta that represents the gap to the next exponent.
    /// Callers must seed the stepper first, reuse the same divisor metadata, and continue feeding exponents in ascending order so the delta remains valid.
    /// </summary>
    /// <param name="exponent">Next exponent in the ascending sequence.</param>
    /// <param name="montgomeryDelta">Montgomery residue delta corresponding to the exponent gap.</param>
    /// <param name="isUnity">Outputs whether the resulting residue equals Montgomery one.</param>
    /// <returns><see langword="true"/> when the advance succeeds.</returns>
    public bool TryAdvanceWithMontgomeryDelta(ulong exponent, ulong montgomeryDelta, out bool isUnity)
    {
        // TODO: Once the divisor-cycle cache exposes a direct Montgomery delta, multiply it here instead
        // of relying on the caller-provided delta so incremental scans remain in sync with the snapshot
        // without computing additional cycles or mutating cache state.
        _currentMontgomery = _currentMontgomery.MontgomeryMultiplyCpu(montgomeryDelta, _modulus, _nPrime);
        PreviousExponent = exponent;
        isUnity = _currentMontgomery == _montgomeryOne;
        return true;
    }

    private void InitializeCpuState(ulong exponent)
    {
        _currentMontgomery = exponent.Pow2MontgomeryModWindowedCpu(_divisor, keepMontgomery: true);
        PreviousExponent = exponent;
    }

    private ulong ReduceCurrent() => _currentMontgomery.MontgomeryMultiplyCpu(1UL, _modulus, _nPrime);
}
