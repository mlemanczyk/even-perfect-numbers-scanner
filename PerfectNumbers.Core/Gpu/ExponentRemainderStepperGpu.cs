using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core.Gpu;

/// <summary>
/// Tracks successive residues for a single divisor while kernels advance exponent candidates on the GPU path.
/// Callers must seed the stepper once per divisor and then supply exponents in strictly ascending order.
/// </summary>
internal struct ExponentRemainderStepperGpu
{
    private readonly MontgomeryDivisorData _divisor;
    private readonly ulong _modulus;
    private readonly ulong _nPrime;
    private readonly ulong _montgomeryOne;
    private ulong _previousExponent;
    private ulong _currentMontgomery;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ExponentRemainderStepperGpu(in MontgomeryDivisorData divisor)
    {
        _divisor = divisor;
        _modulus = divisor.Modulus;
        _nPrime = divisor.NPrime;
        _montgomeryOne = divisor.MontgomeryOne;
        _previousExponent = 0UL;
        _currentMontgomery = divisor.MontgomeryOne;
    }

    public ulong PreviousExponent
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _previousExponent;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool MatchesDivisor(in MontgomeryDivisorData divisor)
    {
        return _modulus == divisor.Modulus && _nPrime == divisor.NPrime && _montgomeryOne == divisor.MontgomeryOne;
    }

    /// <summary>
    /// Clears the cached Montgomery residue so the stepper can be reused for another divisor.
    /// Callers must reinitialize the stepper before consuming any residues after a reset.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Reset()
    {
        _previousExponent = 0UL;
        _currentMontgomery = _montgomeryOne;
    }

    /// <summary>
    /// Initializes the stepper for GPU calculations using the first exponent in a strictly ascending sequence.
    /// The returned value is the canonical residue corresponding to the provided exponent.
    /// </summary>
    /// <param name="exponent">First exponent evaluated for the current divisor. The caller must reuse the same divisor metadata.</param>
    /// <returns>The canonical residue produced by <c>2^exponent mod divisor</c>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ulong InitializeGpu(ulong exponent)
    {
        InitializeGpuState(exponent);
        return ReduceCurrent();
    }

    /// <summary>
    /// Initializes the GPU stepping state and reports whether the seed exponent already yields unity.
    /// </summary>
    /// <param name="exponent">First exponent evaluated for the current divisor. Must match the divisor passed to the constructor.</param>
    /// <returns><see langword="true"/> when the seed exponent produces a Montgomery residue equal to one.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool InitializeGpuIsUnity(ulong exponent)
    {
        InitializeGpuState(exponent);
        return _currentMontgomery == _montgomeryOne;
    }

    /// <summary>
    /// Advances the GPU stepping state to the next exponent in a strictly ascending sequence that shares the same divisor.
    /// Callers must invoke <see cref="InitializeGpu(ulong)"/> beforehand.
    /// </summary>
    /// <param name="exponent">Next exponent in the ascending sequence. Callers must never move backwards or reuse an older exponent.</param>
    /// <returns>The canonical residue produced by <c>2^exponent mod divisor</c>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ulong ComputeNextGpu(ulong exponent)
    {
        AdvanceGpu(exponent);
        return ReduceCurrent();
    }

    /// <summary>
    /// Advances the GPU stepping state to the next exponent and reports whether it evaluates to unity.
    /// Callers must seed the stepper through <see cref="InitializeGpu(ulong)"/> or <see cref="InitializeGpuIsUnity(ulong)"/> and then
    /// pass exponents in strictly ascending order.
    /// </summary>
    /// <param name="exponent">Next exponent in the ascending sequence.</param>
    /// <returns><see langword="true"/> when the canonical residue equals one.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool ComputeNextGpuIsUnity(ulong exponent)
    {
        AdvanceGpu(exponent);
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
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryInitializeFromMontgomeryResult(ulong exponent, ulong montgomeryResult, out bool isUnity)
    {
        _currentMontgomery = montgomeryResult;
        _previousExponent = exponent;
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
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryAdvanceWithMontgomeryDelta(ulong exponent, ulong montgomeryDelta, out bool isUnity)
    {
        _currentMontgomery = _currentMontgomery.MontgomeryMultiply(montgomeryDelta, _modulus, _nPrime);
        _previousExponent = exponent;
        isUnity = _currentMontgomery == _montgomeryOne;
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void InitializeGpuState(ulong exponent)
    {
        _currentMontgomery = ULongExtensions.Pow2MontgomeryModWindowedGpuKeepMontgomery(_divisor, exponent);
        _previousExponent = exponent;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void AdvanceGpu(ulong exponent)
    {
        ulong delta = exponent - _previousExponent;
        ulong multiplier = ULongExtensions.Pow2MontgomeryModWindowedGpuKeepMontgomery(_divisor, delta);
        _currentMontgomery = _currentMontgomery.MontgomeryMultiply(multiplier, _modulus, _nPrime);
        _previousExponent = exponent;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private ulong ReduceCurrent() => _currentMontgomery.MontgomeryMultiply(1UL, _modulus, _nPrime);
}
