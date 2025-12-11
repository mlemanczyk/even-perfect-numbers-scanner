using System.Collections.Concurrent;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

/// <summary>
/// Tracks successive Montgomery residues for a single divisor while scanning exponent candidates on the CPU path.
/// Callers are responsible for seeding the stepper once per divisor and then advancing exponents in ascending order.
/// </summary>
internal struct ExponentRemainderStepperCpu(in MontgomeryDivisorData divisor, ulong cycleLength = 0UL)
{
    private readonly MontgomeryDivisorData _divisor = divisor;
    private readonly ulong _modulus = divisor.Modulus;
    private readonly ulong _nPrime = divisor.NPrime;
    private readonly ulong _montgomeryOne = divisor.MontgomeryOne;
    private readonly ulong _cycleLength = cycleLength;
    private ulong PreviousExponent = 0UL;
    private ulong _currentMontgomery = divisor.MontgomeryOne;
    private static readonly ConcurrentDictionary<int, ulong> _cycleDeltaCache = new(20_480, PerfectNumberConstants.MaxQForDivisorCycles);

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public readonly bool MatchesDivisor(in MontgomeryDivisorData divisor, ulong cycleLength)
        => _modulus == divisor.Modulus && _nPrime == divisor.NPrime && _montgomeryOne == divisor.MontgomeryOne && _cycleLength == cycleLength;

	/// <summary>
	/// Clears the cached Montgomery residue so the stepper can be reused for another divisor.
	/// Callers must reinitialize the stepper before consuming any residues after a reset.
	/// </summary>
	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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
	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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
	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public ulong ComputeNextCpu(ulong exponent)
    {
        ulong delta = exponent - PreviousExponent;
        delta = ReduceDelta(delta);
        delta = delta == 0UL
            ? _montgomeryOne
            : delta.Pow2MontgomeryModWindowedKeepMontgomeryCpu(_divisor);
        _currentMontgomery = _currentMontgomery.MontgomeryMultiplyCpu(delta, _modulus, _nPrime);
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
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public bool ComputeNextIsUnity(ulong exponent)
    {
        ulong delta = exponent - PreviousExponent;
        PreviousExponent = exponent;
        delta = ReduceDelta(delta);
        delta = delta == 0UL
            ? _montgomeryOne
            : delta.Pow2MontgomeryModWindowedKeepMontgomeryCpu(_divisor);
        delta = _currentMontgomery.MontgomeryMultiplyCpu(delta, _modulus, _nPrime);
		_currentMontgomery = delta;
        return delta == _montgomeryOne;
    }

    /// <summary>
    /// Seeds the stepper with a precomputed Montgomery residue produced by the caller.
    /// The provided exponent becomes the new baseline and all subsequent calls must continue with greater exponents.
    /// </summary>
    /// <param name="exponent">Exponent associated with <paramref name="montgomeryResult"/>.</param>
    /// <param name="montgomeryResult">Residue in Montgomery form produced externally for the configured divisor.</param>
    /// <param name="isUnity">Outputs whether the supplied residue equals Montgomery one.</param>
    /// <returns><see langword="true"/> when the residue is accepted and cached.</returns>
	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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
	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public bool TryAdvanceWithMontgomeryDelta(ulong exponent, ulong montgomeryDelta, out bool isUnity)
	{
        ulong delta = exponent - PreviousExponent;
        delta = ReduceDelta(delta);

        // Prefer a cycle-reduced powmod so stepping stays aligned with the divisor-cycle cache.
        if (delta == 0UL)
        {
            montgomeryDelta = _montgomeryOne;
        }
        else if (_cycleLength != 0UL)
        {
            montgomeryDelta = delta.Pow2MontgomeryModWindowedKeepMontgomeryCpu(_divisor);
        }

        delta = _currentMontgomery.MontgomeryMultiplyCpu(montgomeryDelta, _modulus, _nPrime);
        PreviousExponent = exponent;
        isUnity = delta == _montgomeryOne;
		_currentMontgomery = delta;
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private void InitializeCpuState(ulong exponent)
    {
        ulong rotation = ReduceDelta(exponent);
        _currentMontgomery = rotation == 0UL
            ? _montgomeryOne
            : rotation.Pow2MontgomeryModWindowedKeepMontgomeryCpu(_divisor);
        PreviousExponent = exponent;
    }

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private readonly ulong ReduceCurrent() => _currentMontgomery.MontgomeryMultiplyCpu(1UL, _modulus, _nPrime);

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private readonly ulong ReduceDelta(ulong delta)
	{
		ulong cycleLength = _cycleLength;
		if (cycleLength == 0UL)
		{
			return delta;
		}

		int key = BuildKey(delta, cycleLength);
		if (_cycleDeltaCache.TryGetValue(key, out ulong cached))
		{
			return cached;
		}

		cycleLength = delta.ReduceCycleRemainder(cycleLength);
		_ = _cycleDeltaCache.TryAdd(key, cycleLength);
		return cycleLength;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static int BuildKey(ulong delta, ulong cycleLength)
	{
		int h1 = delta.GetHashCode();
		int h2 = cycleLength.GetHashCode();
		return h1 * 33 ^ h2;
	}
}
