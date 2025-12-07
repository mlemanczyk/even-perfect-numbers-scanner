using System.Runtime.CompilerServices;
using PerfectNumbers.Core.Gpu.Kernels;

namespace PerfectNumbers.Core.Gpu;

/// <summary>
/// Tracks successive residues for a single divisor while kernels advance exponent candidates on the GPU path.
/// Callers must seed the stepper once per divisor and then supply exponents in strictly ascending order.
/// </summary>
internal struct ExponentRemainderStepperGpu
{
    private readonly ulong _modulus;
    private readonly ReadOnlyGpuUInt128 _modulusWide;
    private ulong PreviousExponent;
    private GpuUInt128 _currentResidue;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ExponentRemainderStepperGpu(in GpuDivisorPartialData divisor)
    {
        ulong modulus = divisor.Modulus;
        _modulus = modulus;
        _modulusWide = divisor.ModulusWide;
        PreviousExponent = 0UL;
        _currentResidue = GpuUInt128.One;
    }

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public readonly bool MatchesDivisor(in GpuDivisorPartialData divisor) => _modulus == divisor.Modulus;

	/// <summary>
	/// Clears the cached residue so the stepper can be reused for another divisor.
	/// Callers must reinitialize the stepper before consuming any residues after a reset.
	/// </summary>
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Reset()
    {
        PreviousExponent = 0UL;
        _currentResidue = GpuUInt128.One;
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
        InitializeStateGpu(exponent);
        return _currentResidue.Low;
    }

    /// <summary>
    /// Initializes the GPU stepping state and reports whether the seed exponent already yields unity.
    /// </summary>
    /// <param name="exponent">First exponent evaluated for the current divisor. Must match the divisor passed to the constructor.</param>
    /// <returns><see langword="true"/> when the seed exponent produces a residue equal to one.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool InitializeIsUnityGpu(ulong exponent)
    {
        InitializeStateGpu(exponent);
        return _currentResidue.Low == 1UL;
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
        AdvanceStateGpu(exponent);
        return _currentResidue.Low;
    }

    /// <summary>
    /// Advances the GPU stepping state to the next exponent and reports whether it evaluates to unity.
    /// Callers must seed the stepper through <see cref="InitializeGpu(ulong)"/> or <see cref="InitializeIsUnityGpu(ulong)"/> and then
    /// pass exponents in strictly ascending order.
    /// </summary>
    /// <param name="exponent">Next exponent in the ascending sequence.</param>
    /// <returns><see langword="true"/> when the canonical residue equals one.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool ComputeNextIsUnityGpu(ulong exponent)
    {
        AdvanceStateGpu(exponent);
        return _currentResidue.Low == 1UL;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void InitializeStateGpu(ulong exponent)
    {
        ulong residue = exponent.Pow2ModWindowedGpuKernel(_modulus);
        _currentResidue = new GpuUInt128(residue);
        PreviousExponent = exponent;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void AdvanceStateGpu(ulong exponent)
    {
        ulong delta = exponent - PreviousExponent;
        ulong multiplier = delta.Pow2ModWindowedGpuKernel(_modulus);
        _currentResidue.MulMod(multiplier, _modulusWide);
        PreviousExponent = exponent;
    }
}
