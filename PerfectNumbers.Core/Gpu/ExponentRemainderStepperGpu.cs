using System.Runtime.CompilerServices;
using ILGPU.Algorithms;
using PerfectNumbers.Core;

namespace PerfectNumbers.Core.Gpu;

/// <summary>
/// Tracks successive residues for a single divisor while kernels advance exponent candidates on the GPU path.
/// Callers must seed the stepper once per divisor and then supply exponents in strictly ascending order.
/// </summary>
internal struct ExponentRemainderStepperGpu
{
    private const int WindowSizeMax = 8;

    private readonly ulong _modulus;
    private ulong _previousExponent;
    private ulong _currentResidue;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ExponentRemainderStepperGpu(in MontgomeryDivisorData divisor)
    {
        _modulus = divisor.Modulus;
        _previousExponent = 0UL;
        _currentResidue = 1UL;
    }

    public ulong PreviousExponent
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _previousExponent;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool MatchesDivisor(in MontgomeryDivisorData divisor)
    {
        return _modulus == divisor.Modulus;
    }

    /// <summary>
    /// Clears the cached residue so the stepper can be reused for another divisor.
    /// Callers must reinitialize the stepper before consuming any residues after a reset.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Reset()
    {
        _previousExponent = 0UL;
        _currentResidue = 1UL;
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
        return _currentResidue;
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
        return _currentResidue == 1UL;
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
        return _currentResidue;
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
        return _currentResidue == 1UL;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void InitializeStateGpu(ulong exponent)
    {
        _currentResidue = Pow2ModWindowed(exponent, _modulus);
        _previousExponent = exponent;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void AdvanceStateGpu(ulong exponent)
    {
        ulong delta = exponent - _previousExponent;
        ulong multiplier = Pow2ModWindowed(delta, _modulus);
        _currentResidue = _currentResidue.MulMod64Gpu(multiplier, _modulus);
        _previousExponent = exponent;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ulong Pow2ModWindowed(ulong exponent, ulong modulus)
    {
        if (exponent == 0UL)
        {
            // The GPU by-divisor flow never routes modulus ≤ 1 here because divisors follow q = 2kp + 1 ≥ 3.
            return 1UL;
        }

        int bitLength = GetBitLength(exponent);
        int windowSize = GetWindowSize(bitLength);
        ulong result = 1UL;
        int index = bitLength - 1;

        while (index >= 0)
        {
            ulong currentBit = (exponent >> index) & 1UL;
            ulong squared = result.MulMod64Gpu(result, modulus);
            bool processWindow = currentBit != 0UL;

            result = processWindow ? result : squared;
            index -= (int)(currentBit ^ 1UL);

            int windowStartCandidate = index - windowSize + 1;
            int negativeMask = windowStartCandidate >> 31;
            windowStartCandidate &= ~negativeMask;

            int windowStart = processWindow ? GetNextSetBitIndex(exponent, windowStartCandidate) : windowStartCandidate;
            int windowLength = processWindow ? index - windowStart + 1 : 0;

            for (int square = 0; square < windowLength; square++)
            {
                result = result.MulMod64Gpu(result, modulus);
            }

            if (!processWindow)
            {
                continue;
            }

            ulong mask = (1UL << windowLength) - 1UL;
            ulong windowValue = (exponent >> windowStart) & mask;
            ulong multiplier = ComputeWindowedOddPower(windowValue, modulus);
            result = result.MulMod64Gpu(multiplier, modulus);
            index = windowStart - 1;
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong ComputeWindowedOddPower(ulong windowValue, ulong modulus)
    {
        // Production divisors on the GPU path always satisfy modulus ≥ 3 (q = 2kp + 1), so the base stays 2 without a reduction.
        ulong baseValue = 2UL;
        if (windowValue == 1UL)
        {
            return baseValue;
        }

        ulong result = baseValue;
        ulong remaining = (windowValue - 1UL) >> 1;
        ulong squareBase = baseValue.MulMod64Gpu(baseValue, modulus);

        while (remaining != 0UL)
        {
            result = ((remaining & 1UL) != 0UL) ? result.MulMod64Gpu(squareBase, modulus) : result;
            remaining >>= 1;
            squareBase = (remaining != 0UL) ? squareBase.MulMod64Gpu(squareBase, modulus) : squareBase;
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int GetBitLength(ulong value)
    {
        return value == 0UL ? 0 : 64 - XMath.LeadingZeroCount(value);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int GetWindowSize(int bitLength)
    {
        int window = WindowSizeMax;
        window = bitLength <= 671 ? 7 : window;
        window = bitLength <= 239 ? 6 : window;
        window = bitLength <= 79 ? 5 : window;
        window = bitLength <= 23 ? 4 : window;
        int clamped = bitLength >= 1 ? bitLength : 1;
        window = bitLength <= 6 ? clamped : window;
        return window;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int GetNextSetBitIndex(ulong exponent, int startIndex)
    {
        ulong guard = (ulong)(((long)startIndex - 64) >> 63);
        int shift = startIndex & 63;
        ulong mask = (~0UL << shift) & guard;
        ulong masked = exponent & mask;
        return XMath.TrailingZeroCount(masked);
    }
}
