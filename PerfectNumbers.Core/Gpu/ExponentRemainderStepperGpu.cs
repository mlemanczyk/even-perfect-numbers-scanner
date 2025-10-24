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
    public ulong PreviousExponent;
    private ulong _currentResidue;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ExponentRemainderStepperGpu(in MontgomeryDivisorData divisor)
    {
        _modulus = divisor.Modulus;
        PreviousExponent = 0UL;
        _currentResidue = 1UL;
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
        PreviousExponent = 0UL;
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
        PreviousExponent = exponent;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void AdvanceStateGpu(ulong exponent)
    {
        ulong delta = exponent - PreviousExponent;
        ulong multiplier = Pow2ModWindowed(delta, _modulus);
        _currentResidue = MultiplyMod(_currentResidue, multiplier, _modulus);
        PreviousExponent = exponent;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ulong Pow2ModWindowed(ulong exponent, ulong modulus)
    {
        // EvenPerfectBitScanner never provides zero exponents on the GPU path, so the guard stays commented out.
        // if (exponent == 0UL)
        // {
        //     return 1UL;
        // }

        int bitLength = GetBitLength(exponent);
        int windowSize = GetWindowSize(bitLength);
        ulong result = 1UL;
        int index = bitLength - 1;

        while (index >= 0)
        {
            ulong currentBit = (exponent >> index) & 1UL;
            ulong squared = MultiplyMod(result, result, modulus);
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
                result = MultiplyMod(result, result, modulus);
            }

            if (!processWindow)
            {
                continue;
            }

            ulong mask = (1UL << windowLength) - 1UL;
            ulong windowValue = (exponent >> windowStart) & mask;
            ulong multiplier = ComputeWindowedOddPower(windowValue, modulus);
            result = MultiplyMod(result, multiplier, modulus);
            index = windowStart - 1;
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MultiplyMod(ulong left, ulong right, ulong modulus)
    {
        // GpuUInt128.MulMod is the fastest GPU-compatible multiply-reduce helper available to this kernel path.
        var product = new GpuUInt128(left);
        return product.MulMod(right, modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong ComputeWindowedOddPower(ulong windowValue, ulong modulus)
    {
        // Production divisors on the GPU path always satisfy modulus ≥ 3 (q = 2kp + 1), so the base stays 2 without a reduction.
        ulong baseValue = 2UL;
        if (windowValue == 1UL)
        {
            // Single-bit windows occur frequently in production primes, so keep the fast return for window value 1.
            return baseValue;
        }

        ulong result = baseValue;
        ulong remaining = (windowValue - 1UL) >> 1;
        ulong squareBase = MultiplyMod(baseValue, baseValue, modulus);

        while (remaining != 0UL)
        {
            result = ((remaining & 1UL) != 0UL) ? MultiplyMod(result, squareBase, modulus) : result;
            remaining >>= 1;
            squareBase = (remaining != 0UL) ? MultiplyMod(squareBase, squareBase, modulus) : squareBase;
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
