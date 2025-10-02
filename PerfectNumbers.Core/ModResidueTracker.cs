using System.Runtime.CompilerServices;

using UInt128 = System.UInt128;

namespace PerfectNumbers.Core;

public enum ResidueModel
{
	Identity,   // residue of N modulo d
	Mersenne,   // residue of (2^N - 1) modulo d
}

/// Tracks residues for a monotonically increasing sequence of numbers against
/// a dynamic set of divisors. Designed to support repeated divisor checks
/// while the tested number grows step-by-step without ever decreasing.
public sealed class ModResidueTracker
{
	public static readonly ModResidueTracker Shared = new(ResidueModel.Mersenne, initialNumber: PerfectNumberConstants.BiggestKnownEvenPerfectP, initialized: true);

	private readonly ResidueModel _model;
	private UInt128 _currentNumber;
	private bool _initialized;
	// Per-divisor sorted state at _currentNumber. Aligned arrays for speed.
	private readonly List<UInt128> _divisors = new(1024);   // sorted ascending
	private readonly List<UInt128> _residues = new(1024);   // same index as _divisors
	private int _cursor;                                    // merge pointer for ascending candidate stream
	private bool _hasLastAppended;
	private UInt128 _lastAppended;

	public ModResidueTracker(ResidueModel model, UInt128 initialNumber = default, bool initialized = false)
	{
		_model = model;
		_currentNumber = initialNumber;
		_initialized = initialized;
	}

	/// Returns true if the provided number (either p or M_p depending on model)
	/// is divisible by the divisor. Advances internal state forward if needed
	/// (number must be non-decreasing across calls). Adds and initializes new
	/// divisors on first use.
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public bool IsDivisible(UInt128 number, UInt128 divisor)
	{
		if (divisor == UInt128.Zero)
		{
			throw new ArgumentOutOfRangeException(nameof(divisor));
		}

		EnsureAt(number);

		List<UInt128> divisors = _divisors;
		int idx = LowerBound(divisors, divisor);
		if (idx == divisors.Count || divisors[idx] != divisor)
		{
			UInt128 residue = ComputeInitialResidue(number, divisor);
			divisors.Insert(idx, divisor);
			_residues.Insert(idx, residue);
			if (_hasLastAppended && idx == divisors.Count - 1 && divisor >= _lastAppended)
			{
				_lastAppended = divisor;
			}
			else if (!_hasLastAppended)
			{
				_hasLastAppended = true;
				_lastAppended = divisor;
			}

			return residue == UInt128.Zero;
		}

		return _residues[idx] == UInt128.Zero;
	}

	// Merge-walk support: prepare for increasing sequence of candidate divisors.
	public void BeginMerge(UInt128 number)
	{
		EnsureAt(number);
		_cursor = 0;
	}

	// Merge or append current candidate divisor (ascending order required across calls).
	// Returns true and sets 'divisible' based on residue at 'number'.
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public bool MergeOrAppend(UInt128 number, UInt128 divisor, out bool divisible)
	{
		// Assume number is non-decreasing; advance residues lazily when number changes.
		List<UInt128> divisors = _divisors;
		if (number != _currentNumber)
		{
			EnsureAt(number);
			// After EnsureAt we can keep _cursor (still valid monotonic pointer)
			if (_cursor > divisors.Count) _cursor = divisors.Count;
		}

		UInt128 residue;
		// Fast path: append to tail when candidates are strictly increasing.
		List<UInt128> residues = _residues;
		if (divisors.Count == 0 || (_hasLastAppended && divisor >= _lastAppended && divisors[^1] <= divisor))
		{
			residue = ComputeInitialResidue(number, divisor);
			divisors.Add(divisor);
			residues.Add(residue);
			_hasLastAppended = true;
			_lastAppended = divisor;
			_cursor = divisors.Count; // position after last
			divisible = residue == UInt128.Zero;
			return true;
		}

		// Merge pointer: advance until current >= divisor
		var cursor = _cursor;
		while (cursor < divisors.Count && divisors[cursor] < divisor)
		{
			cursor++;
		}

		if (cursor < divisors.Count && divisors[cursor] == divisor)
		{
			divisible = residues[cursor] == UInt128.Zero;
			_cursor = cursor + 1; // position after matched
			return true;
		}


		// Insert at cursor to keep order
		residue = ComputeInitialResidue(number, divisor);
		divisors.Insert(cursor, divisor);
		residues.Insert(cursor, residue);
		divisible = residue == UInt128.Zero;
		_cursor = cursor + 1;

		// update last appended hint
		if (!_hasLastAppended || divisor >= _lastAppended)
		{
			_hasLastAppended = true;
			_lastAppended = divisor;
		}

		return true;
	}

	/// Ensures the internal state corresponds to the provided number by moving
	/// forward from the current number (never backwards). O(Count(divisors)).
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void EnsureAt(UInt128 number)
	{
		if (!_initialized)
		{
			_currentNumber = number;
			// Initialize residues lazily upon first check for each divisor.
			_initialized = true;
			return;
		}

		if (number == _currentNumber)
		{
			return;
		}

		if (number < _currentNumber)
		{
			throw new InvalidOperationException("Number must be non-decreasing.");
		}

		UInt128 delta = number - _currentNumber;
		List<UInt128> divisors = _divisors;
		int divisorsCount = divisors.Count;
		if (divisorsCount == 0)
		{
			_currentNumber = number;
			return;
		}

		// Advance all tracked residues from _currentNumber to 'number'.
		// For Identity: r' = (r + delta) mod d
		// For Mersenne: r' = (2^delta * r + (2^delta - 1)) mod d
		// Optimize small deltas with simple forms when possible.
		int i = 0;
		UInt128 d, value;
		List<UInt128> residues = _residues;

                if (_model == ResidueModel.Identity)
                {
                        // Avoid 128-bit division: delta is small in scanner (<= 6) so use add/sub.
                        for (; i < divisorsCount; i++)
                        {
                                d = divisors[i];
                                value = residues[i] + delta;
                                // TODO: Replace this subtraction loop with the divisor-cycle stepping helper so identity
                                // residues reuse cached cycle deltas, computing the missing cycle on the configured
                                // device without caching the result or scheduling extra blocks when the snapshot lacks
                                // the divisor.
                                while (value >= d)
                                {
                                        value -= d;
                                }

                                residues[i] = value;
                        }
                }
		else
		{
                        for (; i < divisors.Count; i++)
                        {
                                d = divisors[i];
                                // TODO: Replace this PowMod128 call with the upcoming eight-bit window helper once the
                                // ProcessEightBitWindows scalar implementation lands so residue updates benefit from the
                                // ~2× win recorded in GpuPow2ModBenchmarks for large moduli.
                                value = PowMod128(UInt128Numbers.Two, delta, d);
                                // r' = pow2Delta*r + (pow2Delta - 1) mod d
                                value = MulMod128(value, residues[i], d) + value - UInt128.One;
                                if (value >= d)
                                {
					value -= d;
				}

				residues[i] = value;
			}
		}

		_currentNumber = number;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static int LowerBound(List<UInt128> list, UInt128 value)
	{
		int lo = 0, hi = list.Count, mid;
		while (lo < hi)
		{
			mid = (int)(((uint)lo + (uint)hi) >> 1);
			if (list[mid] < value)
			{
				lo = mid + 1;
			}
			else
			{
				hi = mid;
			}
		}

		return lo;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private UInt128 ComputeInitialResidue(UInt128 number, UInt128 divisor)
	{
                if (_model == ResidueModel.Identity)
                {
                        // Fast path for 64-bit inputs
                        if (number <= ulong.MaxValue && divisor <= ulong.MaxValue)
                        {
                                // TODO: Swap this modulo fallback for the shared divisor-cycle remainder helper so 64-bit
                                // residues reuse cached cycles instead of recomputing `%` on every lookup.
                                return (ulong)number % (ulong)divisor;
                        }

                        // TODO: Use the Montgomery folding helper measured faster in the Pow2Montgomery benchmarks so this
                        // wide residue path avoids general `%` on UInt128 inputs.
                        return number % divisor;
                }

                // residue of Mersenne number M_number = 2^number - 1 modulo divisor
                // TODO: Consult MersenneDivisorCycles.Shared (or similar caches) here so we reuse precomputed cycle lengths
                // instead of recomputing powmods for every divisor; the divisor-cycle benchmarks showed large wins once the
                // cached orders were used across scans, and a miss should trigger the configured device to compute the cycle
                // immediately without storing it back or requesting additional cache blocks.
                UInt128 pow = PowMod128(UInt128Numbers.Two, number, divisor);

                // q divides M_p ⇔ 2^p ≡ 1 (mod q)  AND  ord_q(2) | p
                if (pow != UInt128.One)
                        return (pow - UInt128.One) % divisor;

		ulong ord = divisor.CalculateOrder();
		if (ord == 0UL || divisor % ord != UInt128.Zero)
			return UInt128.One; // does NOT divide

		return UInt128.Zero; // q really divides M_p
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static UInt128 PowMod128(UInt128 baseValue, UInt128 exponent, UInt128 modulus)
        {
                if (modulus == UInt128.One)
                {
                        return UInt128.Zero;
                }

                UInt128 result = UInt128.One;
                UInt128 value = baseValue % modulus;

                while (exponent != UInt128.Zero)
                {
                        if ((exponent & UInt128.One) != UInt128.Zero)
                        {
                                // TODO: Route this through the planned UInt128 windowed powmod so the inner MulMod adopts the
                                // UInt128BuiltIn path that dominated the MulHighBenchmarks on wide operands instead of the
                                // current square-and-multiply loop.
                                result = MulMod128(result, value, modulus);
                        }

                        exponent >>= 1;
                        if (exponent != UInt128.Zero)
			{
				value = MulMod128(value, value, modulus);
			}
		}

		return result;
	}
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static UInt128 MulMod128(UInt128 a, UInt128 b, UInt128 modulus)
        {
                // Same approach as MulMod in MersenneNumberTester: double-and-add to avoid overflow.
                // TODO: Replace this double-and-add fallback with the UInt128 intrinsic-backed multiplier once the
                // MulHighBenchmarks-guided implementation lands; the intrinsic path measured dozens of times faster for
                // dense 128-bit workloads.
                UInt128 result = UInt128.Zero;
                a %= modulus;

                while (b != UInt128.Zero)
                {
			if ((b & UInt128.One) != UInt128.Zero)
			{
				result += result < modulus ? a : a - modulus;
			}

			a = a < modulus ? a << 1 : (a << 1) - modulus;
			b >>= 1;
		}

		return result;
	}
}
