# CPU `--mersenne=bydivisor` pipeline notes

This document summarizes how the `EvenPerfectBitScanner` executable executes the
`--mersenne=bydivisor` mode on the CPU, and how the analytic observations we
derived for the quotients

\[
Q_k = \frac{N}{d_k}, \qquad d_k = 2pk + 1, \qquad N = 2^p - 1,
\]

can help us identify concrete optimization opportunities.

## High-level control flow

* `Program.Main` wires `_byDivisorTester` to a CPU implementation whenever the
  CLI switches resolve to `--mersenne=bydivisor` and
  `--mersenne-device=cpu`. The CLI also forces a block size of 1 so that the
  pipeline pushes exponents to the tester one at a time.
  (`Program.cs`, lines 459-520.)
* When the mode starts, `RunByDivisorMode` delegates to
  `MersenneNumberDivisorByDivisorTester.Run`, passing the candidate list from
  `--filter-p`, the CPU tester instance and the optional replay of previous
  outcomes. (`Program.cs`, lines 809-826.)
* `MersenneNumberDivisorByDivisorTester.Run` sorts and deduplicates the
  candidate exponents, drops the ones already confirmed in prior runs, validates
  primality with the `Open.Numeric.Primes` enumerator and configures the tester
  with the largest remaining exponent. (`MersenneNumberDivisorByDivisorTester.cs`,
  lines 14-162.)
* For each confirmed prime exponent, the CPU tester computes the largest divisor
  it needs to probe through `ComputeAllowedMaxDivisor`. If the range is non-empty
  it enters the `CheckDivisors` loop. (`MersenneNumberDivisorByDivisorCpuTester.cs`,
  lines 83-110 and 161-266.)

## Inner loop behaviour (`CheckDivisors`)

The hot path is the `while (divisor <= limit)` loop in
`MersenneNumberDivisorByDivisorCpuTester.CheckDivisors`.

1. The loop enumerates divisor candidates of the form `d_k = 2pk + 1` by
   incrementing `divisor` by `step = 2p` every iteration. (`MersenneNumberDivisorByDivisorCpuTester.cs`,
   lines 177-256.)
2. Small modulus filters track `d_k (mod 3·5·7·8·10·11)` to discard impossible
   candidates without touching the divisor-cycle cache. (`MersenneNumberDivisorByDivisorCpuTester.cs`,
   lines 195-263.)
3. For every admissible divisor the tester fetches its Montgomery data and the
   cached cycle length. When the cache does not contain an entry the
   implementation recomputes the cycle by factoring the divisor. (`MersenneNumberDivisorByDivisorCpuTester.cs`,
   lines 223-254.)
4. A cycle equal to the tested exponent proves divisibility, so the loop aborts
   as soon as one hit is found. (`MersenneNumberDivisorByDivisorCpuTester.cs`,
   lines 241-247.)

`DivisorScanSession.CheckDivisor` follows the same `d_k` progression when a
batch API is used. It keeps a running Montgomery residue via
`ExponentRemainderStepper`, but the stepper currently recomputes the delta by
calling a full `Pow2MontgomeryModWindowed` whenever the exponent jumps.
(`MersenneNumberDivisorByDivisorCpuTester.cs`, lines 358-405 and
`ExponentRemainderStepper.cs`, lines 26-95.)

## Where the analytic model fits

The formulas we established for the quotient sequence

\[
\frac{Q_{k+1}}{Q_k} = \frac{2pk + 1}{2p(k+1) + 1}, \qquad
Q_k - Q_{k+1} = \frac{N·2p}{(2pk + 1)(2p(k+1) + 1)}
\]

tell us that the successive quotients and remainders shrink smoothly and that the
per-step update depends only on the previous `(Q_k, r_k)` state and the next
`d_{k+1}`. This lines up with several TODO comments that highlight missing
incremental updates:

* `ExponentRemainderStepper` already caches the previous exponent and a running
  Montgomery residue, but it falls back to a full modular exponentiation to
  compute the delta between consecutive divisors. (`ExponentRemainderStepper.cs`,
  lines 49-72.)
* `CheckDivisors` rebuilds the Montgomery data for every hit candidate even
  though the sequence of `d_k` values is strictly monotone.

The quotient identity gives us a way to compute `Δ_k = Q_k - Q_{k+1}` by reading
`Q_k` and the inexpensive division `(2pQ_k - r_k) / d_{k+1}`. Once we know `Δ_k`
we can update

\[
Q_{k+1} = Q_k - Δ_k, \qquad r_{k+1} = r_k - 2p·Q_k + Δ_k · d_{k+1},
\]

without ever evaluating `2^p mod d_{k+1}` from scratch. Incorporating this into
`ExponentRemainderStepper` would let the CPU path reuse a single Montgomery state
across the entire scan and swap the current delta exponentiation (which costs a
full modular ladder) for one multiply-high with a precomputed reciprocal of
`d_{k+1}`. The loop already maintains `d_k` and `step`, so the extra bookkeeping
fits naturally.

## Optimization ideas grounded in the code

1. **Delta-driven Montgomery updates.** Extend `DivisorScanSession` so it tracks
   `(Q_k, r_k)` alongside the Montgomery residue and uses the analytic delta
   formula to derive the next residue. This replaces the per-step powmod in
   `ExponentRemainderStepper.ComputeNext` with a handful of 64-bit operations.
   (`MersenneNumberDivisorByDivisorCpuTester.cs`, lines 358-405;
   `ExponentRemainderStepper.cs`, lines 49-72.)
2. **Bit-range skipping.** Because `Q_{k+1} / Q_k` is exactly the rational
   `(2pk + 1)/(2p(k+1) + 1)`, we can predict when `⌊log₂ Q_k⌋` loses a bit and
   jump the loop counter to the first `k` that crosses a target quotient bound
   instead of checking each divisor. This is compatible with the existing
   `allowedMax` guard because it only modifies how we arrive at the stopping
   divisor. (`MersenneNumberDivisorByDivisorCpuTester.cs`, lines 161-266.)
3. **Reuse Montgomery metadata.** The loop already calls `MontgomeryDivisorDataCache.Get`
   for each divisor, but the cache lookup is repeated even though the next
   divisor is a small increment away. Persisting the previous `MontgomeryDivisorData`
   together with the `(Q_k, r_k)` tuple lets us skip redundant cache reads when
   the divisor stays within the hot range. (`MersenneNumberDivisorByDivisorCpuTester.cs`,
   lines 223-256.)

These adjustments directly leverage the quotient progression we analysed and
align with the existing TODO list, giving us a clear path to reduce the number of
modular exponentiations and cache lookups in the CPU by-divisor path.
