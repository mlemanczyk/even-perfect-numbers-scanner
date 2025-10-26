# Heuristic Prime Tester Specification

This document describes the heuristic prime testing procedure implemented by `HeuristicPrimeTester.IsPrimeCpu` and `HeuristicPrimeTester.IsPrimeGpu`. The guidance consolidates the residue analysis and empirical statistics for odd inputs that are not divisible by five, based on the first 500,000 odd composite numbers after removing multiples of five. It also records the integration contracts required by the by-divisor pipeline so the heuristic covers every path that participates in divisor searches, residue sieves, and factorisation checks.

## Input Assumptions

* `n` is odd, `n > 2`, and `n` is not divisible by 5.
* The goal is to **prove compositeness** by finding a single proper divisor.
* Trial division always stops at `⌊√n⌋`, which the caller computes and supplies alongside the candidate.

## Candidate Partitioning

### Group A (special case)

```
A = {3, 7, 11, 13} ∪ { d : d ≡ 3 (mod 10), d ≤ ⌊√n⌋ }
```

Group A always executes first.

### Group B (remaining candidates)

Group B contains every other proper divisor `≤ ⌊√n⌋` that does **not** end with digit 3.

#### Allowed endings for Group B (after removing A)

The allowed endings depend on `n mod 10`:

* `n ≡ 1 (mod 10)` → `{1, 9}`
* `n ≡ 3 (mod 10)` → `{7, 9}`
* `n ≡ 7 (mod 10)` → `{1, 7}`
* `n ≡ 9 (mod 10)` → `{1, 7, 9}`

These sets are exhaustive. Do **not** add other endings.

## Wheel Strategy

* **Group A** uses the wheel based on 30 (residues coprime with 2 and 5: `{1, 7, 11, 13, 17, 19, 23, 29}`) combined with the `d ≡ 3 (mod 10)` filter. The constants `{3, 7, 11, 13}` are checked explicitly before wheel generation. The implementation uses wheel 30; a wheel of 210 may also be used for full uniformity, but 30 is lighter and already filtered to the `≡ 3 (mod 10)` residue.
* **Group B** uses a wheel based on 210 (coprime with 2, 3, 5, and 7). Only numbers whose final digit belongs to the allowed-ending set for the current `n mod 10` are retained.

## Testing Order

After Group A finishes, Group B enumerates candidates in strictly increasing order up to `⌊√n⌋`. Within that ascending order, endings are prioritised based on empirical hit rates **after excluding Group A**:

* `n ≡ 1 (mod 10)` → try endings `9` first, then `1`.
* `n ≡ 3 (mod 10)` → endings `9` then `7`.
* `n ≡ 7 (mod 10)` → endings `7` then `1`.
* `n ≡ 9 (mod 10)` → endings `9`, then `7`, then `1`.

No other endings are allowed or reordered.

## Algorithm Steps

1. Accept `R = ⌊√n⌋` and `nMod10` as parameters computed by the caller to avoid redundant square-root and modulo work and to enable reuse in other calculations.
2. **Group A constants**: test `{3, 7, 11, 13}` when they are `≤ R`. Any hit marks `n` as composite and terminates the search.
3. **Group A wheel**: generate candidates using the wheel-30 sequence, filter to `d ≡ 3 (mod 10)`, stop at `R`. Any hit marks `n` as composite and terminates the search. 
4. **Group B**: read the precomputed `n mod 10` (made available by the caller configuration such as `--gpu-prime-batch`). Iterate candidates via wheel 210, maintaining ascending order and applying the ending-priority sequence above. Any hit marks `n` as composite and terminates the search.
5. **Decision**: if no divisors were found up to `R`, report `n` as “prime by trial”. This outcome is final—no further confirmation runs are performed.

> **Temporary status:** Current builds disable Group B via `UseHeuristicGroupBTrialDivision`. After exhausting Group A, `HeuristicPrimeTester.IsPrime*` delegates to `Open.Numeric.Primes.Prime.Numbers.IsPrime`, preserving the Group B implementation for future use while ensuring production checks mirror the external library once Group A filters pass.

## Default Prime Tester Entry Points

* `PrimeTester.IsPrime` and `PrimeTester.IsPrimeGpu` continue to run the legacy trial-division and GPU sieve paths. Use `HeuristicPrimeTester` when the heuristic CPU/GPU flow is required.
* `HeuristicPrimeTester` always executes the heuristic CPU/GPU implementations; opt into the legacy paths by calling `PrimeTester.IsPrimeCpu` or `PrimeTester.Exclusive.IsPrimeGpu` directly.
* Benchmark harnesses can compare behaviours by calling the legacy helpers on `PrimeTester` alongside the heuristic equivalents on `HeuristicPrimeTester`.

## Integration with the By-Divisor Pipeline

The by-divisor executors (`MersenneNumberDivisorByDivisorCpuTester` and `MersenneNumberDivisorByDivisorGpuTester`) reuse prime heuristics to seed their divisor scans, residue sieves, and cycle computations. The heuristic implementations must therefore expose the following data and behaviours:

* **Residue filters** – The CPU scan maintains rolling residues modulo 10/8/5/3/7/11 to match the filters in `MersenneCpuDivisorScanSession.CheckDivisors`. `HeuristicPrimeTester.IsPrime*` must either share the residue update helpers or produce identical stepping deltas so the divisor scanners can reuse the pre-filtered candidate stream without recomputing moduli. The shared `MersenneDivisorResidueStepper` now centralises those modulo calculations so both paths reuse the same state transitions, and the GPU by-divisor tester instantiates the same stepper before filtering candidates.
* **Stepper integration** – CPU sessions advance residues and unity checks through `ExponentRemainderStepperCpu` together with `CycleRemainderStepper`; GPU kernels mirror this contract via `ExponentRemainderStepperGpu`. The heuristic implementation must call into these steppers (or emit the same remainder deltas) so the shared stepping state remains aligned across CPU/GPU scans and factor validation.
* **Wheel alignment** – `HeuristicPrimeSieves` precomputes the wheel 30/210 sequences once at startup so both the heuristic prime tester and the by-divisor executors consume identical Group A/B streams without regenerating residues on every call.
* **Divisor streaming API** – `HeuristicPrimeTester.CreateHeuristicDivisorEnumerator` exposes the same precomputed sieves so by-divisor scans can pull candidates directly into Montgomery preparation and GPU kernels without bespoke residue loops.
* **GPU batching** – `HeuristicPrimeTester.IsPrimeGpu` streams Group A/B candidates from `HeuristicPrimeSieves`, packs them into batches controlled by `HeuristicPrimeTester.HeuristicGpuDivisorBatchSize`, and launches `PrimeTesterKernels.HeuristicTrialDivisionKernel` to test each divisor with standard `%` arithmetic. The implementation now assumes the accelerator path is available and no longer routes through the CPU fallback.
* **Cycle preparation** – When the heuristic discovers a divisor candidate, it should populate `MontgomeryDivisorData` and expose the computed cycle length (or the need to recompute it) so that `DivisorCycleCache` and `MersenneDivisorCycles` do not duplicate work for the same `(prime, divisor)` pair during factor verification. `HeuristicPrimeTester.PrepareHeuristicDivisor` together with `HeuristicPrimeTester.ResolveHeuristicCycleLength` now provides this payload to both CPU and GPU by-divisor scans, and the trial-division summaries capture the data whenever a divisor is found.

* **Caching contract** – The by-divisor GPU implementation pools `DivisorScanSession` instances and reuses GPU buffers. The heuristic must accept externally allocated spans/buffers for divisor candidates, cycle lengths, and hits so the existing resource pools remain valid.

## Statistical Guidance

The following summaries cover the first 500,000 odd composite numbers after removing multiples of five (≈381,651 values). Among those, roughly 67,190 numbers had **no** divisor in Group A.

### Distribution of the smallest Group B divisor (when Group A contains a divisor)

* `n ≡ 1 (mod 10)` → ending `7`: **40.30%**, ending `9`: **38.38%**, ending `1`: **21.32%`. In these cases, `min(B) < A_max(n)` occurred **60.77%** of the time, while `min(B) > A_max(n)` occurred **39.23%** of the time.
* `n ≡ 3 (mod 10)` → ending `9`: **40.91%**, ending `1`: **38.92%**, ending `7`: **20.17%`. Here, `min(B) < A_max(n)` occurred **54.92%** of the time, while `min(B) > A_max(n)` occurred **45.08%** of the time.
* `n ≡ 7 (mod 10)` → ending `9`: **56.98%**, ending `7`: **21.52%**, ending `1`: **21.50%`. The relation `min(B) < A_max(n)` held **56.09%** of the time versus **43.91%** for `min(B) > A_max(n)`.
* `n ≡ 9 (mod 10)` → ending `9`: **49.64%**, ending `7`: **27.61%**, ending `1`: **22.75%`. In this class, `min(B) < A_max(n)` occurred **76.20%** of the time, with **23.80%** for `min(B) > A_max(n)`.

### Behaviour when no Group A divisor exists

* With full Group A processing, the minimum Group B divisor satisfied `min(B) < A_max(n)` about **99.67%** of the time, `min(B) = A_max(n)` essentially never, and `min(B) > A_max(n)` about **0.33%** of the time.
* The ratio `r = min(B) / A_max(n)` showed a median of ≈0.115, with quartiles `Q1 ≈ 0.044` and `Q3 ≈ 0.335`, confirming that the first Group B hit typically appears well before the tail of the Group A sweep.
* Minimum Group B divisors appeared **only** with the endings listed in the “Allowed endings” table above once Group A candidates were excluded.

### Heuristic cheat sheet

* Group A: test `3, 7, 11, 13` and every `d ≡ 3 (mod 10)` up to `⌊√n⌋` from the precomputed wheel-30 sieve.
* Group B: test only the allowed endings determined by `n mod 10`, scanning in ascending order from the wheel-210 sieves and applying the per-class priority order: `n ≡ 1` → `9` then `1`; `n ≡ 3` → `9` then `7`; `n ≡ 7` → `7` then `1`; `n ≡ 9` → `9`, `7`, `1`.
* Because `min(B)` is overwhelmingly smaller than `A_max(n)` for the “no Group A divisor” cohort, always start Group B from its smallest candidates instead of resuming above `A_max(n)`.

## Benchmark Reporting

Benchmark results comparing `HeuristicPrimeTester.IsPrimeGpu`, `HeuristicPrimeTester.IsPrimeCpu`, and `Open.Numeric.Primes.Prime.Numbers.IsPrime` will be documented here once the dedicated benchmarks covering ≤100, ≤4,000,000, and ≥138,000,000 candidates are executed. The specification will be updated with the measured timings and analysis so operators can evaluate heuristic rollouts against legacy and external implementations. Benchmark harnesses are available via `EvenPerfectBitScanner.Benchmarks/PrimeTesterBenchmarks.cs`, which runs the heuristic CPU and GPU paths alongside `Open.Numeric.Primes` for the three target ranges.

