# `--mersenne=bydivisor` Divisibility Review

This note captures the invariants established while auditing divisibility checks across the by-divisor path.

## Small-prime filters

- ✅ Every divisor candidate is generated as `2 * k * p + 1` with `p` prime, so the decimal mask derived from `LastDigit` limits candidates to endings `{1,3,7,9}`. Multiples of five therefore fail the mask without requiring an explicit `% 5` guard.
- ✅ Factoring stages reuse the same staging buffers and decimal-mask gate, so even when the GPU cycle resolver introduces smaller primes, multiples of five still cannot pass to the kernel queue.
- ✅ The remaining `% 3`, `% 7`, and `% 11` checks are necessary because those residues are not covered by the decimal mask and do appear for admissible `k` values.

## CPU staging (`PerfectNumbers.Core/MersenneNumberDivisorByDivisorCpuTester.cs`)

- ✅ Removed the `% 5` remainder tracker and the associated accumulator helpers; the decimal mask is sufficient to reject multiples of five before any expensive cycle work.
- ✅ Verified that the loop only evaluates `% 3`, `% 7`, and `% 11` once per candidate while the modulus-8 and decimal-mask filters provide the rest of the coverage.

## GPU staging (`PerfectNumbers.Core/Gpu/MersenneNumberDivisorByDivisorGpuTester.cs`)

- ✅ Mirrored the CPU cleanup by removing the `% 5` accumulator from the GPU preparation loop. The staging span now records only the residues that influence admissibility.
- ✅ Confirmed that kernel dispatch bypasses the redundant `% 5` guard without altering the admissible divisor sequence.

## Kernels

- ✅ Reviewed `PerfectNumbers.Core/Gpu/Kernels/DivisorByDivisorKernels.cs` to ensure no duplicate divisibility tests remain; kernels only consume the pre-filtered candidate list.

## Synchronization guard

- ✅ The GPU tester now relies on the existing initialization contract (`EnsureExecutionResourcesLocked`) and no longer emits a redundant null check before launching the kernel.

## Parity (mod 2) audit plan

- [x] Trace divisor candidate generation from `MersenneNumberDivisorByDivisorTester.Run` through the CPU and GPU staging loops to confirm they only advance odd sequences derived from `2 * k * p + 1`.
- [x] Inspect `PerfectNumbers.Core/Cpu/MersenneNumberDivisorByDivisorCpuTester.cs` for explicit parity guards inside the staging loops.
- [x] Inspect `PerfectNumbers.Core/Gpu/MersenneNumberDivisorByDivisorGpuTester.cs` and the associated kernels for redundant even checks.
- [x] Review factoring helpers (`MersenneDivisorCycles`, `MersennePrimeFactorTester`) to determine whether their parity filters remain necessary when factoring introduces small primes.

## Parity audit findings

- ✅ CPU and GPU staging loops rely on the modulus-8 remainder gate (`remainder8 == 1 || remainder8 == 7`) to enforce odd divisors, so no explicit `(divisor & 1)` checks remain in the hot paths.
- ✅ `MersenneDivisorResidueStepper` and the decimal mask restrict admissible residues so every generated divisor respects the `2 * k * p + 1` structure without additional parity branches.
- ✅ `MersenneDivisorCycles.TryCalculateCycleLengthForExponentCpu` legitimately strips powers of two from `phi = divisor - 1` while factoring; those divisions operate on the totient rather than the divisor candidates and must stay in place.
- ✅ Removed the redundant `(q & 1) == 0` guard from `MersennePrimeFactorTester.IsPrimeFactor` because the subsequent `(q & 7)` filter already excludes even candidates while the `q < 2` guard handles the remaining edge cases.

## Next steps

- Keep the decimal mask aligned with the divisor generator if additional last-digit heuristics are introduced so the `% 5` filter remains unnecessary.
