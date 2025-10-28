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

## Next steps

- Keep the decimal mask aligned with the divisor generator if additional last-digit heuristics are introduced so the `% 5` filter remains unnecessary.
