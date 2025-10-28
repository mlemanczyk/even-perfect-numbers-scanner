# `--mersenne=bydivisor` Divisibility Check Plan

The goal is to prove that every divisibility test along the by-divisor execution path is necessary. The plan below tracks the inspection order and the evidence required to remove redundant `%`/`if` combinations.

## Scope confirmation

- [x] Map the runtime flow from the CLI option handler through the CPU and GPU testers to the kernel dispatchers.
- [x] List every method involved in divisor generation, filtering, and validation, including the factorization helpers that introduce small primes.

## Checkpoint 1 – CPU-side divisibility guards

- [x] Review `EvenPerfectBitScanner/Program.cs` by-divisor setup for repeated modulo checks.
- [x] Inspect `PerfectNumbers.Core/MersenneNumberDivisorByDivisorTester.cs` for redundant primality or divisibility checks when staging GPU or CPU work.
- [x] Audit helper utilities under `PerfectNumbers.Core` that prepare divisor candidates (e.g., cycle calculators, factorization helpers) to ensure they do not repeat the same `%` conditions already enforced upstream.

## Checkpoint 2 – GPU staging layers

- [x] Examine `PerfectNumbers.Core/Gpu/MersenneNumberDivisorByDivisorGpuTester.cs` for duplicate `%` or remainder comparisons when feeding kernels, especially around cycle reuse.
- [x] Trace the inputs passed to each kernel to confirm the permissible value ranges make certain divisibility checks impossible.

## Checkpoint 3 – Kernel implementations

- [x] Verify all kernels under `PerfectNumbers.Core/Gpu/Kernels` that participate in the by-divisor path, documenting every explicit divisibility check.
- [x] For each `%` or `%=` inside the kernels, cross-reference the invariants guaranteed by the staging layers to justify keeping or removing the branch.

## Checkpoint 4 – Factorization edge cases

- [x] Confirm which stages can introduce small prime divisors during factorization even though `q = 2kp + 1` dominates the search, and adjust the above audits accordingly.

## Reporting

- [x] Update the divisibility review documentation with findings, noting every check removed or retained along with its rationale.
- [x] Ensure associated tests (unit, integration, or benchmarks) still pass after applying the verified optimizations.
