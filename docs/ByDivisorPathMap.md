# --mersenne=bydivisor Path Map

This temporary map tracks the code paths exercised by the `--mersenne=bydivisor` mode and notes the latest review status for each branch. Entries move from _Queued_ to _Reviewed_ once the surrounding loops have been audited for invariant work.

## High-level Flow
- `EvenPerfectBitScanner` CLI parsing → _Reviewed_
- `MersenneNumberDivisorByDivisorTester.Run` → _Reviewed_
- `EvenPerfectBitScanner.IO.CalculationResultsFile`
  - `LoadCandidatesWithinRange` → _Reviewed_
  - `EnumerateCandidates` → _Reviewed_
- `IMersenneNumberDivisorByDivisorTester` implementations
  - `MersenneNumberDivisorByDivisorCpuTester`
    - `IsPrime` → _Reviewed_
    - `CheckDivisors` (128-bit path) → _Reviewed_
    - `CheckDivisors64` → _Reviewed_
  - `MersenneNumberDivisorByDivisorGpuTester`
    - `IsPrime` → _Reviewed_
    - `CheckDivisors` → _Reviewed_
    - `DivisorByDivisorKernels.CheckKernel` → _Reviewed_

## Next Steps
- Trace the result aggregation and reporting helpers invoked after divisor scans to confirm their loops only compute invariants once per batch.
- Revisit ancillary utilities invoked by the by-divisor flow (result handlers, task schedulers) to confirm their loops remain free of redundant invariant evaluations.

