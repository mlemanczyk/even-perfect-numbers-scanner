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

- `MersenneDivisorCycles`
  - `TryCalculateCycleLengthForExponentCpu` → _Reviewed_
  - `TryFactorIntoCountsInternal` → _Reviewed_
  - `PollardRho64` → _Reviewed_
  - `ReduceOrder` (includes `ProcessReduceOrderPrime`) → _Reviewed_
- `PrimeOrderCalculator`
  - `InitializeStartingOrderCpu` → _Reviewed_
  - `ExponentLoweringCpu` / `ProcessExponentLoweringPrime` → _Reviewed_
  - `ValidateOrderAgainstFactors` / `ValidateOrderForFactor` → _Reviewed_
  - `TryConfirmCandidateCpu` / `CheckCandidateViolation` → _Reviewed_
  - `TryHeuristicFinishCpu` / `BuildCandidates` / `SortCandidates` → _Reviewed_

## Next Steps
- Hold for additional guidance on further --mersenne=bydivisor audits; computation branches documented above are fully reviewed.
- Defer result aggregation and reporting reviews until instructed otherwise, keeping the focus on computation-only branches.
