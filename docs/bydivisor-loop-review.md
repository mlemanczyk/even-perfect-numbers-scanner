# --mersenne=bydivisor Loop Review

This checklist tracks each loop that executes when the scanner runs with `--mersenne=bydivisor`. Every entry lists the branch, the loop construct, and the early-exit behavior once the result of the computation is known.

- [x] `EvenPerfectBitScanner/Program.cs`
  - The by-divisor branch loads candidates with `CalculationResultsFile.LoadCandidatesWithinRange` and filters previous results. The loops here iterate the entire input because the program must consume every line from the filter file and prior results before it can enqueue candidates. No early break is possible without skipping data.
  - `Parallel.ForEach` dispatches `ProcessPrime` tasks. Each task exits immediately after printing the classification for its assigned prime, so no extra work continues once a result is produced.
- [x] `EvenPerfectBitScanner/IO/CalculationResultsFile.LoadCandidatesWithinRange`
  - The `while`/`for` loops stream the filter file once, emitting each numeric token. They terminate as soon as the file ends; the parser never performs redundant work after discovering that a value lies outside the configured limit.
- [x] `PerfectNumbers.Core/MersenneNumberDivisorByDivisorTester.Run`
  - Candidate compaction, start-prime filtering, and ArrayPool copies use `for` loops that must inspect every element to build the working set. The per-prime lambda exits with `return` as soon as the tester reports the outcome, preventing any follow-up processing for that prime.
- [x] `PerfectNumbers.Core/Cpu/MersenneNumberDivisorByDivisorCpuTester`
  - Both `CheckDivisors` variants (`while (divisor.CompareTo(limit) <= 0)` and `while (true)`) return immediately once a divisor cycle equals the tested prime. This short-circuits the scan when compositeness is established and avoids further modulus or Montgomery work.
- [x] `PerfectNumbers.Core/Gpu/MersenneNumberDivisorByDivisorGpuTester`
  - `CheckDivisors` now stops the inner candidate loop the moment a divisor with cycle length `p` appears. The method records the processed count, publishes the last divisor, flags the range as covered, and returns without enqueuing additional GPU work. Subsequent loops are skipped entirely after the early exit.
- [x] `PerfectNumbers.Core/Gpu/Kernels/DivisorByDivisorKernels.CheckKernel`
  - Each GPU thread traverses its assigned exponent list. The loops break naturally because the exponent sequences are finite; a hit triggers an atomic update so the host stops scheduling new batches after reading the first index.

All loops reachable through `--mersenne=bydivisor` now either require a complete pass to gather inputs or cut off the moment a decisive result is available.
