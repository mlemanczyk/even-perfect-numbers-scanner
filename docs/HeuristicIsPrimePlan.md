# EvenPerfectBitScanner `--mersenne=bydivisor` Flow and Heuristic Prime Tester Plan

## 1. Execution Flow in `--mersenne=bydivisor` Mode

The following map captures every branch that executes when the scanner runs with `--mersenne=bydivisor`, highlighting interactions with prime testing, divisor heuristics, residue reuse, and factor verification.

1. **CLI processing and guards**
   * `CliArguments.Parse` fills `_cliArguments`, then `Main` validates thread count, help flag, and shared GPU settings.
   * `useByDivisor` is set from CLI. If enabled without `--filter-p` (and not in test mode), the scanner prints "--mersenne=bydivisor requires --filter-p=<path>." and exits early.【F:EvenPerfectBitScanner/Program.cs†L32-L132】

2. **Global configuration**
   * GPU settings (`GpuPrimeWorkLimiter`, `PrimeTester.GpuBatchSize`, transforms) are configured for both CPU and GPU runs.
   * By-divisor specific state records `_byDivisorStartPrime`, `_byDivisorTester`, and `_byDivisorPreviousResults` for later reuse.【F:EvenPerfectBitScanner/Program.cs†L134-L232】

3. **Tester initialization**
   * When `useByDivisor` is active, `_byDivisorTester` is instantiated as GPU or CPU implementation, with `BatchSize` set from `--scan-batch`. `PrimeTester`/`MersenneNumberTester` pools are not created for this mode because the flow exits before the generic candidate loop.【F:EvenPerfectBitScanner/Program.cs†L200-L232】【F:EvenPerfectBitScanner/Program.cs†L400-L460】

4. **Results bootstrap**
   * The scanner builds the results filename, initializes `CalculationResultHandler`, and loads previous results. Previous entries populate `_byDivisorPreviousResults` so the by-divisor runner can skip already processed primes.【F:EvenPerfectBitScanner/Program.cs†L232-L318】

5. **Candidate acquisition**
   * `byDivisorCandidates` is populated either from deterministic test data (test mode) or by reloading the filter file with `CalculationResultsFile.LoadCandidatesWithinRange`. Limits (`--max-prime`) and skipped candidates are reported to the console.【F:EvenPerfectBitScanner/Program.cs†L318-L362】
   * In contrast to other modes, generic residue filters and candidate queues are bypassed; only the provided prime list is considered.【F:EvenPerfectBitScanner/Program.cs†L318-L362】

6. **Previous-result pruning**
   * `MersenneNumberDivisorByDivisorTester.Run` sorts the input primes, removes entries already present in `_byDivisorPreviousResults`, applies `--start-prime`, and converts the remaining list to spans for batch preparation.【F:PerfectNumbers.Core/MersenneNumberDivisorByDivisorTester.cs†L11-L138】

7. **Tester configuration per maximum prime**
   * `IMersenneNumberDivisorByDivisorTester.ConfigureFromMaxPrime` and `PrepareCandidates` pre-compute divisor limits for every prime. All primes with `allowedMax >= 3` are retained (current implementations never filter any out in production because limits are huge).【F:PerfectNumbers.Core/MersenneNumberDivisorByDivisorTester.cs†L140-L201】

8. **Residue-sieved divisor scanning (CPU)**
   * `MersenneNumberDivisorByDivisorCpuTester.CheckDivisors` initialises Montgomery stepping data (`GpuUInt128`), computes rolling residues for mod 10/8/5/3/7/11, and advances divisors by `2p` while applying decimal masks tied to the exponent’s last digit.【F:PerfectNumbers.Core/Cpu/MersenneNumberDivisorByDivisorCpuTester.cs†L63-L213】
   * When a candidate passes residue filters, the CPU session obtains cycle lengths from `DivisorCycleCache` or `MersenneDivisorCycles`, validates them via Montgomery exponentiation, and reports hits while updating `_lastStatusDivisor`.【F:PerfectNumbers.Core/Cpu/MersenneNumberDivisorByDivisorCpuTester.cs†L213-L318】

9. **Residue-sieved divisor scanning (GPU)**
   * `MersenneNumberDivisorByDivisorGpuTester.CheckDivisors` reuses the same wheel offsets and residue tests but performs them inside `DivisorByDivisorKernels.CheckKernel`, writing hits to pooled GPU buffers. CPU fallbacks honour the same residue, decimal mask, and cycle rules when GPU context is unavailable.【F:PerfectNumbers.Core/Gpu/MersenneNumberDivisorByDivisorGpuTester.cs†L1-L240】【F:PerfectNumbers.Core/Gpu/MersenneNumberDivisorByDivisorGpuTester.cs†L240-L520】

10. **Prime decision path**
    * By-divisor sessions internally reuse divisor remainders across primes, but they **do not call `PrimeTester.IsPrime*`**. The primality decision depends entirely on divisor scans up to the limit configured for each prime candidate.
    * If `IsPrime` returns `false`, the result file records `(detailedCheck=true, passedAllTests=false)`. When `true`, `divisorsExhausted` reports whether the scan reached the configured divisor bound (driving the `DetailedCheck` flag).【F:PerfectNumbers.Core/MersenneNumberDivisorByDivisorTester.cs†L203-L272】

11. **Shutdown**
    * After `MersenneNumberDivisorByDivisorTester.Run` completes, the main loop flushes results and exits before any of the generic candidate enumeration logic runs.【F:EvenPerfectBitScanner/Program.cs†L400-L460】

## 2. Baseline Prime Tester Methods

The heuristics will replace the internal logic of the current GPU-assisted and CPU-only paths. The snapshots below capture the existing implementations before modification (copied verbatim for reference).

```csharp
public bool IsPrimeGpu(ulong n, CancellationToken ct)
{
    bool forceCpu = GpuContextPool.ForceCpu;
    Span<ulong> one = stackalloc ulong[1];
    Span<byte> outFlags = stackalloc byte[1];
    one[0] = n;
    outFlags[0] = 0;

    if (!forceCpu)
    {
        IsPrimeBatchGpu(one, outFlags);
    }

    bool belowGpuRange = n < 31UL;
    bool gpuReportedPrime = !forceCpu && !belowGpuRange && outFlags[0] != 0;
    bool requiresCpuFallback = forceCpu || belowGpuRange || !gpuReportedPrime;

    return requiresCpuFallback ? IsPrimeCpu(n, ct) : true;
}
```

```csharp
internal static bool IsPrimeCpu(ulong n, CancellationToken ct)
{
    bool isTwo = n == 2UL;
    bool isOdd = (n & 1UL) != 0UL;
    ulong mod5 = n % 5UL;
    bool divisibleByFive = n > 5UL && mod5 == 0UL;

    bool result = n >= 2UL && (isTwo || isOdd) && !divisibleByFive;
    bool requiresTrialDivision = result && n >= 7UL && !isTwo;

    if (requiresTrialDivision)
    {
        bool sharesMaxExponentFactor = n.Mod10() == 1UL && SharesFactorWithMaxExponent(n);
        result &= !sharesMaxExponentFactor;

        if (result)
        {
            var smallPrimeDivisorsLength = PrimesGenerator.SmallPrimes.Length;
            uint[] smallPrimeDivisors = PrimesGenerator.SmallPrimes;
            ulong[] smallPrimeDivisorsMul = PrimesGenerator.SmallPrimesPow2;
            for (int i = 0; i < smallPrimeDivisorsLength; i++)
            {
                if (smallPrimeDivisorsMul[i] > n)
                {
                    break;
                }

                if (n % smallPrimeDivisors[i] == 0)
                {
                    result = false;
                    break;
                }
            }
        }
    }

    return result;
}
```

## 3. Implementation Plan for `HeuristicPrimeTester`

The plan incorporates the updated divisor-class heuristics (Groups A/B, wheels, residue priorities) and connects them to both CPU and GPU execution, including every by-divisor integration point. Items already completed are marked with `[done]`.

1. [done] **Snapshot current implementations** – preserve `IsPrimeGpu` and `IsPrimeCpu` as references for regression comparisons and migration checkpoints (see Section 2).
2. [done] **Introduce placeholder heuristic entry points** – add `IsPrimeCpu` and `IsPrimeGpu` by copying the current internal and GPU methods so subsequent commits can iterate without disrupting legacy behavior.
3. [done] **Design shared heuristic scaffolding**
   * [done] Added `HeuristicPrimeSieves` to precompute 4M-entry Group A/B divisor tables so heuristic prime checks reuse cached sequences without regenerating wheel steps at runtime; the enumerator remains available for residue-driven consumers.
   * [done] Surface helpers that accept the precomputed `R = ⌊√n⌋` and `nMod10` values from callers via the new overload of `IsPrimeCpu`; HeuristicPrimeTester still computes them locally until call sites forward the parameters.
   * [done] Simplified the heuristic execution path so it runs without cancellation tokens or callback plumbing while keeping the shared enumerator available to other components.
4. **Share residue and wheel state with by-divisor sessions**
   * [done] Extracted the rolling-modulus helpers (mod 10/8/5/3/7/11) and decimal masks from `MersenneNumberDivisorByDivisorCpuTester.CheckDivisors` into the shared `MersenneDivisorResidueStepper` so heuristic prime tests and by-divisor sessions consume the same filters.
   * [done] Audited how `MersenneCpuDivisorScanSession` advances residues via `ExponentRemainderStepperCpu`/`CycleRemainderStepper` and updated the GPU by-divisor path to reuse `MersenneDivisorResidueStepper`, keeping CPU/GPU residue stepping aligned.
   * [done] Publish wheel iterators that can stream candidates directly into `DivisorCycleCache`, GPU hit buffers, or CPU Montgomery checks without recomputing offsets by exposing `HeuristicPrimeTester.CreateHeuristicDivisorEnumerator` and the reusable `HeuristicGroupBSequenceState` buffer.
5. **Expose cycle-preparation helpers**
   * [done] Extend the heuristic scaffolding so it returns `MontgomeryDivisorData` descriptors and cycle-length hints for each candidate divisor via `HeuristicDivisorPreparation` and `PrepareHeuristicDivisor`.
   * Wire these outputs into the existing cycle calculators to avoid duplicate cache lookups inside the CPU/GPU divisor scans.
6. [done] **Implement `IsPrimeCpu`**
   * Baseline Group A/B enumeration now executes inside `IsPrimeCpu` via the cached `HeuristicPrimeSieves` tables, computing `⌊√n⌋` locally and short-circuiting on the first divisor.
   * [done] The CPU heuristic now operates as a pure trial-division sweep without maintaining per-call summaries, relying on `PrepareHeuristicDivisor` only when downstream consumers request Montgomery data.
   * [done] Introduced the temporary `UseHeuristicGroupBTrialDivision` gate so current builds execute Group A locally and then fall back to `Open.Numeric.Primes.Prime.Numbers.IsPrime`, keeping the full Group B implementation available for reactivation.
7. [done] **Implement `IsPrimeGpu`**
   * [done] Mirror CPU structure while batching divisor checks on the accelerator via `HeuristicTrialDivisionGpu`, drawing divisors from `HeuristicPrimeSieves`, reusing caller-provided `sqrtLimit`/`nMod10` inputs, and falling back to CPU when GPUs are disabled.
   * [done] Added `PrimeTesterKernels.HeuristicTrialDivisionKernel` so GPU batches respect Group A/B ordering while emitting hit flags for early exits.
   * [done] Kept GPU divisibility in standard modular arithmetic while reserving Montgomery transforms for CPU confirmation logic.
   * [done] When the temporary Group B gate is disabled, delegate GPU calls to the CPU fallback so Group A coverage remains in place while the accelerator path stays available for future re-enablement.
8. [done] **Refactor by-divisor CPU/GPU testers to reuse heuristics**
   * [done] Drove `MersenneNumberDivisorByDivisorCpuTester.CheckDivisors` from `HeuristicPrimeTester.CreateMersenneDivisorEnumerator`, eliminating the bespoke residue loop and retiring the legacy status counter.
   * [done] Streamed GPU batches from the same enumerator and Montgomery preparation data, removing the filtered-divisor scratch arrays while continuing to feed the existing kernel and hit bookkeeping.
9. [done] **Route legacy prime checks through heuristics**
   * [done] Introduce `HeuristicPrimeTester` so heuristic CPU/GPU checks live alongside the legacy `PrimeTester` implementations while callers opt in explicitly.
   * [done] Maintain legacy paths for benchmarking via `PrimeTester.IsPrimeCpu`, `PrimeTester.Exclusive.IsPrimeGpu`, and the explicit `HeuristicPrimeTester` entry points so benchmark harnesses can compare behaviours without runtime switches.
10. **Validation and benchmarking**
    * Create targeted unit tests covering boundary cases for Group A/B transitions, wheel enumerations, residue deltas, and the stop-on-hit behaviour.
    * Expand GPU test harnesses to assert equivalence between CPU/GPU heuristics on sampled composite and prime inputs and to verify the residue-stream contract with by-divisor sessions.
    * Benchmark against historical data (hundreds of thousands of composites) to confirm that heuristic ordering improves early divisor detection and that residue sharing reduces duplicate computation.
    * [done] Build BenchmarkDotNet comparisons for `HeuristicPrimeTester.IsPrimeGpu`, `HeuristicPrimeTester.IsPrimeCpu`, and `Open.Numeric.Primes.Prime.Numbers.IsPrime` across representative ranges (≤100, ≤4,000,000, and ≥138,000,000) before routing heuristic code into production. Implemented in `EvenPerfectBitScanner.Benchmarks/PrimeTesterBenchmarks.cs` to exercise both CPU and GPU heuristic paths alongside the external library.
11. **Documentation and rollout**
    * Update README/docs to explain heuristic prerequisites (odd, not divisible by 5) and how configuration toggles affect execution across prime testing and by-divisor scans.
    * Provide migration guidance for operators toggling between legacy and heuristic modes and describe how telemetry/metrics change after integration.
    * Publish benchmark results in the heuristic specification once measurements are available so the documented guidance reflects the observed CPU/GPU/external comparisons.

## 4. Remaining Work Summary

* Complete the remaining rollout tasks in steps 10–11, including legacy-call-site integration, validation, and documentation updates.
* Develop comprehensive validation suites (unit tests and performance benchmarks) to compare heuristic behaviour with current implementations across both prime testing and by-divisor divisor scans.
* Document configuration switches, runtime metrics, and operational playbooks once the heuristics are in place.
