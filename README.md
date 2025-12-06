# License
You're allowed to use the source code for education purposes, only.
* You can download it,
* You can review it,
* You can run it to learn how it works,
* You can run it to learn if it works,
* You can use it for learning how certain operations can be done, optimized etc.,
* You can use it to search and announce new perfect numbers, but please always include the information that you used my scanner. I have family and I need to bring money to the house.

\
What I'm asking you **not** to do, without license:

* You're **not** allowed to copy / paste or integrate my code and / or assemblies with your code.
* You're **not** allowed to replicate it 1 : 1 in your project. Please write your own implementation with the understanding how it works. We may end up with the same solution.


### **Please play fair and don't use it to claim your success in the area.**
\
\
Any other use requires the payment of the licensing fee, which can be done using the links below. The payment is one-time payment and includes all future updates, if any.

In case of any concerns or questions, please feel free to reach out using my [LinkedIn account](https://www.linkedin.com/in/marek-lemanczyk-432819154).



* Personal License - $10
https://buymeacoffee.com/mlemanczyk/e/455507

* Earning / Commercial License - $100
https://buymeacoffee.com/mlemanczyk/e/455508

[![Donate](buy-me-a-coffee.svg)](https://www.buymeacoffee.com/mlemanczyk)

# Thirdparty Components
The source code relays on selected functionality from the following packages:
* [FluentAssertions](https://www.nuget.org/packages/FluentAssertions)
* [ILGPU](https://www.nuget.org/packages/ILGPU)
* [ILGPU.Algorithms](https://www.nuget.org/packages/ILGPU.Algorithms)
* [Open.Numeric.Primes](https://www.nuget.org/packages/Open.Numeric.Primes)
* [PeterO.Numbers](https://www.nuget.org/packages/PeterO.Numbers)
* [xunit](https://www.nuget.org/packages/xunit)

# Excluded p candidates by `--mersenne=residue` method.
Below you can find the filtering results coming from `--mersenne=residue` method, sorted and filtered to primes, only.
* [138,000,001 - 200,000,000](/sorted-primes-from-residue-method.zip)

# Excluded p candidates by `--mersenne=bydivisor` method.
The candidates below were excluded by the scanner. All calculated on AMD Ryzen 7 laptop. The divisors are randomly checked as sanity. The list is due complete validation when all candidates within each range are calculated.
* [138,000,001 - 139,000,000](/excluded-138m.csv)

# Description

This document explains **what the program does**, **how it works step-by-step**, and **why** each optimization exists. It reflects the *current* code paths in `Program.cs` and the helper library (`PerfectNumbers.Core`), describing only what the code actually does (comments ignored).

# Supported modes
* `--mersenne=pow2mod`
* `--mersenne=incremental`
* `--mersenne=lucas`
* `--mersenne=residue`
* `--mersenne=divisor`
* `--mersenne=bydivisor`

## Two-stage scanning flow

* Stage 1 - `--mersenne=residue` acts as the fast, coarse filter for entire ranges of candidate exponents. It quickly rules out most `p` values before any deep work, producing result files that record which primes survived.
* Stage 2 - `--mersenne=bydivisor` consumes the survivors from stage 1 via `--filter-p=<results>` and performs the expensive divisor-by-divisor scan. It will not run without that filter.
* Sample stage-1 output: `[sorted-primes-from-residue-method.zip](sorted-primes-from-residue-method.zip)` contains primes that passed the residue sweep and can seed stage 2 directly.
* Heuristic order/residue rules reused by stage 2 are documented in `docs/HeuristicIsPrimeSpecification.md`; the by-divisor path reuses those wheel priorities, residue steppers, and cycle heuristics when walking `q = 2kp + 1`.

# EvenPerfectBitScanner - Configuration and Pipeline
Please run `EvenPerfectBitScanner.exe /?` to display all command line parameters with a short description. The description below describes `--mersenne=residue` mode.

## What the exact CLI config does

```bash
./EvenPerfectBitScanner.exe
--increment=add
--mersenne=residue
--threads=1024
--primes-device=gpu
--mersenne-device=gpu
--block-size=10000000
--gpu-prime-threads=1024
--write-batch-size=10000
--gpu-prime-batch=512000
````

* **Candidate stepping**:
  `--increment=add` -> the producer uses `TransformPAdd` (see my `Program.cs`), which locks into `p == 1 or 5 (mod 6)`.

* **Prime tests (p)**:
  `--primes-device=gpu` -> `PrimeTester` runs a GPU-batched small-prime sieve (and light CPU checks) (`PrimeTester.cs`).

* **Mersenne stage**:
  `--mersenne=residue` + `--mersenne-device=gpu` -> `MersenneNumberTester` selects the residue divisor path and routes its scanning kernels to GPU (`MersenneNumberTester.cs` + `Gpu/MersenneNumberResidueGpuTester.cs`). We do not run Lucas-Lehmer in this config.

* **GPU tuning**:
  `--gpu-prime-threads=1024` gates concurrent GPU prime work via `GpuPrimeWorkLimiter`.
  `--gpu-prime-batch=512000` sets the batch size in `PrimeTester`.

* **I/O**:
  `--write-batch-size=10000` batches CSV writes.

---

## The pipeline, line-by-line logic of each stage (and every early rejection)

I'll walk you through what happens for a single candidate exponent `p`. The exact code paths below are the ones used in my run.

### Candidate acquisition (producer)

* We atomically reserve the next `p` using a packed `ulong` CAS over `{p << 3 | remainder}` (my `_state`). That's the lock-free distributor in `Program.cs`.

* **TransformPAdd**:
  First step normalizes to `p % 6 in {1,5}` and after that it fixes remainders to 1 or 5 with each increase.

### Cheap filters on p

(All in `PrimeTester.cs` unless noted)

* **Trivial rejects**:

  * `n <= 3`: accept 2 and 3 only.
  * even `n`: reject.
  * `n % 5 == 0`: reject quickly.

* **"Ends-with-1" GCD heuristic**:
  If `n % 10 == 1` and `SharesFactorWithMaxExponent(n)` -> reject.

  `SharesFactorWithMaxExponent(n)` computes `m = floor(log2 n)` and rejects if `gcd(n, m) != 1`.
  There's also a GPU batch form `SharesFactorWithMaxExponentBatch` used elsewhere when screening arrays.

  *Why it's valid*: numbers of the form `10t+1` have specific small-factor structure; combining that with `gcd(n, floor(log2 n))` cheaply kills many composites without MR. (This is a heuristic prefilter based on my analysis of primes ending with 1 or 7 and testing; it won't reject primes incorrectly.)

* **Small-prime sieve (CPU loop, still inside PrimeTester)**:
  Iterate `PrimesGenerator.SmallPrimes` while `prime^2 <= n` using the ready-made `SmallPrimesPow2` to avoid sqrt. Reject immediately if divisible.

  With `--primes-device=gpu`, we also have a device-side sieve for batches (`IsPrimeGpuBatch`): it copies chunks to the device, runs a simple sieve kernel over an uploaded small-primes table, and copies flags back. My per-candidate path uses the single-value version, but the machinery (stateful kernel cache, scratch buffers) is there for vectorized checks.

-> If `p` fails any of the above -> we never touch `M_p`. That's my first wall of early rejections.

### Mersenne stage

(All driven by `MersenneNumberTester.cs`)

* **Hard mathematical pre-rejects (top of IsMersennePrime)**:

  * If `p % 3 == 0` and `p != 3` -> reject (since `7 | M_p` when `3 | p`).
  * If `(p & 3) == 1` and `p.SharesFactorWithExponentMinusOne()` -> reject.

    * `SharesFactorWithExponentMinusOne` (in `ULongExtensions.cs`) factors `p-1` by trial over `SmallPrimes`, removes multiplicities, and for each distinct prime `r` checks whether the multiplicative order of 2 modulo `r` divides `p` (via `CalculateOrder()`), also handling the leftover cofactor. If any such order divides `p`, we can certify a small factor for `M_p` and reject `p` outright.
  * If `p` is even -> reject (trivial, here `p` is already odd from the prime screen, but it's kept for safety).

* **Residue divisor scanning (`_useResidue == true`)**:

  * We set `twoP = 2*p`. Valid Mersenne divisors have the form `q = 2kp + 1, k >= 1` with congruence constraints `q == 1 (mod 2p)` and `q == 1 or 7 (mod 8)`.
  * **Digit filter**:

    * Compute `lastIsSeven = ((p & 3) == 3)` which encodes the last decimal digit of `M_p = 2^p - 1`.

    * If last digit of `M_p` is 7 -> we allow only `q` with last digit in `{3, 7, 9}`.

    * Else (last digit 1) -> we allow only `{1, 3, 7, 9}`.
      This is implemented both on CPU (`CpuConstants.LastSevenMask10`, `LastOneMask10`) and GPU by calculating `q % 10` incrementally as `k` grows.

  * **Rolling residues**: we keep four rolling residues for each lane: `q mod 10, 8, 5, 3`. That allows immediate rejection of `q` divisible by 2, 3, or 5 without any division (just adds, bit-ands and subtracts).

  * **CPU residue scanner**: uses `ModResidueTracker` with update formula `r' == 2^Delta*r+(2^Delta-1) (mod d)`.

  * **GPU residue scanner**: batches `k`, evaluates residues, prunes with cycles table, runs mod-pow on survivors. Uses pooled contexts and kernel caches.

  * **Short-circuit**: as soon as we find any valid divisor `q | M_p` in the scanned window, we stop.

-> If scanning completes up to `k 2^p / 2 + 1` and no divisor was found -> `IsMersennePrime(p)` returns true and the CSV line has `isPerfect=true`. This means that `p` passed all the tests and is a **candidate** for constructing a perfect number, which requires testing with other methods.

## --mersenne=bydivisor: divisor-first sweep for filtered primes

This is the second-stage scan that only runs after a residue pass. Feed it the stage-1 results (for example from `sorted-primes-from-residue-method.zip`) via `--filter-p=<path>` so it can focus solely on primes that already cleared the residue filter.

* **Prerequisites and setup**: `--filter-p` is mandatory (skips execution otherwise), candidates come from prior results filtered to `passedAllTests=true`, and the list is sorted, deduplicated, and optionally trimmed by `--start-prime` and `--max-prime`. No fresh prime sieve runs: the path assumes the filter already contains prime exponents. `--min-k` seeds the scan window and per-prime state is persisted under `Checks/<p>.bin` (CPU path only) every `ByDivisorStateSaveInterval` to resume after restarts.

* **Divisor generation and residue stepper**:
  * Divisors follow `q = 2kp + 1` with monotonically increasing integer `k`, bounded by `allowedMax = min(2^(p-1)-1, divisorLimitFromMaxPrime)`. The limit comes from the largest prime in the batch (`ConfigureFromMaxPrime`) and saturates at 2^256-1 for huge inputs.
  * Decimal masks come from my composite-divisor residue study: when `(p & 3) == 3` the Mersenne number ends with 7, so only divisors with last digit `{3,7,9}` pass; otherwise `{1,3,7,9}`. All divisors must satisfy `q % 8 in {1,7}`.
  * Rolling remainders are maintained for each small modulus instead of recomputing divisions: CPU checks mod {3,7,11,13,17,19,8,10}, GPU checks mod {3,7,11,8,10}. The stride `step = 2p` is pre-split into per-modulus deltas (`step10`, `step8`, ...) and residues advance via branch-free `AddMod*` helpers, so the stepper cheaply skips entire residue classes that never yield valid divisors.

* **CPU scan path**:
  * Fast lane when `k`/`q` fit in 64 bits: `GpuUInt128` holds stride, divisor, and limit; residues drive the wheel, `MontgomeryDivisorDataPool` supplies reduction constants, and every admissible divisor asks `MersenneDivisorCycles.TryCalculateCycleLengthForExponent*` for the order of 2 modulo `q`. If the order equals `p`, the divisor is accepted immediately; otherwise the code falls back to `CalculateCycleLength*` (CPU/GPU/hybrid based on `--order-device`) and short-circuits on the same condition. The scan records progress (`RecordState`) every 10k steps so it can resume from the last `k`.
  * When bounds overflow 64 bits, the loop switches to `BigInteger`: the same residue masks apply, composites are rejected with a Miller-Rabin check before the expensive `BigInteger.ModPow(2, p, q)`, and hits still require `2^p mod q == 1`. `divisorsExhausted` reports whether the configured window was fully covered (including resumes from `--min-k`).

* **GPU scan path**:
  * Requires `MinK <= ulong.MaxValue`; allowedMax is `min(2^(p-1)-1, divisorLimitFromMaxPrime)` with an extra 170M cap for tiny test primes. `CheckDivisors` chunks the `k` range, applies the same decimal mask and small-modulus filters on CPU, and asks `ResolveDivisorCycle` (GPU order solver with CPU fallback) for each admissible divisor. A cycle equal to `p` ends the scan immediately.
  * Survivors are batched into pooled buffers and pushed to `DivisorByDivisorKernels.CheckKernel`. The kernel reuses `ExponentRemainderStepperGpu` plus cycle remainders to avoid recomputing full powers for consecutive exponents (counts are 1 today but the stepper is shared). Hits are captured via an atomic `firstHit` flag; accelerator-local bags reuse streams, buffers, and even divisor scan sessions. Unlike the CPU path, the GPU path does not persist state files; resuming relies solely on `--min-k`.

* **Decision flags**: `IsPrime` returns `(isPrime, divisorsExhausted, divisor)`. The main loop treats `divisorsExhausted=true` as a detailed check; a composite divisor sets `isPrime=false` and stops the scan as soon as it is confirmed.

For the residue wheels, digit masks, and the heuristic order calculation that drive this stage, see `docs/HeuristicIsPrimeSpecification.md` (statistics from the composite study remain unchanged even as the code evolved).



## --mersenne=divisor: cycle-gated GPU sweep

* **Where it plugs in**: shares the same producer, residue locks (p % 6 is 1 or 5), and prefilters as the `--mersenne=residue` description above. Once `--mersenne=divisor` is selected, the Mersenne stage delegates to `_divisorTester.IsPrime` and skips Lucas/residue kernels.

* **Cached small-divisor stage**: `MersenneNumberDivisorGpuTester.BuildDivisorCandidates` lifts all divisors <= 4,000,000 with known cycle lengths from the `divisor_cycles` snapshot (generated from my composite-divisor residue analysis). Each cached `(divisor, cycle)` is tried only when `p % cycle == 0`, cutting out residue groups whose order of 2 provably cannot divide `p`.

* **Generated divisor sweep**:
  * If a specific divisor is supplied (tests/manual runs), it is checked directly via `DivisorKernels.Kernel`.
  * Otherwise the scanner walks `k = 1..divisorCyclesSearchLimit` (`--divisor-cycles-limit`, default 64), building `q = 2kp + 1` and skipping the trivial `q == M_p` case for small `p`. For each `q`, `MersenneDivisorCycles.GetCycleGpu` computes the order of 2 (using the same heuristic factorization/cycle grouping as the snapshot); only when `p % cycle == 0` do we launch the GPU divisibility kernel. The first hit stops the loop.
  * `divisorsExhausted` flips to true only if the configured limit covers all admissible `k` up to `floor(UInt128.MaxValue / (2p))`, so callers know whether the divisor window was fully searched.

* **GPU resource handling**: per-accelerator queues reuse the one-byte result buffers used by `DivisorKernels`, and kernels run through `AcceleratorPool` without CPU fallbacks on this path (the exponent primality screen already ran on CPU/GPU earlier). Cycle data are loaded up front from `--divisor-cycles` for both the cached and generated phases.

* **Early-rejection rules specific to this mode**: Only divisors of shape `2kp+1` whose order divides `p` ever hit the GPU. The small-cycle snapshot already bakes in the decimal-digit and residue exclusions observed across millions of composite divisors, so the runtime sweep focuses exclusively on the most promising groups.


---

## What we cache, pool, and batch (and why)

* **Per-thread singletons**: `PrimeTester`, `MersenneNumberTester`, `ModResidueTracker` -> avoid hot locks, reuse GPU/CPU state.

* **Mod residue tracker**: avoids recomputing residues from scratch, supports incremental updates.

* **Divisor cycles**: persistent, precomputed orders for `divisors <= 4,000,000`, shared singleton, uploaded to GPU.

* **GPU infra**: pooled contexts/kernels, concurrency gating.

* **I/O pooling**: `StringBuilderPool` + batched writes.

---

## Every mathematical rule that we're using to reject early

* Only prime exponents matter.
* 7 divides `M_p` if `3 | p`.
* Order-based rejections using factors of `p-1`.
* Shape of Mersenne divisor: `q=2kp+1`, `q==1 or 7 (mod 8)`.
* Cycle length filter for small `divisors`.
* Monotone residue stepping with closed forms.

---

## Hot-path micro-optimizations already in the code

* Lock-free work distribution.
* Per-thread testers & trackers.
* Sqrt-free stopping in sieves.
* Device-side constant tables.

---

## Risks, correctness gaps, and performance pitfalls (with precise fixes)

* **Per-thread buffers are huge**: `--block-size=10_000_000` -> \~80 GB at 1024 threads.

* **1024 CPU threads + 1024 GPU permits** oversubscribe virtually any GPU. When running such configurations, make sure that you have enough RAM or swap file.

  * How: set `--threads` to `Environment.ProcessorCount` and `--gpu-prime-threads` to \~8-32.

---

## "Full accounting" of what gets computed, rejected, cached, or pooled

### Per p

* **Computed**: next `p`, small-prime residues, heuristic gcd if ending in 1, trial division.
* **Rejected**: trivial composite checks; small-prime hits.
* **If p prime**: pre-rules, order checks, residue scan inputs, k-batches.

### Cached/pooled

* **Thread scope**: testers, trackers, GPU contexts, buffers.
* **Process scope**: divisor cycles dataset, small primes.
* **IO**: pooled `StringBuilder` + batched writes.

---