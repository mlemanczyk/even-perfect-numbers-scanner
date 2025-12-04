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

# Known factors
The lists below contain known factors identified by the scanner.
* [138,000,001 - 139,000,000](/factors-138m.csv) - in progress

# Excluded without factors
The candidates below were excluded without recording the factor.
* [138,000,001 - 139,000,000](/excluded-138m.csv)

# Description

This document explains **what the program does**, **how it works step-by-step**, and **why** each optimization exists. It reflects the *current* code paths in `Program.cs` and the helper library (`PerfectNumbers.Core`), describing only what the code actually does (comments ignored).

# Supported modes
* `--mersenne=pow2mod`
* `--mersenne=incremental`
* `--mersenne=lucas`
* `--mersenne=residue`

# EvenPerfectBitScanner – Configuration and Pipeline
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
  `--increment=add` → the producer uses `TransformPAdd` (see my `Program.cs`), which locks into `p ≡ 1 or 5 (mod 6)`.

* **Prime tests (p)**:
  `--primes-device=gpu` → `PrimeTester` runs a GPU-batched small-prime sieve (and light CPU checks) (`PrimeTester.cs`).

* **Mersenne stage**:
  `--mersenne=residue` + `--mersenne-device=gpu` → `MersenneNumberTester` selects the residue divisor path and routes its scanning kernels to GPU (`MersenneNumberTester.cs` + `Gpu/MersenneNumberResidueGpuTester.cs`). We do not run Lucas–Lehmer in this config.

* **GPU tuning**:
  `--gpu-prime-threads=1024` gates concurrent GPU prime work via `GpuPrimeWorkLimiter`.
  `--gpu-prime-batch=512000` sets the batch size in `PrimeTester`.

* **I/O**:
  `--write-batch-size=10000` batches CSV writes.

---

## The pipeline, line-by-line logic of each stage (and every early rejection)

I’ll walk you through what happens for a single candidate exponent `p`. The exact code paths below are the ones used in my run.

### Candidate acquisition (producer)

* We atomically reserve the next `p` using a packed `ulong` CAS over `{p << 3 | remainder}` (my `_state`). That’s the lock-free distributor in `Program.cs`.

* **TransformPAdd**:
  First step normalizes to `p % 6 ∈ {1,5}` and after that it fixes remainders to 1 or 5 with each increase.

### Cheap filters on p

(All in `PrimeTester.cs` unless noted)

* **Trivial rejects**:

  * `n <= 3`: accept 2 and 3 only.
  * even `n`: reject.
  * `n % 5 == 0`: reject quickly.

* **“Ends-with-1” GCD heuristic**:
  If `n % 10 == 1` and `SharesFactorWithMaxExponent(n)` → reject.

  `SharesFactorWithMaxExponent(n)` computes `m = floor(log2 n)` and rejects if `gcd(n, m) ≠ 1`.
  There’s also a GPU batch form `SharesFactorWithMaxExponentBatch` used elsewhere when screening arrays.

  *Why it’s valid*: numbers of the form `10t+1` have specific small-factor structure; combining that with `gcd(n, floor(log2 n))` cheaply kills many composites without MR. (This is a heuristic prefilter based on my analysis of primes ending with 1 or 7 and testing; it won’t reject primes incorrectly.)

* **Small-prime sieve (CPU loop, still inside PrimeTester)**:
  Iterate `PrimesGenerator.SmallPrimes` while `prime^2 ≤ n` using the ready-made `SmallPrimesPow2` to avoid sqrt. Reject immediately if divisible.

  With `--primes-device=gpu`, we also have a device-side sieve for batches (`IsPrimeGpuBatch`): it copies chunks to the device, runs a simple sieve kernel over an uploaded small-primes table, and copies flags back. My per-candidate path uses the single-value version, but the machinery (stateful kernel cache, scratch buffers) is there for vectorized checks.

➡ If `p` fails any of the above → we never touch `M_p`. That’s my first wall of early rejections.

### Mersenne stage

(All driven by `MersenneNumberTester.cs`)

* **Hard mathematical pre-rejects (top of IsMersennePrime)**:

  * If `p % 3 == 0` and `p != 3` → reject (since `7 ∣ M_p` when `3 ∣ p`).
  * If `(p & 3) == 1` and `p.SharesFactorWithExponentMinusOne()` → reject.

    * `SharesFactorWithExponentMinusOne` (in `ULongExtensions.cs`) factors `p-1` by trial over `SmallPrimes`, removes multiplicities, and for each distinct prime `r` checks whether the multiplicative order of 2 modulo `r` divides `p` (via `CalculateOrder()`), also handling the leftover cofactor. If any such order divides `p`, we can certify a small factor for `M_p` and reject `p` outright.
  * If `p` is even → reject (trivial, here `p` is already odd from the prime screen, but it’s kept for safety).

* **Residue divisor scanning (`_useResidue == true`)**:

  * We set `twoP = 2*p`. Valid Mersenne divisors have the form `q = 2kp + 1, k ≥ 1` with congruence constraints `q ≡ 1 (mod 2p)` and `q ≡ 1 or 7 (mod 8)`.
  * **Digit filter**:

    * Compute `lastIsSeven = ((p & 3) == 3)` which encodes the last decimal digit of `M_p = 2^p - 1`.

    * If last digit of `M_p` is 7 → we allow only `q` with last digit in `{3, 7, 9}`.

    * Else (last digit 1) → we allow only `{1, 3, 7, 9}`.
      This is implemented both on CPU (`CpuConstants.LastSevenMask10`, `LastOneMask10`) and GPU by calculating `q % 10` incrementally as `k` grows.

  * **Rolling residues**: we keep four rolling residues for each lane: `q mod 10, 8, 5, 3`. That allows immediate rejection of `q` divisible by 2, 3, or 5 without any division (just adds, bit-ands and subtracts).

  * **CPU residue scanner**: uses `ModResidueTracker` with update formula `r′ ≡ 2^Δ⋅r+(2^Δ−1) (mod d)`.

  * **GPU residue scanner**: batches `k`, evaluates residues, prunes with cycles table, runs mod-pow on survivors. Uses pooled contexts and kernel caches.

  * **Short-circuit**: as soon as we find any valid divisor `q | M_p` in the scanned window, we stop.

➡ If scanning completes up to `k 2^p / 2 + 1` and no divisor was found → `IsMersennePrime(p)` returns true and the CSV line has `isPerfect=true`. This means that `p` passed all the tests and is a **candidate** for constructing a perfect number, which requires testing with other methods.

---

## What we cache, pool, and batch (and why)

* **Per-thread singletons**: `PrimeTester`, `MersenneNumberTester`, `ModResidueTracker` → avoid hot locks, reuse GPU/CPU state.

* **Mod residue tracker**: avoids recomputing residues from scratch, supports incremental updates.

* **Divisor cycles**: persistent, precomputed orders for `divisors ≤ 4,000,000`, shared singleton, uploaded to GPU.

* **GPU infra**: pooled contexts/kernels, concurrency gating.

* **I/O pooling**: `StringBuilderPool` + batched writes.

---

## Every mathematical rule that we’re using to reject early

* Only prime exponents matter.
* 7 divides `M_p` if `3 ∣ p`.
* Order-based rejections using factors of `p−1`.
* Shape of Mersenne divisor: `q=2kp+1`, `q≡1 or 7 (mod 8)`.
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

* **Per-thread buffers are huge**: `--block-size=10_000_000` → \~80 GB at 1024 threads.

* **1024 CPU threads + 1024 GPU permits** oversubscribe virtually any GPU. When running such configurations, make sure that you have enough RAM or swap file.

  * How: set `--threads` to `Environment.ProcessorCount` and `--gpu-prime-threads` to \~8–32.

---

## “Full accounting” of what gets computed, rejected, cached, or pooled

### Per p

* **Computed**: next `p`, small-prime residues, heuristic gcd if ending in 1, trial division.
* **Rejected**: trivial composite checks; small-prime hits.
* **If p prime**: pre-rules, order checks, residue scan inputs, k-batches.

### Cached/pooled

* **Thread scope**: testers, trackers, GPU contexts, buffers.
* **Process scope**: divisor cycles dataset, small primes.
* **IO**: pooled `StringBuilder` + batched writes.

---