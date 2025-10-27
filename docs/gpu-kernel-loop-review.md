# GPU Kernel Loop Review

This checklist enumerates every GPU kernel under `PerfectNumbers.Core/Gpu/Kernels`. Each entry captures the loops and branch
constructs that can terminate early when the result is already known, along with justification when a full pass is required.

## DivisorByDivisorKernels

- [x] `CheckKernel`
  - The per-thread loop over the exponent list always finishes because the caller expects the `hits` span to be completely
    populated. The cycle-aware branch only calls `ComputeNextIsUnityGpu` when the reduced remainder is zero and avoids redundant
    modulo work by folding deltas before the comparison.

## DivisorCycleKernels

- [x] `GpuDivisorCycleKernel`
  - No loops; the kernel delegates to `CalculateCycleLengthGpu` for the heavy lifting.
- [x] `CalculateCycleLengthGpu`
  - The `while (true)` ladder exits immediately through the inlined `GpuStep` helper the first time `pow` wraps to one, so no
    further doubling iterations execute once the order is known.
- [x] `GpuAdvanceDivisorCyclesKernel`
  - The `while (currentPow != 1UL)` path only runs for divisors that need the slow fallback and stops as soon as the residue
    cycles back to one. The outer `do { ... } while (--steps != 0);` terminates early with a `return` the moment the order is
    discovered, otherwise it preserves state for the next launch.

## DivisorKernels

- [x] `Kernel`
  - Uses straight-line arithmetic only; every `return` fires as soon as the divisor properties make the answer deterministic.

## IncrementalKernels

- [x] `IncrementalKernelScan`
  - Branches short-circuit on each residue test: the kernel returns immediately when any residue class disqualifies the current
    candidate or the small-cycle table rules it out. No loops beyond those implicit in the residue arithmetic execute in this
    kernel body.
- [x] `IncrementalOrderKernelScan`
  - Shares the same residue-based early exits as the full scan. Once a residue, small-cycle lookup, or power test fails, the
    kernel exits without touching later predicates. It only falls through to the atomic flag when every guard succeeds, so no
    redundant iterations occur.

## KernelMathHelpers

- [x] `Mod128By64`
  - The two 64-iteration loops are required for the bit-by-bit Barrett-style remainder; every iteration feeds the next bit and
    no early termination is possible without losing precision. Branches inside the loops immediately subtract the modulus when
    the accumulator crosses it, preventing extra arithmetic once the bit is processed.
- [x] `CalculateCycleLengthSmall`
  - The `while (pow != 1UL)` loop increments the order until the residue returns to one and then stops; it cannot exit sooner
    because every doubling affects the eventual cycle length.

## LucasLehmerKernels

- [x] `AddSmallKernel`
  - The `for` loop exits as soon as the carry is zero, so limbs past the last mutation are never touched.
- [x] `SubtractSmallKernel`
  - The limb walk stops once the borrow clears, mirroring the addition kernel’s early-out behavior.
- [x] `ReduceModMersenneKernel`
  - Each loop either continues because more folding is required or breaks as soon as the carry/borrow resolves. The inner `while`
    ladders that propagate carries terminate immediately when the next limb no longer wraps.
- [x] `IsZeroKernel`
  - Breaks out on the first non-zero limb so no extra comparisons are performed after a positive result.
- [x] `KernelBatch`
  - The Lucas–Lehmer iteration must run `p - 2` times by definition; there is no earlier conclusion to draw, so the loop spans
    the full range.
- [x] `Kernel`
  - Likewise iterates exactly `p - 2` steps, selecting between the Mersenne-specific and generic paths up front so no redundant
    modulus operations execute once the exponent class is known.

## NttBarrettKernels

- [x] `ScaleBarrett128Kernel`
  - Straight-line arithmetic with nested conditionals. The double subtraction branch executes at most twice, cutting off as soon
    as the reduced value enters `[0, n)`.
- [x] `SquareBarrett128Kernel`
  - Shares the same double-subtraction early exit as the scale kernel.
- [x] `StageBarrett128Kernel`
  - The butterfly is branchless aside from the `if` guard that runs at most two subtractions to fold the product back into range,
    so no iterative structures linger after the modulus comparison.

## NttButterflyKernels

- [x] `StageKernel`
  - Pure butterfly math with a single `%` and no loops; the branchless add/sub sequence completes in one pass per index.

## NttMontgomeryKernels

- [x] `StageMontKernel`
  - No loops; the conditional subtracts the modulus at most once to keep the sum in range and otherwise returns immediately.
- [x] `ToMont64Kernel`, `FromMont64Kernel`, `SquareMont64Kernel`, `ScaleMont64Kernel`
  - Each kernel performs a single Montgomery multiply and stores the result without any repetition or branching beyond the
    subtraction that enforces the modulus bound inside `MontMul64`.

## NttPointwiseKernels

- [x] `MulKernel`
  - No loops; multiplies the matching elements and writes the result.
- [x] `ScaleKernel`
  - Mirror of `MulKernel` with a fixed scalar.

## NttTransformKernels

- [x] `ForwardKernel`
  - The `for (int i = 0; i < length; i++)` loop must visit every source element to accumulate the full convolution term. No
    early exit exists because each product contributes to the final sum.
- [x] `InverseKernel`
  - Same O(n²) loop shape as the forward transform. Every iteration is required to reconstruct the original coefficients before
    the final scaling step.
- [x] `BitReverseKernel`
  - Swaps at most once per index and exits immediately when `i >= j`, so no redundant exchanges occur.

## NumericProbeKernels

- [x] `OpenNumericPrimesIsPrimeKernel`, `BigIntegerKernel`, `EIntegerKernel`, `ERationalKernel`
  - All four kernels are straight-line probes without loops. They branch only to select between literal operations and assign the
    final result immediately.

## OrderKernels

- [x] `OrderKernelScan`
  - This kernel is branch-driven with no loops. Each guard returns instantly once an invalid condition appears, so the modulus
    and GCD work stops as soon as a failure is detected.

## Pow2ModKernels

- [x] `Pow2ModKernelScan`
  - The residue checks return at the first disqualifying branch, and the small prime sweep breaks as soon as either the squared
    prime exceeds the candidate or a factor is detected. The kernel only reaches the assignment after every guard passes.
- [x] `Pow2ModOrderKernelScan`
  - Uses the same residue early exits and small-cycle rejection, returning immediately when a predicate fails so no extra modulo
    work runs on rejected candidates. The atomic flag flip only occurs after all tests succeed.

## Pow2MontgomeryKernels

- [x] `Pow2MontgomeryKernelKeepMontgomery`, `Pow2MontgomeryKernelConvertToStandard`
  - Single-call wrappers around the Montgomery helper with no loops or branching.

## PrimeOrderKernels

- [x] `PartialFactorKernel`
  - The `for` loop stops as soon as the limit or square threshold makes further factors impossible, and it breaks out after
    filling the available slots. Inner residue divisions short-circuit the moment the remaining cofactor no longer divides by the
    current prime.
- [x] `CalculateOrderKernel`
  - Returns immediately for every trivial branch (non-zero status, small `p`, special-case success). Subsequent helper calls only
    run when the prior checks fail, so no extra work remains once the order is confirmed.
- [x] `FactorWithSmallPrimes`
  - Mirrors the partial factor loop with early breaks when the sieve limit or square bound cuts off more factors. The modulo test
    uses `continue` to skip primes that do not divide the remaining cofactor.
- [x] `SortFactors`
  - Standard insertion sort; the inner `while` exits as soon as the sorted order is restored, so each element shifts only until it
    reaches its slot.
- [x] `TrySpecialMaxKernel`
  - The single `for` loop returns `false` on the first factor that still divides `2^phi - 1`; otherwise it finishes after testing
    each candidate once.
- [x] `ExponentLoweringKernel`
  - The outer loop walks each factor once, and the inner loop stops lowering as soon as the order is no longer divisible by the
    factor or a power test fails. When the reduced order stays valid it keeps dividing, otherwise it breaks.
- [x] `TryConfirmOrderKernel`
  - Returns immediately when a prerequisite power test fails, when Pollard factoring cannot complete, or when any lowered order
    still maps `2^k` back to one. All loops are bounded by factor counts and exit as soon as a counterexample appears.
- [x] `TryHeuristicFinishKernel`
  - Every branch guards downstream work: rejected candidates skip the expensive stack expansion, the power-check budget stops
    once `maxPowChecks` is exhausted, and the helper loops (`GenerateCandidates`, `SortCandidatesKernel`, etc.) all return early
    when their limit or stack capacity is hit.
- [x] `GenerateCandidates`
  - The depth-first stack pushes break out when the product would overflow the target order or when the stack capacity is
    reached. Each branch prunes the search tree so no redundant candidate expansion occurs.
- [x] `SortCandidatesKernel`
  - Insertion sort with an early `break` once the prior element is already in order, preventing unnecessary shifts.
- [x] `BuildCandidateKey` and helpers
  - Straight-line scoring logic without loops. Branches collapse to a single return path per classification.
- [x] `TryPreviousOrderKernel`
  - Returns once the previous order verifies or after rejecting it, without scanning further candidates.
- [x] `TryFillGapsKernel`
  - Stops when the candidate list is exhausted or the stack capacity is hit, and the inner loop breaks as soon as a factor fails
    the divisibility or power test.
- [x] `ConsumeCandidateKernel`
  - Returns immediately on the first successful confirmation or any terminal failure flag, so it never continues scanning after a
    decisive outcome.

## PrimeTesterKernels

- [x] `SmallPrimeSieveKernel`
  - The `for` loop breaks on the first square exceeding `n` and returns immediately after finding a dividing prime, eliminating
    extra trial divisions.
- [x] `HeuristicTrialDivisionKernel`
  - Straight-line test with early return for invalid divisors.
- [x] `SharesFactorKernel`
  - The binary GCD loop shortens both operands each iteration and exits when the remainder becomes zero, so no redundant
    iterations survive after the GCD is found.

## SmallPrimeFactorKernels

- [x] `SmallPrimeFactorKernelScan`
  - The sieve loop stops when the prime list runs out, the remaining cofactor is one, or the next prime square exceeds the
    remaining value. After a prime factor no longer divides the cofactor, the inner `do/while` exits immediately.
